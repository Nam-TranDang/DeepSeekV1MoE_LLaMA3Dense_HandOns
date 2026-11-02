"""GPT hiện đại nhỏ gọn: RMSNorm, RoPE, GQA, SwiGLU, và MoE (shared + routed expert).

Thiết kế để học: mọi thứ nằm trong 1 file, dễ thay từng phần bằng kernel Triton sau này.

Full LLM Architecture - File

RMSNorm, RoPE (Rotary Embeddings), GQA (Grouped Query Attention), SwiGLU, and MoE (Shared + Routed
xperts with auxiliary load-balancing loss and router z-loss). Weight-tied embeddings and LM head.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 16384
    n_layer: int = 12
    d_model: int = 512
    n_head: int = 8
    n_kv_head: int = 2
    block_size: int = 1024
    ffn_hidden: int = 1024      # hidden của MỖI expert
    # MoE
    moe: bool = True
    n_expert: int = 8
    top_k: int = 2
    n_shared: int = 1
    n_dense_first: int = 1      # số layer đầu giữ FFN dense (ổn định hơn)
    aux_coef: float = 0.01      # load-balancing loss
    z_coef: float = 1e-3        # router z-loss
    attn_impl: str = "sdpa"     # sau này thêm "triton"

    @property
    def dense_hidden(self):
        # dense baseline có active params ~ MoE: (top_k + n_shared) * ffn_hidden
        return (self.top_k + self.n_shared) * self.ffn_hidden


PRESETS = {
    # chạy vài phút, để kiểm tra pipeline
    "tiny": dict(n_layer=4, d_model=256, n_head=4, n_kv_head=2, block_size=256,
                 ffn_hidden=512, n_expert=4, top_k=2, n_shared=1, n_dense_first=1),
    # ~190M total, ~75M active (MoE)
    "small": dict(n_layer=12, d_model=512, n_head=8, n_kv_head=2, block_size=1024,
                  ffn_hidden=1024, n_expert=8, top_k=2, n_shared=1, n_dense_first=1),
}


# ---------------------------------------------------------------- building blocks
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return xf.type_as(x) * self.w


def rope_cache(T, head_dim, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    f = torch.outer(torch.arange(T).float(), inv)  # (T, hd/2)
    return f.cos(), f.sin()


def apply_rope(x, cos, sin):
    # x: (B, H, T, hd)
    T = x.shape[2]
    cos, sin = cos[None, None, :T], sin[None, None, :T]
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).type_as(x)


def sdpa_attn(q, k, v):
    # PyTorch tự chọn flash kernel. Đây là BASELINE để so với Triton kernel của bạn.
    return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)


ATTN_IMPLS = {"sdpa": sdpa_attn}  # ATTN_IMPLS["triton"] = my_triton_attn  (cùng signature q,k,v)


class Attention(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.nh, self.nkv = c.n_head, c.n_kv_head
        self.hd = c.d_model // c.n_head
        self.q = nn.Linear(c.d_model, self.nh * self.hd, bias=False)
        self.k = nn.Linear(c.d_model, self.nkv * self.hd, bias=False)
        self.v = nn.Linear(c.d_model, self.nkv * self.hd, bias=False)
        self.o = nn.Linear(self.nh * self.hd, c.d_model, bias=False)
        self.impl = c.attn_impl

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.nh, self.hd).transpose(1, 2)
        k = self.k(x).view(B, T, self.nkv, self.hd).transpose(1, 2)
        v = self.v(x).view(B, T, self.nkv, self.hd).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = ATTN_IMPLS[self.impl](q, k, v)
        return self.o(y.transpose(1, 2).reshape(B, T, -1))


class SwiGLU(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.w1 = nn.Linear(d, h, bias=False)
        self.w3 = nn.Linear(d, h, bias=False)
        self.w2 = nn.Linear(h, d, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# ---------------------------------------------------------------- MoE
class MoE(nn.Module):
    """Top-k routed experts + shared experts, dropless.

    Dispatch hiện tại: sort token theo expert rồi lặp từng expert (baseline đơn giản).
    Đây là chỗ để thay bằng grouped GEMM bằng Triton ở giai đoạn sau.
    """

    def __init__(self, c: Config):
        super().__init__()
        self.E, self.top_k = c.n_expert, c.top_k
        self.router = nn.Linear(c.d_model, c.n_expert, bias=False)
        self.experts = nn.ModuleList(SwiGLU(c.d_model, c.ffn_hidden) for _ in range(c.n_expert))
        self.shared = nn.ModuleList(SwiGLU(c.d_model, c.ffn_hidden) for _ in range(c.n_shared))

    def forward(self, x):
        B, T, D = x.shape
        xf = x.reshape(-1, D)
        N, E, k = xf.shape[0], self.E, self.top_k

        logits = self.router(xf).float()                    # (N, E)
        probs = logits.softmax(-1)
        topv, topi = probs.topk(k, dim=-1)                  # (N, k)
        topv = topv / topv.sum(-1, keepdim=True)            # renormalize

        # --- aux losses
        counts = torch.bincount(topi.reshape(-1), minlength=E)       # (E,)
        f = counts.float() / (N * k)                                 # tỉ lệ assignment (không có grad)
        P = probs.mean(0)                                            # xác suất router trung bình
        aux = E * (f * P).sum()                                      # =1 khi cân bằng hoàn hảo
        z = torch.logsumexp(logits, dim=-1).pow(2).mean()

        # --- dispatch
        flat_e = topi.reshape(-1)
        flat_w = topv.reshape(-1)
        tok = torch.arange(N, device=x.device).repeat_interleave(k)
        order = flat_e.argsort()
        tok, flat_w = tok[order], flat_w[order]

        out = torch.zeros_like(xf)
        start = 0
        for e, cnt in enumerate(counts.tolist()):           # .tolist() gây sync GPU->CPU (chậm, sẽ tối ưu sau)
            if cnt == 0:
                continue
            idx = tok[start:start + cnt]
            w = flat_w[start:start + cnt, None]
            y = self.experts[e](xf[idx])
            out.index_add_(0, idx, (y * w.to(y.dtype)).to(out.dtype))
            start += cnt

        for s in self.shared:
            out = out + s(xf).to(out.dtype)
        return out.view(B, T, D), aux, z, counts


class Block(nn.Module):
    def __init__(self, c: Config, use_moe: bool):
        super().__init__()
        self.n1, self.n2 = RMSNorm(c.d_model), RMSNorm(c.d_model)
        self.attn = Attention(c)
        self.use_moe = use_moe
        self.ffn = MoE(c) if use_moe else SwiGLU(c.d_model, c.dense_hidden)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin)
        if self.use_moe:
            y, aux, z, cnt = self.ffn(self.n2(x))
        else:
            y, aux, z, cnt = self.ffn(self.n2(x)), x.new_zeros(()), x.new_zeros(()), None
        return x + y, aux, z, cnt


class GPT(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.tok = nn.Embedding(c.vocab_size, c.d_model)
        self.blocks = nn.ModuleList(
            Block(c, use_moe=(c.moe and i >= c.n_dense_first)) for i in range(c.n_layer))
        self.norm = RMSNorm(c.d_model)
        self.head = nn.Linear(c.d_model, c.vocab_size, bias=False)
        self.head.weight = self.tok.weight  # weight tying
        cos, sin = rope_cache(c.block_size, c.d_model // c.n_head)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        self.apply(self._init)
        for n, p in self.named_parameters():  # scaled init cho projection ghi vào residual
            if n.endswith("o.weight") or n.endswith("w2.weight"):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * c.n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def num_params(self):
        total = sum(p.numel() for p in self.parameters())
        active = total
        for b in self.blocks:
            if b.use_moe:
                per = sum(p.numel() for p in b.ffn.experts[0].parameters())
                active -= (b.ffn.E - b.ffn.top_k) * per
        return total, active

    def forward(self, idx, targets=None):
        x = self.tok(idx)
        aux = z = x.new_zeros(())
        counts = []
        for b in self.blocks:
            x, a, zz, cnt = b(x, self.cos, self.sin)
            if cnt is not None:
                aux, z = aux + a, z + zz
                counts.append(cnt)
        logits = self.head(self.norm(x))
        out = {"logits": logits, "counts": torch.stack(counts) if counts else None}
        if targets is not None:
            n_moe = max(len(counts), 1)
            ce = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.view(-1))
            out["ce"] = ce
            out["aux"], out["z"] = aux / n_moe, z / n_moe
            out["loss"] = ce + self.c.aux_coef * out["aux"] + self.c.z_coef * out["z"]
        return out