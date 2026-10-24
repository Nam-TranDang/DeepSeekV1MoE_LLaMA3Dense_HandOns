"""DeepSeek-MoE Architecture - inherited, reimplement DeepSeekMoE 16B v1 

Arvix: DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models

 ### Key Architectural Signatures Confirming DeepSeekMoE v1

  1. Shared Expert Isolation + Fine-Grained Routed Experts (DeepSeekMoE):
      • 1 Isolated Shared Expert (n_shared_experts = 1): Always receives all tokens unrouted to capture common
      knowledge across tokens.
      • Top-2 Routed Experts (n_routed_experts = 8, num_experts_per_tok = 2): Tokens are dispatched only to
      specialized sub-experts.
      • Fine-grained intermediate dimension (moe_intermediate_size = 1024), where dense layer size equals (K_{routed} +
      K_{shared}) × moe_intermediate_size.
  2. Dense First Layer for Training Stability (DeepSeekDecoderLayer / DeepSeekConfig.n_dense_first):
      • Layer 0 stays as a standard dense DeepSeekMLP (use_moe = False), while layers 1–11 use MoE blocks. This is a
      design introduced by the DeepSeekMoE paper to stabilize early representation learning.
  3. Auxiliary Load-Balancing & Router z-Loss (DeepSeekMoE.forward):
      • Uses the classical auxiliary balancing loss aux = E ∑ᵢ₌₁ᴱ(fᵢ · Pᵢ) combined with Router z-loss (mean(log ∑
      e^{logits})²) to prevent router logit explosion.
  4. Standard Grouped-Query Attention (GQA) (DeepSeekAttention):
      • Uses conventional GQA (4:1 query-to-KV head ratio, q_proj, k_proj, v_proj, o_proj) with apply_rotary_emb and
      DeepSeekRMSNorm.




Components:
- RMSNorm
- RoPE (Rotary Position Embeddings)
- Grouped-Query Attention (GQA) with PyTorch SDPA
- DeepSeek-style MoE:
  - Isolated Shared Expert(s) + Top-K Routed Experts (SwiGLU)
  - Auxiliary Load-Balancing Loss + Router z-Loss
  - First layer(s) keep dense FFN for training stability
- Weight-tied Embedding and LM Head


  • Isolated MoE Architecture: Shared expert + Top-2 routed experts + Load-balancing
  auxiliary loss + Router z-loss.
  • Classes: DeepSeekConfig, DeepSeekAttention, DeepSeekMLP, DeepSeekMoE,
  DeepSeekDecoderLayer, and DeepSeekForCausalLM.
  • Presets:
      • "tiny": 4 layers, 4 routed experts, 1 shared (~15M params).
      • "small": 12 layers, 8 routed experts, 1 shared (~190M total / 75M active params).

   Components                     | DeepSeek-MoE Small (Sparse)
  --------------------------------|-------------------------------
   Embedding                      | Vocab 16, 384 × 512 = 𝟖.𝟑𝟗𝐌
   Number of Layers               | 12 Layers (1 Dense đầu + 11 MoE)
   Attention / layer              | GQA: 𝟎.𝟔𝟔𝐌
   FFN / layer                    | Layer 0: 4.72M (Dense)Layer 1–11:14.16M (MoE)
   LM Head                        | 0 (Weight-tied với Embedding)
   Total Params                   | 176,742,912 (~176.7M)
   Active token/tok               | 72,933,888 (~72.9M)

"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DeepSeekConfig:
    vocab_size: int = 16384
    n_layer: int = 12
    d_model: int = 512
    n_head: int = 8
    n_kv_head: int = 2                  # 4:1 Query:KV ratio (GQA)
    block_size: int = 1024
    moe_intermediate_size: int = 1024   # Hidden dim of each expert
    n_routed_experts: int = 8           # Total routed experts (E)
    num_experts_per_tok: int = 2        # Top-K routed experts active per token
    n_shared_experts: int = 1           # Shared expert(s) always active (DeepSeek style)
    n_dense_first: int = 1              # First N layers remain dense FFN for stability
    aux_loss_coef: float = 0.01         # Load-balancing auxiliary loss coefficient
    router_z_loss_coef: float = 1e-3    # Router logits regularization (z-loss)
    rms_norm_eps: float = 1e-6
    rope_base: float = 10000.0
    attn_impl: str = "sdpa"

    @property
    def dense_intermediate_size(self) -> int:
        # Dense layer has equivalent active parameters: (top_k + n_shared) * moe_intermediate_size
        return (self.num_experts_per_tok + self.n_shared_experts) * self.moe_intermediate_size


PRESETS = {
    "tiny": dict(
        n_layer=4, d_model=256, n_head=4, n_kv_head=2, block_size=256,
        moe_intermediate_size=512, n_routed_experts=4, num_experts_per_tok=2,
        n_shared_experts=1, n_dense_first=1
    ),
    "small": dict(
        n_layer=12, d_model=512, n_head=8, n_kv_head=2, block_size=1024,
        moe_intermediate_size=1024, n_routed_experts=8, num_experts_per_tok=2,
        n_shared_experts=1, n_dense_first=1
    ),
}


# ---------------------------------------------------------------- Building blocks
class DeepSeekRMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * self.weight).type_as(x)


def rope_cache(max_seq_len: int, head_dim: int, base: float = 10000.0):
    """Precomputes rotary frequency table."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Applies rotary position embeddings."""
    T = x.shape[2]
    cos, sin = cos[None, None, :T], sin[None, None, :T]
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).type_as(x)


def sdpa_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Scaled Dot-Product Attention with PyTorch native kernel."""
    return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)


ATTN_IMPLS = {"sdpa": sdpa_attn}


class DeepSeekAttention(nn.Module):
    """Grouped-Query Attention."""

    def __init__(self, c: DeepSeekConfig):
        super().__init__()
        self.n_head = c.n_head
        self.n_kv_head = c.n_kv_head
        self.head_dim = c.d_model // c.n_head
        self.q_proj = nn.Linear(c.d_model, self.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(c.d_model, self.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(c.d_model, self.n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_head * self.head_dim, c.d_model, bias=False)
        self.impl = c.attn_impl

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        y = ATTN_IMPLS[self.impl](q, k, v)
        return self.o_proj(y.transpose(1, 2).reshape(B, T, -1))


class DeepSeekMLP(nn.Module):
    """SwiGLU Feed-Forward Network."""

    def __init__(self, d_model: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, intermediate_size, bias=False)
        self.up_proj = nn.Linear(d_model, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------- MoE Module
class DeepSeekMoE(nn.Module):
    """DeepSeek-style Mixture-of-Experts with Shared Experts + Top-K Routed Experts."""

    def __init__(self, c: DeepSeekConfig):
        super().__init__()
        self.E = c.n_routed_experts
        self.top_k = c.num_experts_per_tok
        self.router = nn.Linear(c.d_model, self.E, bias=False)
        self.experts = nn.ModuleList(
            DeepSeekMLP(c.d_model, c.moe_intermediate_size) for _ in range(self.E)
        )
        self.shared_experts = nn.ModuleList(
            DeepSeekMLP(c.d_model, c.moe_intermediate_size) for _ in range(c.n_shared_experts)
        )

    def forward(self, x: torch.Tensor):
        B, T, D = x.shape
        xf = x.reshape(-1, D)
        N, E, k = xf.shape[0], self.E, self.top_k

        logits = self.router(xf).float()                    # (N, E)
        probs = logits.softmax(dim=-1)
        topv, topi = probs.topk(k, dim=-1)                  # (N, k)
        topv = topv / topv.sum(dim=-1, keepdim=True)        # Renormalize

        # --- Auxiliary Load-Balancing & Router z-loss
        counts = torch.bincount(topi.reshape(-1), minlength=E)       # (E,)
        f = counts.float() / (N * k)                                 # Expert token fraction (no grad)
        P = probs.mean(0)                                            # Mean router probability
        aux = E * (f * P).sum()                                      # ~1.0 when perfectly balanced
        z = torch.logsumexp(logits, dim=-1).pow(2).mean()            # Router z-loss for logits stability

        # --- Dispatch tokens to routed experts
        flat_e = topi.reshape(-1)
        flat_w = topv.reshape(-1)
        tok = torch.arange(N, device=x.device).repeat_interleave(k)
        order = flat_e.argsort()
        tok, flat_w = tok[order], flat_w[order]

        out = torch.zeros_like(xf)
        start = 0
        for e, cnt in enumerate(counts.tolist()):
            if cnt == 0:
                continue
            idx = tok[start:start + cnt]
            w = flat_w[start:start + cnt, None]
            y = self.experts[e](xf[idx])
            out.index_add_(0, idx, (y * w.to(y.dtype)).to(out.dtype))
            start += cnt

        # --- Shared expert forward pass
        for s in self.shared_experts:
            out = out + s(xf).to(out.dtype)

        return out.view(B, T, D), aux, z, counts


class DeepSeekDecoderLayer(nn.Module):
    """Transformer Block with Pre-RMSNorm, GQA, and either Dense or MoE FFN."""

    def __init__(self, c: DeepSeekConfig, use_moe: bool):
        super().__init__()
        self.input_layernorm = DeepSeekRMSNorm(c.d_model, c.rms_norm_eps)
        self.self_attn = DeepSeekAttention(c)
        self.post_attention_layernorm = DeepSeekRMSNorm(c.d_model, c.rms_norm_eps)
        self.use_moe = use_moe
        self.mlp = DeepSeekMoE(c) if use_moe else DeepSeekMLP(c.d_model, c.dense_intermediate_size)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        if self.use_moe:
            y, aux, z, cnt = self.mlp(self.post_attention_layernorm(x))
        else:
            y = self.mlp(self.post_attention_layernorm(x))
            aux, z, cnt = x.new_zeros(()), x.new_zeros(()), None
        return x + y, aux, z, cnt


# ---------------------------------------------------------------- Full Model
class DeepSeekForCausalLM(nn.Module):
    """DeepSeek-MoE Language Model."""

    def __init__(self, c: DeepSeekConfig):
        super().__init__()
        self.c = c
        self.embed_tokens = nn.Embedding(c.vocab_size, c.d_model)
        self.layers = nn.ModuleList([
            DeepSeekDecoderLayer(c, use_moe=(i >= c.n_dense_first))
            for i in range(c.n_layer)
        ])
        self.norm = DeepSeekRMSNorm(c.d_model, c.rms_norm_eps)
        self.lm_head = nn.Linear(c.d_model, c.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight  # Weight tying

        cos, sin = rope_cache(c.block_size, c.d_model // c.n_head, base=c.rope_base)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        self.apply(self._init_weights)
        for name, param in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * c.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self):
        total = sum(p.numel() for p in self.parameters())
        active = total
        for layer in self.layers:
            if layer.use_moe:
                expert_params = sum(p.numel() for p in layer.mlp.experts[0].parameters())
                inactive_routed = layer.mlp.E - layer.mlp.top_k
                active -= inactive_routed * expert_params
        return total, active

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        x = self.embed_tokens(idx)
        aux = z = x.new_zeros(())
        counts = []
        for layer in self.layers:
            x, a, zz, cnt = layer(x, self.cos, self.sin)
            if cnt is not None:
                aux, z = aux + a, z + zz
                counts.append(cnt)

        logits = self.lm_head(self.norm(x))
        out = {"logits": logits, "counts": torch.stack(counts) if counts else None}

        if targets is not None:
            ce = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.view(-1))
            n_moe = max(len(counts), 1)
            mean_aux = aux / n_moe
            mean_z = z / n_moe
            loss = ce + self.c.aux_loss_coef * mean_aux + self.c.router_z_loss_coef * mean_z
            out["loss"] = loss
            out["ce"] = ce
            out["aux"] = mean_aux
            out["z"] = mean_z
        return out
