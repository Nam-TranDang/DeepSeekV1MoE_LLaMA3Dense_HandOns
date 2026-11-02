"""LLaMA-3 Dense Architecture (100% self-contained).

Components:
- RMSNorm
- RoPE (Rotary Position Embeddings)
- Grouped-Query Attention (GQA) with PyTorch SDPA (FlashAttention baseline)
- SwiGLU Feed-Forward Network
- Weight-tied Embedding and LM Head

 #### A. model_llama.py (100% Standalone LLaMA-3 Dense)

  • Zero MoE clutter: No router code, no auxiliary loss, no expert loops.
  • Classes: LlamaConfig, LlamaRMSNorm, LlamaAttention (GQA 4:1), LlamaMLP (SwiGLU),
  LlamaDecoderLayer, and LlamaForCausalLM.
  • Presets:
      • "tiny": 4 layers, 256 dim (~10M params).
      • "small": 12 layers, 512 dim, intermediate 3072 (~75M params).

"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LlamaConfig:
    vocab_size: int = 16384
    n_layer: int = 12
    d_model: int = 512
    n_head: int = 8
    n_kv_head: int = 2              # 4:1 Query:KV ratio (GQA)
    block_size: int = 1024
    intermediate_size: int = 3072   # SwiGLU hidden dim (~6 * d_model / 2)
    rms_norm_eps: float = 1e-6
    rope_base: float = 10000.0
    attn_impl: str = "sdpa"         # extensible to "triton"


PRESETS = {
    "tiny": dict(
        n_layer=4, d_model=256, n_head=4, n_kv_head=2,
        block_size=256, intermediate_size=1536
    ),
    "small": dict(
        n_layer=12, d_model=512, n_head=8, n_kv_head=2,
        block_size=1024, intermediate_size=3072
    ),
}


# ---------------------------------------------------------------- Building blocks
class LlamaRMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)."""

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
    """Applies rotary position embeddings (Su et al., 2021)."""
    # x: (B, H, T, head_dim)
    T = x.shape[2]
    cos, sin = cos[None, None, :T], sin[None, None, :T]
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).type_as(x)


def sdpa_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Scaled Dot-Product Attention with PyTorch native kernel (SDPA/FlashAttention)."""
    return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)


ATTN_IMPLS = {"sdpa": sdpa_attn}


class LlamaAttention(nn.Module):
    """Grouped-Query Attention (Ainslie et al., 2023)."""

    def __init__(self, c: LlamaConfig):
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


class LlamaMLP(nn.Module):
    """SwiGLU Feed-Forward Network (Shazeer, 2020)."""

    def __init__(self, d_model: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, intermediate_size, bias=False)  # w1
        self.up_proj = nn.Linear(d_model, intermediate_size, bias=False)    # w3
        self.down_proj = nn.Linear(intermediate_size, d_model, bias=False)  # w2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LlamaDecoderLayer(nn.Module):
    """Single LLaMA-3 Transformer Block with Pre-RMSNorm."""

    def __init__(self, c: LlamaConfig):
        super().__init__()
        self.input_layernorm = LlamaRMSNorm(c.d_model, c.rms_norm_eps)
        self.self_attn = LlamaAttention(c)
        self.post_attention_layernorm = LlamaRMSNorm(c.d_model, c.rms_norm_eps)
        self.mlp = LlamaMLP(c.d_model, c.intermediate_size)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


# ---------------------------------------------------------------- Full Model
class LlamaForCausalLM(nn.Module):
    """LLaMA-3 Autoregressive Language Model."""

    def __init__(self, c: LlamaConfig):
        super().__init__()
        self.c = c
        self.embed_tokens = nn.Embedding(c.vocab_size, c.d_model)
        self.layers = nn.ModuleList([LlamaDecoderLayer(c) for _ in range(c.n_layer)])
        self.norm = LlamaRMSNorm(c.d_model, c.rms_norm_eps)
        self.lm_head = nn.Linear(c.d_model, c.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight  # Weight tying

        cos, sin = rope_cache(c.block_size, c.d_model // c.n_head, base=c.rope_base)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        self.apply(self._init_weights)
        # Scaled residual projection initialization (GPT-2/LLaMA style)
        for name, param in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * c.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self):
        total = sum(p.numel() for p in self.parameters())
        return total, total  # For dense model, active == total

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        x = self.embed_tokens(idx)
        for layer in self.layers:
            x = layer(x, self.cos, self.sin)
        logits = self.lm_head(self.norm(x))

        out = {"logits": logits}
        if targets is not None:
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.view(-1))
            out["loss"] = loss
            out["ce"] = loss
        return out
