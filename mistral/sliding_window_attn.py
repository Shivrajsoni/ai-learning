# Implement Self Attn with Sliding Window Protocol & Grouped Query Attention (GQA) for Mistral

import math
import torch
import torch.nn as nn


def precompute_rope(head_dim: int, max_seq_len: int, theta: float = 10000.0, device=None):
    i = torch.arange(0, head_dim, 2, device=device).float()
    inv_freq = 1.0 / (theta ** (i / head_dim))
    pos = torch.arange(max_seq_len, device=device).float()
    angles = torch.outer(pos, inv_freq)
    return torch.cos(angles), torch.sin(angles)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
    # x: (B, num_heads, T, head_dim)
    T = x.size(2)
    cos_slice = cos[start_pos : start_pos + T, :][None, None, :, :].to(device=x.device, dtype=x.dtype)
    sin_slice = sin[start_pos : start_pos + T, :][None, None, :, :].to(device=x.device, dtype=x.dtype)
    cos_full = torch.cat([cos_slice, cos_slice], dim=-1)
    sin_full = torch.cat([sin_slice, sin_slice], dim=-1)
    return (x * cos_full) + (rotate_half(x) * sin_full)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeats key/value heads for Grouped Query Attention (GQA)."""
    if n_rep == 1:
        return x
    B, n_kv_heads, T, head_dim = x.shape
    return (
        x[:, :, None, :, :]
        .expand(B, n_kv_heads, n_rep, T, head_dim)
        .reshape(B, n_kv_heads * n_rep, T, head_dim)
    )


class SlidingWindowattn(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int = None,
        window_size: int = 4096,
        max_seq_len: int = 32768,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.num_rep = self.num_heads // self.num_kv_heads
        self.head_dim = d_model // num_heads
        self.window_size = window_size
        self.max_seq_len = max_seq_len

        # Q projection: num_heads * head_dim = d_model
        self.w_q = nn.Linear(self.d_model, self.num_heads * self.head_dim, bias=False)
        # K, V projections for GQA: num_kv_heads * head_dim
        self.w_k = nn.Linear(self.d_model, self.num_kv_heads * self.head_dim, bias=False)
        self.w_v = nn.Linear(self.d_model, self.num_kv_heads * self.head_dim, bias=False)
        self.w_o = nn.Linear(self.d_model, self.num_heads * self.head_dim, bias=False)

        # Precompute RoPE buffers
        cos, sin = precompute_rope(self.head_dim, max_seq_len, theta=rope_theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        # Dynamic KV Cache for inference
        self.cache_k = None
        self.cache_v = None

    def reset_cache(self):
        self.cache_k = None
        self.cache_v = None

    def forward(self, x: torch.Tensor, start_pos: int = 0, use_cache: bool = False) -> torch.Tensor:
        B, T, _ = x.shape

        # Projections
        Q = self.w_q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.w_k(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        V = self.w_v(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE using absolute sequence positions
        Q = apply_rope(Q, self.cos, self.sin, start_pos=start_pos)
        K = apply_rope(K, self.cos, self.sin, start_pos=start_pos)

        # Manage Sliding Window KV Cache
        if use_cache:
            if self.cache_k is None or start_pos == 0:
                self.cache_k = K
                self.cache_v = V
            else:
                self.cache_k = torch.cat([self.cache_k, K], dim=2)
                self.cache_v = torch.cat([self.cache_v, V], dim=2)

            # Keep only the last window_size tokens in cache
            if self.cache_k.size(2) > self.window_size:
                self.cache_k = self.cache_k[:, :, -self.window_size :, :]
                self.cache_v = self.cache_v[:, :, -self.window_size :, :]

            if T == 1:
                K = self.cache_k
                V = self.cache_v

        # Expand Key and Value heads for GQA to match Q heads
        K = repeat_kv(K, self.num_rep)
        V = repeat_kv(V, self.num_rep)

        total_k_len = K.size(2)
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if T > 1:
            # Sliding window causal mask for prefill
            i = torch.arange(T, device=x.device).unsqueeze(1)
            j = torch.arange(total_k_len, device=x.device).unsqueeze(0)
            mask = (j > i) | (i - j >= self.window_size)
            scores = scores.masked_fill(mask, float("-inf"))

        attn = torch.softmax(scores, dim=-1, dtype=torch.float32).to(x.dtype)
        output = attn @ V

        output = output.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.w_o(output)
