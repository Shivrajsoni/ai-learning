
import math
import torch.nn as nn 
import torch 

def precompute_rope(head_dim ,max_seq_len ,base=10000.0 ,device=None):
    i = torch.arange(0, head_dim , 2 , device=device).float()
    inv_freq = 1.0 / (base ** (i / head_dim))
    pos = torch.arange(max_seq_len ,device=None).float()
    angle = torch.outer( pos , inv_freq)
    return torch.cos(angle),torch.sin(angle)
    

def rotate_half(x):

    x1 = x[..., :x.shape[-1] // 2 ]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2,x1),dim=-1)
    
def apply_rope(x, sin, cos, start_pos: int = 0):
    # x shape: (B, num_heads, T, head_dim)
    T = x.size(2)
    cos_slice = cos[start_pos : start_pos + T, :][None, None, :, :].to(device=x.device, dtype=x.dtype)
    sin_slice = sin[start_pos : start_pos + T, :][None, None, :, :].to(device=x.device, dtype=x.dtype)

    cos = torch.cat([cos_slice, cos_slice], dim=-1)
    sin = torch.cat([sin_slice, sin_slice], dim=-1)

    return (x * cos) + (rotate_half(x) * sin)


class MultiHeadAttn(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len=2048, linear_cls=nn.Linear):
        super().__init__()
        assert d_model % num_heads == 0 

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads 
        self.w_q = linear_cls(self.d_model, self.num_heads * self.head_dim, bias=False)
        self.w_k = linear_cls(self.d_model, self.num_heads * self.head_dim, bias=False)
        self.w_v = linear_cls(self.d_model, self.num_heads * self.head_dim, bias=False)
        self.w_o = linear_cls(self.d_model, self.num_heads * self.head_dim, bias=False)

        cos, sin = precompute_rope(self.head_dim, max_seq_len)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        # Dynamic KV Cache stored during inference
        self.cache_k = None
        self.cache_v = None

    def reset_cache(self):
        self.cache_k = None
        self.cache_v = None

    def forward(self, x, start_pos: int = 0, use_cache: bool = False):
        B, T, C = x.shape  # Batch, Sequence, Dimension

        Q = self.w_q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.w_k(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.w_v(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE using actual token positions
        Q = apply_rope(Q, self.sin, self.cos, start_pos=start_pos)
        K = apply_rope(K, self.sin, self.cos, start_pos=start_pos)

        # KV-Cache management
        if use_cache:
            if self.cache_k is None or start_pos == 0:
                self.cache_k = K
                self.cache_v = V
            else:
                self.cache_k = torch.cat([self.cache_k, K], dim=2)
                self.cache_v = torch.cat([self.cache_v, V], dim=2)

            K = self.cache_k
            V = self.cache_v

        # Total key sequence length (matches cache length during decoding)
        total_k_len = K.size(2)

        scores = Q @ K.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)

        # Apply causal mask only during prefill (when T > 1)
        if T > 1:
            mask = torch.triu(
                torch.ones(T, total_k_len, device=x.device),
                diagonal=start_pos + 1
            ).bool()
            scores = scores.masked_fill(mask, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        output = attn @ V
        output = output.transpose(1, 2).contiguous().view(B, T, self.d_model)

        return self.w_o(output)







