import torch
import torch.nn as nn
import torch.nn.functional as F
from mistral.sliding_window_attn import SlidingWindowattn
from basics.norm import CustomRMSNorm


class SwigluMLP(nn.Module):
    """
    SwiGLU Feed-Forward Network:
        output = (SiLU(x @ W_gate) * (x @ W_up)) @ W_down
    """
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        # gate_proj (w1)
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        # up_proj (w3)
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        # down_proj (w2)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class MistralTransformerBlock(nn.Module):
    """
    Mistral Transformer Decoder Layer with Pre-RMSNorm:
        1. h = x + SelfAttn(RMSNorm(x))
        2. out = h + MLP(RMSNorm(h))
    """
    def __init__(self, config: dict):
        super().__init__()
        self.dim = config.get("hidden_size", config.get("dim", config.get("d_model", 4096)))
        self.eps = config.get("rms_norm_eps", config.get("eps", 1e-5))
        self.window_size = config.get("sliding_window", config.get("window_size", 4096))
        self.num_heads = config.get("num_attention_heads", config.get("num_heads", 32))
        self.num_kv_heads = config.get("num_key_value_heads", self.num_heads)
        self.max_seq_len = config.get("max_position_embeddings", config.get("max_seq_len", 32768))
        self.rope_theta = config.get("rope_theta", 10000.0)

        # Mistral intermediate_size (defaults to 14336 for 7B)
        hidden_dim = config.get("intermediate_size", config.get("hidden_dim", 14336))

        # Normalization layers (Pre-LN architecture)
        self.attention_norm = CustomRMSNorm(self.dim, eps=self.eps)
        self.ffn_norm = CustomRMSNorm(self.dim, eps=self.eps)

        # Attention with SWA, GQA, and RoPE
        self.self_attn = SlidingWindowattn(
            d_model=self.dim,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            window_size=self.window_size,
            max_seq_len=self.max_seq_len,
            rope_theta=self.rope_theta,
        )
        self.mlp = SwigluMLP(dim=self.dim, hidden_dim=hidden_dim)

    def forward(self, x: torch.Tensor, start_pos: int = 0, use_cache: bool = False) -> torch.Tensor:
        # 1. Attention with Pre-RMSNorm & Residual Connection
        h = x + self.self_attn(self.attention_norm(x), start_pos=start_pos, use_cache=use_cache)
        # 2. MLP with Pre-RMSNorm & Residual Connection
        out = h + self.mlp(self.ffn_norm(h))
        return out


class MistralModel(nn.Module):
    """
    Full Mistral Model for Causal Language Modeling:
        input_ids -> Token Embeddings -> N x MistralTransformerBlock -> RMSNorm -> LM Head -> Logits
    """
    def __init__(self, config: dict):
        super().__init__()
        self.hidden_size = config.get("hidden_size", config.get("dim", config.get("d_model", 4096)))
        self.vocab_size = config.get("vocab_size", 32000)
        self.num_layers = config.get("num_hidden_layers", config.get("num_layers", 32))
        self.eps = config.get("rms_norm_eps", config.get("eps", 1e-5))

        # 1. Token Embeddings
        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size)

        # 2. Stack of Transformer Layers
        self.layers = nn.ModuleList([
            MistralTransformerBlock(config) for _ in range(self.num_layers)
        ])

        # 3. Final RMSNorm
        self.norm = CustomRMSNorm(self.hidden_size, eps=self.eps)

        # 4. Language Model Head (Projects hidden state to vocabulary logits)
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)

    def reset_cache(self):
        """Resets the KV-cache across all layers before running a new prompt."""
        for layer in self.layers:
            layer.self_attn.reset_cache()

    def forward(self, input_ids: torch.Tensor, start_pos: int = 0, use_cache: bool = False) -> torch.Tensor:
        # (B, T) -> (B, T, hidden_size)
        x = self.embed_tokens(input_ids)

        for layer in self.layers:
            x = layer(x, start_pos=start_pos, use_cache=use_cache)

        x = self.norm(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        return logits
