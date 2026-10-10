from multiheadattn import MultiHeadAttn
from basics.norm import CustomRMSNorm
import torch.nn as nn 
import torch 
import torch.nn.functional as F 

#  → embedding lookup                      [seq, 4096]
#  → 32 × transformer block
#  → final RMSNorm
#  → lm_head (linear 4096 → 32000)         logits [seq, 32000]
#  → take last position, sample next token

class SwigluMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, linear_cls=nn.Linear):
        super().__init__()
        self.gate_proj = linear_cls(hidden_size, intermediate_size, bias=False)
        self.up_proj = linear_cls(hidden_size, intermediate_size, bias=False) 
        self.down_proj = linear_cls(intermediate_size, hidden_size, bias=False) 
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)
    


class TransformerBLock(nn.Module):
    def __init__(self, config, linear_cls=nn.Linear):
        super().__init__()

        self.input_layernorm = CustomRMSNorm(
            config["hidden_size"],
            eps=config["rms_norm_eps"]
        )

        self.self_attn = MultiHeadAttn(
            d_model=config["hidden_size"],
            num_heads=config["num_attention_heads"],
            max_seq_len=config["max_position_embeddings"],
            linear_cls=linear_cls
        )

        self.post_attention_layernorm = CustomRMSNorm(
            config["hidden_size"],
            eps=config["rms_norm_eps"]
        )

        self.mlp = SwigluMLP(config["hidden_size"], config["intermediate_size"], linear_cls=linear_cls)


    def forward(self, x, start_pos: int = 0, use_cache: bool = False):
        normed_x = self.input_layernorm(x)
        attn_out  = self.self_attn(normed_x, start_pos=start_pos, use_cache=use_cache)
        x = x + attn_out

        # 2 MLP 
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x 

    


class LLamaModel(nn.Module):
    def __init__(self, config, linear_cls=nn.Linear):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBLock(config, linear_cls=linear_cls) for _ in range(config["num_hidden_layers"])
        ])

        self.norm = CustomRMSNorm(
            config["hidden_size"],
            eps=config["rms_norm_eps"]
        )

        self.lm_head = nn.Linear(
            config["hidden_size"],
            config["vocab_size"],
            bias=False
        )

        self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"])

    def reset_cache(self):
        """Resets KV cache across all layers before a new prompt."""
        for layer in self.layers:
            layer.self_attn.reset_cache()

    def forward(self, input_ids, start_pos: int = 0, use_cache: bool = False):
        x = self.embed_tokens(input_ids)

        for layer in self.layers:
            x = layer(x, start_pos=start_pos, use_cache=use_cache)

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits 



    @torch.no_grad()
    def load_weights(self, checkpoint_dir: str, device: str = "cpu", dtype: torch.dtype = torch.float16):
        """
        Loads weights directly from sharded safetensors into model layers,
        supporting both standard float weights and QuantizedLinear modules.
        """
        from quantize import load_and_quantize_llama
        load_and_quantize_llama(self, checkpoint_dir, device=device, dtype=dtype)


        
