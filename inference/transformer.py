from inference.group_query_attn import GroupQueryAttention
from basics.norm import CustomRMSNorm
import torch
import torch.nn as nn 
import torch.nn.functional as F 


class SwiGLUMLP(nn.Module):
    def __init__(self , hidden_size: int , intermediate_size:int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size , intermediate_size , bias= False)
        self.up_proj = nn.Linear(hidden_size ,intermediate_size , bias=False)
        self.down_proj = nn.Linear(intermediate_size , hidden_size , bias=False)
    
    def forward(self , x: torch.Tensor)-> torch.Tensor:
        # F.silu(gate) * up 
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


# Complete Individual Block 
    # In["Input Tensor x: [B, T, 2048]"] --> Norm1["input_layernorm (RMSNorm)"]
    # Norm1 --> Attn["Attention (Q, K, V, RoPE, O)"]
    # Attn --> Add1["Residual Add: x + Attn(Norm1(x))"]
    # In --> Add1
    # Add1 --> Norm2["post_attention_layernorm (RMSNorm)"]
    # Norm2 --> MLP["SwiGLU MLP (gate, up, down)"]
    # MLP --> Add2["Residual Add: x + MLP(Norm2(x))"]
    # Add1 --> Add2
    # Add2 --> Out["Output to next layer: [B, T, 2048]"]

class TransformerBlock(nn.Module):
    def __init__(self,config):
        super().__init__() 
        self.input_layernorm = CustomRMSNorm(
            config["hidden_size"],
            eps=config["rms_norm_eps"]
        )
        self.self_attn = GroupQueryAttention(
            d_model=config["hidden_size"],
            num_heads=config["num_attention_heads"],
            num_kv_heads=config["num_key_value_heads"],
            max_seq_len=config["max_position_embeddings"]
        )
        self.post_attention_layernorm = CustomRMSNorm(
            config["hidden_size"],
            eps=config["rms_norm_eps"]
        )

        self.mlp = SwiGLUMLP(config["hidden_size"] , config["intermediate_size"])

    def forward(self, x):
        # 1 : Self Attention with Pre-Norm and Residual 
        x = x + self.self_attn(self.input_layernorm(x))
        # 2 MLP with Pre-Norm and Residual 
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x 



# complete 22 Blocks part in 1 Class 
class TinyLlamaModel(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config["num_hidden_layers"])
        ])
        
        self.norm  = CustomRMSNorm(
            config["hidden_size"],
            eps=config["rms_norm_eps"]
        )
        
        self.lm_head = nn.Linear(
            config["hidden_size"],
            config["vocab_size"],
            bias = False 
        )
        
        self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"]) 

    def forward(self,input_ids):
        # Convert token IDS --> Vector [B , T , 2048 ]
        x = self.embed_tokens(input_ids)

        # Pass through 22 Transformer Blocks 
        for layer in self.layers:
            x = layer(x)
        
        # Final RMSNORM + Projection to vocab logits [B, T, 32000]
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits


    def load_weights(self , weights):
     # 1 Embeddings and LM Head 
       self.embed_tokens.weight.data.copy_(weights["model.embed_tokens.weight"])
       self.lm_head.weight.data.copy_(weights["lm_head.weight"])
       self.norm.gamma.data.copy_(weights["model.norm.weight"])
    

    # 2 Iterate Through 22 Layers 
       for i , layer in enumerate(self.layers):
        prefix = f"model.layers.{i}."

        # Attention 
        layer.input_layernorm.gamma.data.copy_(weights[f"{prefix}input_layernorm.weight"])
        layer.self_attn.w_q.weight.data.copy_(weights[f"{prefix}self_attn.q_proj.weight"])
        layer.self_attn.w_k.weight.data.copy_(weights[f"{prefix}self_attn.k_proj.weight"])
        layer.self_attn.w_v.weight.data.copy_(weights[f"{prefix}self_attn.v_proj.weight"])
        layer.self_attn.w_o.weight.data.copy_(weights[f"{prefix}self_attn.o_proj.weight"])

        # MLP 
        layer.post_attention_layernorm.gamma.data.copy_(weights[f"{prefix}post_attention_layernorm.weight"])
        layer.mlp.gate_proj.weight.data.copy_(weights[f"{prefix}mlp.gate_proj.weight"])
        layer.mlp.up_proj.weight.data.copy_(weights[f"{prefix}mlp.up_proj.weight"])
        layer.mlp.down_proj.weight.data.copy_(weights[f"{prefix}mlp.down_proj.weight"])

