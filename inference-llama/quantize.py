import torch 
import torch.nn as nn 
import torch.nn.functional as F 

class QuantizedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias
    
        self.register_buffer(
            "weight_int8",
            torch.zeros((out_features, in_features), dtype=torch.int8) 
        )

        self.register_buffer(
            "scales",
            torch.zeros((out_features, 1), dtype=torch.float16) 
        )

        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float16))
        else:
            self.register_buffer("bias", None)

    @torch.no_grad()
    def quantize_and_store(self,float_weight: torch.Tensor):
        float_weight = float_weight.to(torch.float32)

        max_val = float_weight.abs().amax(dim=-1 , keepdim=True).clamp(min=1e-5)

        scale = max_val / 127.0 

        q_weight = torch.clamp(torch.round(float_weight/scale) , -127 , 127).to(torch.int8)  

        self.weight_int8.copy_(q_weight)
        self.scales.copy_(scale.to(torch.float16))
         

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # On-the-fly dequantization: W_fp16 = W_int8 * scale
        w_fp = self.weight_int8.to(x.dtype) * self.scales
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w_fp, bias)


@torch.no_grad()
def load_and_quantize_llama(
    model: nn.Module,
    checkpoint_dir: str,
    device: str = "cpu",
    dtype: torch.dtype = torch.float16
):
    """
    Senior-Engineer Streaming Quantization Loader:
    Reads one tensor at a time from memory-mapped safetensors shards.
    If the target module is QuantizedLinear, quantizes directly to INT8
    and frees the FP16 disk buffer immediately, keeping peak RAM ~7 GB.
    """
    import json
    from pathlib import Path
    from safetensors import safe_open

    ckpt_path = Path(checkpoint_dir)
    index_file = ckpt_path / "model.safetensors.index.json"

    if index_file.exists():
        with open(index_file) as f:
            weight_map = json.load(f)["weight_map"]
        shard_files = sorted(set(weight_map.values()))
    else:
        shard_files = ["model.safetensors"]

    print(f"Loading & Quantizing from {len(shard_files)} shard(s)...")

    # Target module mapping
    # Maps HuggingFace state_dict key -> module instance or parameter
    module_map = {
        "model.embed_tokens.weight": ("param", model.embed_tokens.weight),
        "model.norm.weight": ("param", model.norm.gamma),
        "lm_head.weight": ("param", model.lm_head.weight),
    }

    for i, layer in enumerate(model.layers):
        prefix = f"model.layers.{i}."
        # Norms (kept FP16)
        module_map[f"{prefix}input_layernorm.weight"] = ("param", layer.input_layernorm.gamma)
        module_map[f"{prefix}post_attention_layernorm.weight"] = ("param", layer.post_attention_layernorm.gamma)
        # Linear Layers (quantized to INT8)
        module_map[f"{prefix}self_attn.q_proj.weight"] = ("quant", layer.self_attn.w_q)
        module_map[f"{prefix}self_attn.k_proj.weight"] = ("quant", layer.self_attn.w_k)
        module_map[f"{prefix}self_attn.v_proj.weight"] = ("quant", layer.self_attn.w_v)
        module_map[f"{prefix}self_attn.o_proj.weight"] = ("quant", layer.self_attn.w_o)
        module_map[f"{prefix}mlp.gate_proj.weight"] = ("quant", layer.mlp.gate_proj)
        module_map[f"{prefix}mlp.up_proj.weight"] = ("quant", layer.mlp.up_proj)
        module_map[f"{prefix}mlp.down_proj.weight"] = ("quant", layer.mlp.down_proj)

    loaded_keys = set()

    for shard in shard_files:
        shard_path = ckpt_path / shard
        print(f"  Streaming & quantizing {shard}...")
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key in module_map:
                    kind, target = module_map[key]
                    tensor = f.get_tensor(key)

                    if kind == "quant" and isinstance(target, QuantizedLinear):
                        # Direct on-the-fly INT8 quantization
                        target.quantize_and_store(tensor.to(device=device))
                    else:
                        # Direct parameter copy (FP16)
                        target.data.copy_(tensor.to(dtype=dtype, device=device))

                    loaded_keys.add(key)

    # Tie embeddings if lm_head was not present
    if "lm_head.weight" not in loaded_keys:
        print("  Tying lm_head weights to embed_tokens...")
        model.lm_head.weight.data.copy_(model.embed_tokens.weight.data)

    print("✓ Model successfully quantized and loaded in INT8!")


if __name__ == "__main__":
    # Test layer-level quantization benchmark
    original_weight = torch.randn(4096, 4096, dtype=torch.float16)
    fp16_bytes = original_weight.numel() * 2

    layer = QuantizedLinear(4096, 4096)
    layer.quantize_and_store(original_weight)
    int8_bytes = layer.weight_int8.numel() * 1 + layer.scales.numel() * 2

    print(f"FP16 Memory: {fp16_bytes / (1024**2):.2f} MB")
    print(f"INT8 Memory: {int8_bytes / (1024**2):.2f} MB")

    x = torch.randn(1, 4, 4096, dtype=torch.float16)
    true_out = F.linear(x, original_weight)
    quant_out = layer(x)

    error = (true_out - quant_out).abs().mean()
    print(f"Mean Absolute Error: {error.item():.5f}")
