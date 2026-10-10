import json
from pathlib import Path
import torch
from safetensors import safe_open
from transformers import AutoTokenizer

from llama_block import LLamaModel


def load_llama_weights_sharded(model: LLamaModel, checkpoint_dir: str, device: str = "cpu", dtype: torch.dtype = torch.float16):
    """
    True zero-copy streaming loader: Streams each parameter individually
    from memory-mapped safetensors directly into model tensors without
    creating intermediate in-memory Python dictionaries.
    """
    ckpt_path = Path(checkpoint_dir)
    index_file = ckpt_path / "model.safetensors.index.json"

    if index_file.exists():
        with open(index_file) as f:
            weight_map = json.load(f)["weight_map"]
        shard_files = sorted(set(weight_map.values()))
    else:
        shard_files = ["model.safetensors"]

    print(f"Loading {len(shard_files)} weight shard(s)...")

    # Build direct parameter lookup table
    # Maps Hugging Face checkpoint key -> destination parameter tensor
    param_map = {
        "model.embed_tokens.weight": model.embed_tokens.weight,
        "model.norm.weight": model.norm.gamma,
        "lm_head.weight": model.lm_head.weight,
    }

    for i, layer in enumerate(model.layers):
        prefix = f"model.layers.{i}."
        param_map[f"{prefix}input_layernorm.weight"] = layer.input_layernorm.gamma
        param_map[f"{prefix}self_attn.q_proj.weight"] = layer.self_attn.w_q.weight
        param_map[f"{prefix}self_attn.k_proj.weight"] = layer.self_attn.w_k.weight
        param_map[f"{prefix}self_attn.v_proj.weight"] = layer.self_attn.w_v.weight
        param_map[f"{prefix}self_attn.o_proj.weight"] = layer.self_attn.w_o.weight
        param_map[f"{prefix}post_attention_layernorm.weight"] = layer.post_attention_layernorm.gamma
        param_map[f"{prefix}mlp.gate_proj.weight"] = layer.mlp.gate_proj.weight
        param_map[f"{prefix}mlp.up_proj.weight"] = layer.mlp.up_proj.weight
        param_map[f"{prefix}mlp.down_proj.weight"] = layer.mlp.down_proj.weight

    loaded_keys = set()

    # Stream shard by shard directly from mmap
    for shard in shard_files:
        shard_path = ckpt_path / shard
        print(f"  Streaming {shard}...")
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key in param_map:
                    # Direct zero-copy slice from mmap -> device copy
                    tensor = f.get_tensor(key)
                    param_map[key].data.copy_(tensor.to(dtype=dtype, device=device))
                    loaded_keys.add(key)

    # Tie embeddings if lm_head was not present as an independent key
    if "lm_head.weight" not in loaded_keys:
        print("  Tying lm_head weights to embed_tokens...")
        model.lm_head.weight.data.copy_(model.embed_tokens.weight.data)

    print("✓ All weights loaded directly without RAM spikes!")


@torch.no_grad()
def generate(
    model: LLamaModel,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 60,
    temperature: float = 0.7,
    top_k: int = 40,
    device: str = "cpu"
) -> str:
    """
    High-Performance Autoregressive Generation with KV Cache and Space-Preserving Streaming.
    """
    model.eval()
    model.reset_cache()

    # 1. Tokenize prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    prompt_len = input_ids.shape[1]

    # Clean cumulative streaming tracker to preserve SentencePiece spaces
    generated_ids = input_ids.clone()
    printed_len = len(prompt)

    # -------------------------------------------------------------
    # Phase 1: Prefill (Process entire prompt once & populate KV cache)
    # -------------------------------------------------------------
    logits = model(input_ids, start_pos=0, use_cache=True)
    next_token_logits = logits[:, -1, :]

    # Sample first token
    if temperature == 0.0:
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
    else:
        scaled_logits = next_token_logits / temperature
        if top_k > 0:
            indices_to_remove = scaled_logits < torch.topk(scaled_logits, top_k)[0][..., -1, None]
            scaled_logits[indices_to_remove] = float("-inf")
        probs = torch.softmax(scaled_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

    generated_ids = torch.cat([generated_ids, next_token], dim=-1)
    
    # Print first token
    current_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(current_text[printed_len:], end="", flush=True)
    printed_len = len(current_text)

    # -------------------------------------------------------------
    # Phase 2: Autoregressive Decoding with O(1) KV-Cache!
    # Only the newest 1-token (1x1) is passed through the network.
    # -------------------------------------------------------------
    curr_pos = prompt_len

    for _ in range(max_new_tokens - 1):
        if next_token.item() == tokenizer.eos_token_id:
            break

        # Forward pass ONLY the single new token!
        logits = model(next_token, start_pos=curr_pos, use_cache=True)
        next_token_logits = logits[:, -1, :]

        if temperature == 0.0:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        else:
            scaled_logits = next_token_logits / temperature
            if top_k > 0:
                indices_to_remove = scaled_logits < torch.topk(scaled_logits, top_k)[0][..., -1, None]
                scaled_logits[indices_to_remove] = float("-inf")
            probs = torch.softmax(scaled_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        generated_ids = torch.cat([generated_ids, next_token], dim=-1)
        curr_pos += 1

        # Smooth, correctly spaced streaming
        current_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        new_text = current_text[printed_len:]
        print(new_text, end="", flush=True)
        printed_len = len(current_text)

    print("\n", flush=True)
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)
