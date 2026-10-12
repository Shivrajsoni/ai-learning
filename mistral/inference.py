import os
import sys
from pathlib import Path

# Ensure root workspace directory is in python search path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import json
import torch
from safetensors import safe_open
from tokenizers import Tokenizer
from mistral.transformer import MistralModel


class MistralInference:
    """
    Mistral Inference Pipeline:
    - Apple Silicon MPS acceleration with torch.float16
    - Zero-copy memory-mapped streaming weight loader (shard-by-shard into MPS memory)
    - Sliding Window Attention (SWA) + Grouped Query Attention (GQA) + RoPE
    - Autoregressive generation loop
    """
    def __init__(
        self,
        checkpoint_dir: str = "./mistral-7b",
        tokenizer_path: str = None,
        device: str = "mps" if torch.backends.mps.is_available() else "cpu",
        dtype: torch.dtype = torch.float16
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        self.ckpt_path = Path(checkpoint_dir)

        print(f"[Init] Initializing Mistral on device: {self.device} (dtype: {self.dtype})")

        # 1. Load Config
        config_path = self.ckpt_path / "config.json"
        if not config_path.exists():
            print(f"[Warning] {config_path} not found. Using default Mistral-7B architecture config.")
            self.config = {
                "vocab_size": 32000,
                "hidden_size": 4096,
                "intermediate_size": 14336,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "num_hidden_layers": 32,
                "sliding_window": 4096,
                "max_position_embeddings": 32768,
                "rope_theta": 10000.0,
                "rms_norm_eps": 1e-5
            }
        else:
            with open(config_path, "r") as f:
                self.config = json.load(f)

        # 2. Load Tokenizer
        tok_file = tokenizer_path or str(self.ckpt_path / "tokenizer.json")
        if os.path.exists(tok_file):
            print(f"[Init] Loading tokenizer from {tok_file}...")
            self.tokenizer = Tokenizer.from_file(tok_file)
        else:
            print(f"[Info] Tokenizer file {tok_file} not found locally.")
            self.tokenizer = None

        # 3. Instantiate Model directly on target device
        print("[Init] Instantiating MistralModel architecture...")
        self.model = MistralModel(self.config).to(dtype=self.dtype, device=self.device)
        self.model.eval()

        # 4. Zero-Copy Streaming Load from Safetensors
        self.load_weights_zero_copy()

    @torch.no_grad()
    def load_weights_zero_copy(self):
        """
        Loads weights using memory-mapped safetensors (`safe_open`).
        Reads one tensor at a time and transfers directly to target device (MPS/CPU),
        avoiding intermediate duplicate full-model RAM copies.
        """
        if not self.ckpt_path.exists():
            print(f"[Warning] Checkpoint directory {self.ckpt_path} does not exist.")
            return

        index_file = self.ckpt_path / "model.safetensors.index.json"
        if index_file.exists():
            with open(index_file) as f:
                weight_map = json.load(f)["weight_map"]
            shard_files = sorted(set(weight_map.values()))
        else:
            single_shard = self.ckpt_path / "model.safetensors"
            shard_files = ["model.safetensors"] if single_shard.exists() else []

        if not shard_files:
            print(f"[Note] No safetensors files found in {self.ckpt_path}.")
            return

        print(f"[Loading] Streaming weights from {len(shard_files)} safetensors shard(s) into {self.device}...")

        # Map HuggingFace state_dict keys to model layers
        mapped_targets = {
            "model.embed_tokens.weight": self.model.embed_tokens.weight,
            "model.norm.weight": self.model.norm.gamma,
            "lm_head.weight": self.model.lm_head.weight,
        }

        for i, layer in enumerate(self.model.layers):
            prefix = f"model.layers.{i}."
            # Norms
            mapped_targets[f"{prefix}input_layernorm.weight"] = layer.attention_norm.gamma
            mapped_targets[f"{prefix}post_attention_layernorm.weight"] = layer.ffn_norm.gamma
            # Attention Projections (GQA)
            mapped_targets[f"{prefix}self_attn.q_proj.weight"] = layer.self_attn.w_q.weight
            mapped_targets[f"{prefix}self_attn.k_proj.weight"] = layer.self_attn.w_k.weight
            mapped_targets[f"{prefix}self_attn.v_proj.weight"] = layer.self_attn.w_v.weight
            mapped_targets[f"{prefix}self_attn.o_proj.weight"] = layer.self_attn.w_o.weight
            # SwiGLU MLP Projections
            mapped_targets[f"{prefix}mlp.gate_proj.weight"] = layer.mlp.w_gate.weight
            mapped_targets[f"{prefix}mlp.up_proj.weight"] = layer.mlp.w_up.weight
            mapped_targets[f"{prefix}mlp.down_proj.weight"] = layer.mlp.w_down.weight

        loaded_keys = set()
        for shard_name in shard_files:
            shard_path = self.ckpt_path / shard_name
            if not shard_path.exists():
                print(f"  [Skip] Shard not found: {shard_path}")
                continue
            print(f"  Loading shard: {shard_name}...")
            with safe_open(shard_path, framework="pt", device="cpu") as f:
                for k in f.keys():
                    if k in mapped_targets:
                        tensor = f.get_tensor(k)
                        target = mapped_targets[k]
                        target.data.copy_(tensor.to(dtype=self.dtype, device=self.device))
                        loaded_keys.add(k)

        # Tie lm_head if untied in config / absent in state_dict
        if "lm_head.weight" not in loaded_keys:
            print("  [Info] Tying lm_head weights to embed_tokens...")
            self.model.lm_head.weight.data.copy_(self.model.embed_tokens.weight.data)

        print(f"[Done] Successfully loaded {len(loaded_keys)} weight tensors onto {self.device}!")

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.0,
        eos_token_id: int = 2
    ) -> str:
        """
        Autoregressive generation with prefill and token-by-token decoding.
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is not loaded. Cannot encode prompt.")

        # Encode input prompt
        encoded = self.tokenizer.encode(prompt)
        prompt_tokens = encoded.ids
        input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=self.device)

        # Reset KV-cache across all layers before new generation
        self.model.reset_cache()

        generated_tokens = []
        prompt_len = input_ids.shape[1]

        # Phase 1: Prefill prompt tokens
        logits = self.model(input_ids, start_pos=0, use_cache=True)
        next_token_logits = logits[:, -1, :]

        if temperature > 0.0:
            probs = torch.softmax(next_token_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        token_id = next_token.item()
        if token_id == eos_token_id:
            return ""

        generated_tokens.append(token_id)
        current_token = next_token
        start_pos = prompt_len

        # Phase 2: Autoregressive Decode step-by-step
        for _ in range(max_new_tokens - 1):
            logits = self.model(current_token, start_pos=start_pos, use_cache=True)
            next_token_logits = logits[:, -1, :]

            if temperature > 0.0:
                probs = torch.softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            token_id = next_token.item()
            if token_id == eos_token_id:
                break

            generated_tokens.append(token_id)
            current_token = next_token
            start_pos += 1

        return self.tokenizer.decode(generated_tokens)


if __name__ == "__main__":
    print("=== Mistral-7B Inference Engine (Apple MPS) ===")
    ckpt_dir = input("Enter checkpoint directory [default: ./mistral-7b]: ").strip()
    if not ckpt_dir:
        ckpt_dir = "./mistral-7b"

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    infer = MistralInference(checkpoint_dir=ckpt_dir, device=device)

    while True:
        prompt = input("\nEnter prompt (or 'exit' to quit): ").strip()
        if prompt.lower() in ("exit", "quit"):
            break
        if not prompt:
            continue

        print("\nGenerating response...")
        output = infer.generate(prompt=prompt, max_new_tokens=40)
        print(f"Output:\n{output}\n")
