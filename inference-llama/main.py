import sys
from pathlib import Path
import torch

# Setup repository and module paths
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import json
from transformers import AutoTokenizer

from llama_block import LLamaModel
from quantize import QuantizedLinear
from inference import generate

def run():
    checkpoint_dir = Path(__file__).parent / "llama-2-7b-chat-hf"

    with open(checkpoint_dir / "config.json") as f:
        config = json.load(f)

    # Use Apple Silicon GPU (mps) if available, otherwise CPU
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Running inference on device: {device}")

    # 1. Instantiate model with INT8 Linear layers (allocates ~7 GB instead of 14–27 GB)
    model = LLamaModel(config, linear_cls=QuantizedLinear)

    # 2. Stream weights directly from safetensors shards into INT8 buffers on device
    model.load_weights(str(checkpoint_dir), device=device)
    model.to(device)

    # 3. Load tokenizer and run interactive loop
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir))

    while True:
        prompt = input("\nAsk Anything: ") 
        if prompt.strip().lower() in ("quit", "exit"):
            print("Goodbye!") 
            break 
        if not prompt.strip():
            continue

        # Format using official LLaMA-2 Chat Template
        # Without [INST] ... [/INST], chat models continue raw text like trivia lists
        messages = [
            {"role": "system", "content": "You are a helpful and concise assistant."},
            {"role": "user", "content": prompt.strip()}
        ]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        generate(model, tokenizer, formatted_prompt, max_new_tokens=100, device=device)         

if __name__ == "__main__":
    run()
