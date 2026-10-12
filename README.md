# Inference Engines & Architecture Implementations

From-scratch PyTorch implementations of modern LLM architectures (TinyLlama, LLaMA, and Mistral) with zero-copy weight loading, dynamic/sliding KV-caches, Rotary Position Embeddings (RoPE), and Apple Silicon (MPS) acceleration.

---

## 📂 Architecture Overview

| Module | Model | Key Features | Weights Format |
| :--- | :--- | :--- | :--- |
| [`inference/`](file:///Users/shivraj/Developer/ai-learning/inference/) | **TinyLlama-1.1B** | Grouped Query Attention (GQA), Dynamic KV-Cache, RMSNorm | Single safetensor |
| [`inference-llama/`](file:///Users/shivraj/Developer/ai-learning/inference-llama/) | **LLaMA** | Multi-Head Attention, RoPE, Streaming INT8 Quantization (`QuantizedLinear`), Zero-Copy mmap loader | Sharded safetensors |
| [`mistral/`](file:///Users/shivraj/Developer/ai-learning/mistral/) | **Mistral-7B-v0.1** | Sliding Window Attention (SWA), GQA ($32 \times 8$), Pre-RMSNorm, SwiGLU MLP, MPS acceleration | Multi-shard safetensors |

---

## 1. 🦙 Mistral (`mistral/`)

Clean implementation of the official **Mistral-7B** architecture from first principles.

### Key Components
- **Sliding Window Attention (SWA) & Rolling KV Cache** ([`mistral/sliding_window_attn.py`](file:///Users/shivraj/Developer/ai-learning/mistral/sliding_window_attn.py)):
  - Restricts token attention to a rolling window $W = 4096$: $0 \le i - j < W$.
  - Reduces inference KV cache memory footprint to $O(W)$ instead of growing indefinitely with sequence length.
- **Grouped Query Attention (GQA)**:
  - 32 Query heads paired with 8 Key/Value heads ($4\times$ head repetition).
- **Rotary Position Embeddings (RoPE)**:
  - Precomputed complex sinusoidal frequencies rotated over query and key projections per position.
- **SwiGLU MLP** ([`mistral/transformer.py`](file:///Users/shivraj/Developer/ai-learning/mistral/transformer.py)):
  - Feed-forward layer computing $\text{SwiGLU}(x) = (\text{SiLU}(xW_{\text{gate}}) \odot xW_{\text{up}}) W_{\text{down}}$.
- **Zero-Copy Streaming Loader** ([`mistral/inference.py`](file:///Users/shivraj/Developer/ai-learning/mistral/inference.py)):
  - Memory-maps sharded `safetensors` on disk with `safe_open` and streams weights directly into Apple Silicon MPS device memory.

### How to Run
```bash
# Interactive Mistral inference on Apple MPS:
uv run mistral/inference.py
```

---

## 2. 🦙 LLaMA (`inference-llama/`)

Full LLaMA transformer implementation featuring memory-efficient INT8 quantization and memory mapping.

### Key Components
- **Streaming INT8 Quantization** ([`inference-llama/quantize.py`](file:///Users/shivraj/Developer/ai-learning/inference-llama/quantize.py)):
  - Custom `QuantizedLinear` layer compressing 16-bit float weights down to `int8` with per-channel FP16 scale factors.
  - Cuts memory usage by ~50%, enabling large models to run comfortably on 16GB RAM without disk swap thrashing.
- **Zero-Copy Memory Mapping**:
  - Streams tensor by tensor from disk shards directly into the model buffer, avoiding duplicate RAM copies.
- **Rotary Position Embeddings (RoPE)** ([`inference-llama/multiheadattn.py`](file:///Users/shivraj/Developer/ai-learning/inference-llama/multiheadattn.py)):
  - Rotates query and key token representations with position tracking across prefill and decoding phases.

### How to Run
```bash
uv run python inference-llama/main.py
```

---

## 3. 🦙 TinyLlama (`inference/`)

Lightweight end-to-end inference pipeline for TinyLlama.

### Key Components
- **Grouped Query Attention** ([`inference/group_query_attn.py`](file:///Users/shivraj/Developer/ai-learning/inference/group_query_attn.py)):
  - Optimized multi-query head sharing.
- **Two-Phase Generation Loop** ([`inference/inference.py`](file:///Users/shivraj/Developer/ai-learning/inference/inference.py)):
  - **Prefill Phase**: Encodes prompt token sequence and generates the initial KV-cache.
  - **Decode Phase**: Generates tokens autoregressively one by one reusing cached key/value states.

### How to Run
```bash
uv run python inference/main.py
```

---

## 🚀 Environment & Setup

Built using [`uv`](https://docs.astral.sh/uv/) and PyTorch with Apple Silicon MPS acceleration:

```bash
# Clone the repository
git clone https://github.com/Shivrajsoni/ai-learning.git
cd ai-learning

# Sync dependencies
uv sync

# Run any module
uv run mistral/inference.py
```
