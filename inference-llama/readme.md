# LLaMA Inference Engine

From-scratch implementation of the LLaMA decoder-only transformer with memory-mapped zero-copy weight streaming and on-the-fly INT8 quantization.

---

## 🏗️ Architecture & Implementation

1. **Multi-Head Attention with RoPE** ([`multiheadattn.py`](file:///Users/shivraj/Developer/ai-learning/inference-llama/multiheadattn.py)):
   - Rotary Position Embeddings (RoPE) precomputed via inverse frequencies and applied through 2D rotation to Query and Key representations.
   - Dynamic KV-Cache appending keys and values during generation to avoid recalculating past context ($O(1)$ decoding per token).

2. **SwiGLU Feed-Forward Network** ([`llama_block.py`](file:///Users/shivraj/Developer/ai-learning/inference-llama/llama_block.py)):
   - Dual-projection gating: $\text{SwiGLU}(x) = (\text{SiLU}(xW_{\text{gate}}) \odot xW_{\text{up}}) W_{\text{down}}$.

3. **On-the-Fly INT8 Quantization** ([`quantize.py`](file:///Users/shivraj/Developer/ai-learning/inference-llama/quantize.py)):
   - **`QuantizedLinear`**: Stores weights as `torch.int8` alongside a per-channel FP16 scale tensor ($W_{\text{fp16}} = W_{\text{int8}} \times \text{scale}$).
   - **Zero-Copy Streaming**: Reads shard-by-shard from disk via `safe_open`, quantizes directly into INT8 buffers, and releases FP16 buffers immediately to keep peak RAM under ~7.5 GB on 16GB systems.

---

## 🚀 How to Run

```bash
# Run inference pipeline
uv run python inference-llama/main.py

# Run unit tests for multihead attention
uv run pytest inference-llama/test_multiheadattn.py
```
