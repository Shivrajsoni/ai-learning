import pytest
import torch 
from multiheadattn import MultiHeadAttn , precompute_rope , apply_rope , rotate_half 

def test_precompute_rope_shapes():
    head_dim = 16
    max_seq_len = 32 
    cos , sin = precompute_rope(head_dim=head_dim, max_seq_len=max_seq_len)
    assert cos.shape == (max_seq_len , head_dim // 2 )
    assert sin.shape == (max_seq_len , head_dim // 2 )


def test_rotate_half_norm_variant():
    x = torch.randn(2,4,8,16)
    rotated = rotate_half(x) 

    assert rotated.shape == x.shape
    torch.testing.assert_close(torch.norm(x,dim=-1) , torch.norm(rotated,dim=-1))


@pytest.fixture 
def attn_module():
    torch.manual_seed(42)
    return MultiHeadAttn(d_model=64 , num_heads=4, max_seq_len=128) 

def test_forward_output_shape(attn_module):
    """Test standard forward shape contract: (B, T, C) -> (B, T, C)."""
    B, T, C = 2, 16, 64
    x = torch.randn(B, T, C)
    
    out = attn_module(x)
    assert out.shape == (B, T, C), f"Expected shape {(B, T, C)}, got {out.shape}"

def test_single_token_forward(attn_module):
    """Verify inference works for prompt/decoding steps (T=1)."""
    B, T, C = 1, 1, 64
    x = torch.randn(B, T, C)
    
    out = attn_module(x)
    assert out.shape == (B, 1, C)
    assert not torch.isnan(out).any()

def test_causal_masking_invariant(attn_module):
    """
    First-Principles Causal Test:
    Modifying future tokens (e.g., token at index T-1) must NEVER
    affect outputs of earlier tokens (e.g., token at index 0).
    """
    B, T, C = 1, 4, 64
    x1 = torch.randn(B, T, C)
    x2 = x1.clone()
    
    # Mutate the last token in x2
    x2[:, -1, :] += 5.0
    out1 = attn_module(x1)
    out2 = attn_module(x2)
    # First T-1 outputs must be strictly identical
    torch.testing.assert_close(out1[:, :-1, :], out2[:, :-1, :])

def test_gradient_flow(attn_module):
    """Ensure all parameters receive gradients during backprop."""
    x = torch.randn(2, 8, 64, requires_grad=True)
    out = attn_module(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    for name, param in attn_module.named_parameters():
        assert param.grad is not None, f"No grad for {name}"
