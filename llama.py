# goal is to make meta llama-2 model implementation 
from typing import Optional
from sympy.polys.polyoptions import Option
from dataclasses import dataclass
import torch 
import torch.nn as nn 
import torch.functional as F 
import math

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers = 32
    n_head = 32 
    vocab_size = -1 
    n_kv_heads: Optional[None] = None  
    multiple_of: int = 256 
    ffn_dim_multipler: Optional[float] = None
    norm_ops = 1e-5


    max_batch_size = 32
    max_seq_len = 2048 

    device = None 


class Transformer(nn.Module):
    def __init__(self ,args:ModelArgs):
        pass 
        






