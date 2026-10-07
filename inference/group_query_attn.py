import math
import torch
import torch.nn as nn 
import torch.nn.functional as F 

# class SelfAttention(nn.Module):
#     def __init__(self,d_model):
#         # Query , Key , Vector Implementation 
#         super().__init__()
#         self.d_model = d_model 

#         self.w_q = nn.Linear(d_model,d_model,bias=False)
#         self.w_k = nn.Linear(d_model,d_model,bias=False)
#         self.w_v = nn.Linear(d_model,d_model,bias=False)

#     def forward(self , x: torch.Tensor)-> torch.Tensor:
#         Q = self.w_q(x)
#         K = self.w_k(x)
#         V = self.w_v(x)

#         # Q @ K^T -> shape: (batch_size, seq_len, seq_len)

#         scores = torch.matmul(Q,K.transpose(-2,-1)) / math.sqrt(self.d_model)

#         # print(" Scores after Multiplying is : ",scores) 

#         # Softmax along the last dimension 
#         attn_wt = torch.softmax(scores,dim=-1)

#         # print(" Attention Weight is : ",attn_wt) 

#         # Multiply by V -> Shape :
#         output = torch.matmul(attn_wt , V)

#         # print(" Output is : ",output) 

#         return output

    
def precompute_rope(head_dim,max_seq_len,base=10000.0,device=None):
    # theta_i = base^(-2i/head_dim), shape: (head_dim/2,)
    i = torch.arange(0,head_dim,2, device=device).float()
    inv_freq = 1.0 / (base ** (i/head_dim))

    # positions : (max_seq_len , )
    pos = torch.arange(max_seq_len,device=None).float()

    # angle[m, i] = m * theta_i  -> (max_seq_len, head_dim/2)
    angles = torch.outer(pos,inv_freq)
    return torch.cos(angles) ,torch.sin(angles)


def rotate_half(x):
    """Splits vector into two halves: [-x2, x1]"""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[...,x.shape[-1] // 2 :]
    return torch.cat((-x2,x1) , dim=-1)


def apply_rope(x,cos,sin):
    # x : (B, H, T, D)  cos/sin : (max_seq_len , D / 2 )
    T = x.size(2)
    cos = cos[:T][None , None , : , :]  #[1,1,T,D/2]
    sin = sin[:T][None , None , : , :]

    cos = torch.cat([cos , cos] , dim=-1)
    sin = torch.cat([sin , sin] , dim=-1) 
    return (x * cos) + (rotate_half(x) * sin)
    

def repeat_kv(x:torch.Tensor,n_rep: int)-> torch.Tensor:
    """
    [B, num_kv_heads, T, head_dim] -> [B, num_q_heads, T, head_dim]
    """
    if n_rep == 1:
        return x 
    B , num_kv_heads , T , head_dim = x.shape

    # Repeat each head n_rep times 
    return x.repeat_interleave(n_rep ,dim=1)  


class GroupQueryAttention(nn.Module):
    def __init__(self , d_model , num_heads ,num_kv_heads ,max_seq_len = 2048): # d_model is the dimension of tokens in which it is represented , d_head is number of head in multiple attn we will create 
        super().__init__()

        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.num_queries_per_kv = num_heads // num_kv_heads 
        self.num_kv_heads = num_kv_heads
        assert self.head_dim % 2 == 0 , "ROPE needs an even head_dim"
        
        self.w_q = nn.Linear(d_model , num_heads * self.head_dim ,bias=False)
        self.w_k = nn.Linear(d_model , num_kv_heads * self.head_dim ,bias=False)
        self.w_v = nn.Linear(d_model , num_kv_heads * self.head_dim ,bias=False)
        self.w_o = nn.Linear(num_heads * self.head_dim ,d_model  ,bias=False) 

        cos, sin = precompute_rope(self.head_dim, max_seq_len)
        self.register_buffer("cos",cos,persistent=False)
        self.register_buffer("sin",sin,persistent=False)
    

    def forward(self , x):
        B , T , C = x.shape

        # Generates Q , K , V    
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)

        # [ B , T , C ] Batch , Token Seq , Dimension 

        Q = Q.view(B , T , self.num_heads , self.head_dim)
        K = K.view(B , T , self.num_kv_heads , self.head_dim)
        V = V.view(B , T , self.num_kv_heads , self.head_dim)
        
        # (B , T , H , D)  -> ( B , H , T , D)
        Q = Q.transpose(1,2)
        K = K.transpose(1,2)
        V = V.transpose(1,2)

        # RoPE: rotate Q and K only
        Q = apply_rope(Q, self.cos, self.sin)
        K = apply_rope(K, self.cos, self.sin)
        
        K = repeat_kv(K,self.num_queries_per_kv)
        V = repeat_kv(V,self.num_queries_per_kv)

        scores = Q @ K.transpose(-2,-1)
        scores = scores / math.sqrt(self.head_dim)

        # mask creation 
        mask = torch.triu(
            torch.ones(T , T , device=x.device),
            diagonal=1 
        ).bool()
        
        # mask implementation 
        scores = scores.masked_fill(mask , float("-inf"))
        
        # softmax implementation 
        attn = torch.softmax(scores , dim=-1)

        output = attn @ V 

        #[B,H,T,D] -> [B,T,H,D]
        output = output.transpose(2,1)

        output = output.contiguous().view(B,T,self.d_model)

        # Final  Projection 
        output = self.w_o(output)
        
        return output
