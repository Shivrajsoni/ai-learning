from functorch import dim
import math
import torch
from torch.nn import ReLU
import torch.nn as nn 

class SelfAttention(nn.Module):
    def __init__(self,d_model):
        # Query , Key , Vector Implementation 
        super().__init__()
        self.d_model = d_model 

        self.w_q = nn.Linear(d_model,d_model,bias=False)
        self.w_k = nn.Linear(d_model,d_model,bias=False)
        self.w_v = nn.Linear(d_model,d_model,bias=False)
        self.w_o = nn.Linear(d_model,d_model,bias=False)

    def forward(self , x: torch.Tensor)-> torch.Tensor:
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)

        # Q @ K^T -> shape: (batch_size, seq_len, seq_len)

        scores = torch.matmul(Q,K.transpose(-2,-1)) / math.sqrt(self.d_model)

        # print(" Scores after Multiplying is : ",scores) 

        # Softmax along the last dimension 
        attn_wt = torch.softmax(scores,dim=-1)

        # print(" Attention Weight is : ",attn_wt) 

        # Multiply by V -> Shape :
        output = torch.matmul(attn_wt , V)

        # print(" Output is : ",output) 

        return output


class MultiheadedSelfAttention(nn.Module):
    def __init__(self , d_model , num_heads): # d_model is the dimension of tokens in which it is represented , d_head is number of head in multiple attn we will create 
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.w_q = nn.Linear(d_model , d_model ,bias=False)
        self.w_k = nn.Linear(d_model , d_model ,bias=False)
        self.w_v = nn.Linear(d_model , d_model ,bias=False)
    
    def forward(self , x):
        B , T , C = x.shape

        # Generates Q , K , V    
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)

        # [ B , T , C ] Batch , Token Seq , Dimension 

        Q = Q.view(B , T , self.num_heads , self.head_dim)
        K = K.view(B , T , self.num_heads , self.head_dim)
        V = V.view(B , T , self.num_heads , self.head_dim)
        
        # (B , T , H , D)  -> ( B , H , T , D)
        Q = Q.transpose(1,2)
        K = K.transpose(1,2)
        V = V.transpose(1,2)

        scores = Q @ K.transpose(-2,-1)
        scores = scores / math.sqrt(self.head_dim)

        attn = torch.softmax(scores , dim=-1)

        output = attn @ V 

        #[B,H,T,D] -> [B,T,H,D]
        output = output.transpose(2,1)

        output = output.contiguous().view(B,T,self.d_model)

        # Final  Projection 

        output = self.w_o(output)
        return output 








        