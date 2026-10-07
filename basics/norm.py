import torch
import torch.nn as nn 
class CustomLayerNorm(nn.Module):
    def __init__(self, normalized_shape: int , eps = 1e-5):
        super().__init__()
        self.eps = eps 
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self , x: torch.Tensor)-> torch.Tensor:
        mean = x.mean(dim=-1,keepdim=True)
        var = x.var(dim=-1 ,unbiased=False, keepdim=True)

        # Normalize
        x_hat = (x - mean) / torch.sqrt(var + self.eps)

        return self.gamma * x_hat + self.beta 


import numpy as np 

class CustomBatcNorm(nn.Module):
    def __init__(self , dim , eps = 1e-5 , momentum = 0.1):
        super().__init__()
        self.eps = eps 
        self.momentun = momentum
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        
        # Running statistics are buffers, not trainable parameters
        self.register_buffer("running_mean", torch.zeros(dim))
        self.register_buffer("running_var", torch.ones(dim))

    def forward(self , x , training = True):
        if training:
            mean = x.mean(axis = 0)
            var  = x.var(axis = 0)
            self.running_mean = (1 - self.momentun) * self.running_mean + self.momentun * mean 
            self.running_var = (1 - self.running_var) * self.running_var + self.momentun * var

        else:
            mean , var = self.running_mean , self.running_var
        
        x_hat = (x - mean) / np.sqrt(var + self.eps)

        return self.gamma * x_hat + self.beta


class CustomRMSNorm(nn.Module):
    def __init__(self, dim ,eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self , x: torch.Tensor)->  torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.gamma * (x / rms)
