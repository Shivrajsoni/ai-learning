import torch
import torch.nn as nn 

class Demo(nn.Module):
    def __init__(self , dim ):
        super().__init__()
        self.dim = dim 
        ans = {}
        one_hot = torch.zeros(dim) 
        
        for i in range(dim):
           one_hot[i] = 1
           ans[i] = one_hot
           one_hot = torch.zeros(dim)

        print(ans)


        