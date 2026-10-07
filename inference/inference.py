from inference.transformer import TinyLlamaModel
import torch 
from safetensors.torch import load_file
from tokenizers import Tokenizer
# Complete Pipeline 
class Inference:
    def __init__(self,config):
        self.tokenizer = Tokenizer.from_file("./tinyllama-weights/tokenizer.json")
        self.model = TinyLlamaModel(config)        
        # Read Safetensors file 
        weights = load_file("./tinyllama-weights/model.safetensors")
        # Copy weights into Model 
        self.model.load_weights(weights)
        # set to eval mode (no grad need for inference )
        self.model.eval()

    def generate(self,prompt:str,max_new_tokens=50):
        # Encode Prompt 
        tokens = self.tokenizer.encode(prompt).ids
        input_ids = torch.tensor([tokens], dtype=torch.long) # [1,T]

        generated_tokens = []

        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = self.model(input_ids) # [1 , T , 32000]
                next_token_logits = logits[:,-1,:] # [1,32000]
                
                # Greedy Decoding : pick highest Prob token 
                next_token_id = torch.argmax(next_token_logits , dim=-1 , keepdim=True)
                
                token_val = next_token_id.item()

                if token_val == 2:
                    break 
                generated_tokens.append(token_val)

                input_ids = torch.cat([input_ids, next_token_id],dim=1)
            
        
        return self.tokenizer.decode(generated_tokens)


