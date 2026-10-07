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
            # Prefill = understand/process the prompt and build the initial KV cache.
            # Prefill Phase : Process the Input Prompt 
                logits , kv_caches = self.model(input_ids,start_pos=0,kv_caches = None) # [1 , T , 32000]
                next_token_logits = logits[:,-1,:] # [1,32000]
                
                # Greedy Decoding : pick highest Prob token 
                next_token_id = torch.argmax(next_token_logits , dim=-1 , keepdim=True)
                
                token_val = next_token_id.item()

                if token_val == 2:
                    return self.tokenizer.decode(generated_tokens)

                generated_tokens.append(token_val)

                start_pos = input_ids.shape[1] # start_pos is now len(1)

                current_token = next_token_id                
            
            # Decode = generate new tokens one-by-one while reusing that cache.
            # 2 Decode Phase : Autoregressive token-by-token generation with kv_cache 
                for _ in range(max_new_tokens - 1):
                    logits , kv_caches = self.model(current_token,start_pos=start_pos,kv_caches=kv_caches)
                    next_token_id = torch.argmax(logits[:,-1,:],dim=-1,keepdim=True)

                    token_val = next_token_id.item()

                    if token_val == 2 : # EOS Token 
                        break 

                    generated_tokens.append(token_val)

                    current_token = next_token_id
                    start_pos+=1 
            
        
                return self.tokenizer.decode(generated_tokens)


