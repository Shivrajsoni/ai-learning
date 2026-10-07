import json
from inference.inference import Inference


with open("./tinyllama-weights/config.json", "r") as f:
    config = json.load(f)

inf = Inference(config)

while True:
    prompt = input("\nAsk anything  >> ")
    
    if prompt  in ("quit", "exit"):
        print("Goodbye")
        break 
    
    formatted_prompt = f"<|user|>\n{prompt}</s>\n<|assistant|>\n"
    out = inf.generate(formatted_prompt,max_new_tokens=60)
    print(out)
