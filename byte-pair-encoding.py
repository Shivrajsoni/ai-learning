import torch

"""
    Byte Pair Encoding 
"""

def get_list(words):
    idx = []

    for ch in words:
        idx.append(ord(ch))
    
    return idx 
        
def get_pairs(idx):
    counts = {}

    for i in range(len(idx)-1):
        pair = (idx[i] , idx[i+1])
        counts[pair] = counts.get(pair , 0) + 1 
    
    return counts


def merge(idx ,pair , new_id ):
    out = []
    i = 0
    while i < len(idx):
        if i < len(idx)-1 and (idx[i],idx[i+1]) == pair:
            out.append(new_id)
            i+=2
        else:
            out.append(idx[i])
            i+=1 
        
    return out 


def byte_pair_encoding(words, num_merges):
    idx = get_list(words)
    merges = {}

    for step in range(num_merges):
        counts = get_pairs(idx)
        if not counts:
            break
        best = max(counts, key=counts.get)   # most frequent pair
        new_id = 256 + step
        idx = merge(idx, best, new_id)
        merges[best] = new_id
        print(f"step {step}: merged {best} -> {new_id}, length now {len(idx)}")

    return idx, merges


        




def main():

    with open('datasets/input.txt','r') as f  :
        words = f.read()
    idx, merges = byte_pair_encoding(words,10)

    print(idx)
    print(merges)


main()
