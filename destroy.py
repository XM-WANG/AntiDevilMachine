import torch
from adm import generateBlock
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-idx', '--cuda_idx', type=int, default=0)
parser.add_argument('-s', '--block_size', type=int, default=10000)
args = parser.parse_args()
    

idx = args.cuda_idx
size = args.block_size

device = torch.device(f"cuda:{idx}")

replaced = False
block = generateBlock(size)
while not replaced:
    try:
        block.to(device)
    except:
        time.sleep(0.1)
        size -= 1
        block = generateBlock(size)
    else:
        replaced = True
print(f"got best size {size}")
while replaced:
    time.sleep(1)

