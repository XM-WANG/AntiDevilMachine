import torch
from adm import generateBlock
import time
import argparse

if __name__ == "__main__":
    '''
    1. Replace business code part with your code.
    2. Otherwise it would generate a 5000M tensor to block the GPU.
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('-id', '--gpu_id', type=int, default=0)
    parser.add_argument('-s', '--block_size', type=int, default=22000)
    args = parser.parse_args()
    
    idx =  args.gpu_id
    block_size = args.block_size
    
    device = torch.device(f"cuda:{idx}")

    ### business code ###
    block = generateBlock(block_size)
    block.to(device)
    while True:
        time.sleep(1)