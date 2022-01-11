import torch
from adm import generateBlock, listenEmptyGPU
import time
import argparse


if __name__ == "__main__":
    '''
    1. Replace business code part with your code.
    2. Otherwise it would generate a 5000M tensor to block the GPU.
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--block_size', type=int, default=10000)
    args = parser.parse_args()

    idx = listenEmptyGPU()
    device = torch.device(f"cuda:{idx}")
    
    block_size = args.block_size

    ### business code ###
    block = generateBlock(block_size)
    block.to(device)
    while True:
        time.sleep(1)
    #####################
