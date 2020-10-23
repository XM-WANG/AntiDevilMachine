import torch
from adm import generateBlock, listenEmptyGPU

# def Abdicate(device,)

if __name__ == "__main__":
    '''
    1. Replace business code part with your code.
    2. Otherwise it would generate a 5000M tensor to block the GPU.
    '''
    idx = listenEmptyGPU()
    device = torch.device(f"cuda:{idx}")
    
    ### business code ###
    block = generateBlock(8000)
    block.to(device)
    while True:
        time.sleep(1)
    #####################