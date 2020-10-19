import torch
import torch.nn as nn
import time
from pynvml import *

'''
雪中送炭三九暖， 视若无睹腊月寒。
恶意占卡业障重， 毕业之后九九六。
'''


def generateBlock(expectedSize=1000):
    ''' Generate a random tensor which would be used to take up GPU memory.
    Parameters
    ----------
    expectedSize : int
        The size (M) of tensor which would be generated. Default is 1000.
    
    Returns
    -------
    block : Torch.FloatTensor.
        A tensor which would be used to take up GPU memory.
    '''
    tensorSize = ((expectedSize-554)*1024*1024)//4
    block = torch.FloatTensor(tensorSize)
    return block

def listenEmptyGPU():
    ''' Listen to all gpus on the server. Return the first empty GPU index.
    Returns
    -------
    idx : int.
        The index of the empty GPU.
    '''
    search = True
    nvmlInit()
    deviceCount = nvmlDeviceGetCount()
    print("start listening...")
    while search:
        time.sleep(1)
        for i in range(deviceCount):
            handle = nvmlDeviceGetHandleByIndex(i)
            info = nvmlDeviceGetMemoryInfo(handle)
            total = info.total
            used = info.used
            rate = used / total
            if rate < 0.1:
                print(f"cuda:{i} is empty! Fuck it now!")
                search = False
                break
    nvmlShutdown()
    return i

if __name__ == "__main__":
    '''
    1. Replace business code part with your code.
    2. Otherwise it would generate a 5000M tensor to block the GPU.
    '''
    idx = listenEmptyGPU()
    device = torch.device(f"cuda:{idx}")
    
    ### business code ###
    block = generateBlock(5000)
    block.to(device)
    while True:
        time.sleep(1)
    #####################