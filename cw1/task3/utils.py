import numpy as np
import torch
import torchvision
#adapted from https://arxiv.org/abs/1708.04552

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import time

def cutout(n, s, img):
    '''
    n = number of cutouts
    s = max length of a cutout side
    img = a tensor image of size (c, h, w)

    returns a tensor image with cutout
    '''
        
    #get hieght and width of image
    h = img.size(1)
    w = img.size(2)

    # print(h,w)

    #initialize mask of same size as image
    mask = np.ones((h,w), np.float32)
    # print(mask)

    for j in range(n):

        #pick random location to be center of mask
        y = np.random.randint(h)
        x = np.random.randint(w)

        # print(y,x)

        #determine length of square mask
        l = np.random.randint(s)
        # l = s

        #apply shape to mask
        y_low = np.clip(y-l//2, 0, h)
        y_high = np.clip(y+l//2, 0, h)
        x_low = np.clip(x-l//2,0,w)
        x_high = np.clip(x+l//2,0,w)

        mask[y_low:y_high, x_low:x_high] = 0.


    # print(mask)
    #transform mask to tensor
    mask = torch.from_numpy(mask)
    mask = mask.expand_as(img)
    img = img*mask

    return img

