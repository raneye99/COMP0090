import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

#decalre new module
class LinearNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(3*32*32, 1024)
        self.linear2 = nn.Linear(1024, 512)
        self.linear3 = nn.Linear(512, 256)
        self.linear4 = nn.Linear(256,10)

    def forward(self,x):
        x = torch.flatten(x,1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        x = F.relu(x)
        x = self.linear4(x)
        return x