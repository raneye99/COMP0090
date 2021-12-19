#implement DenseNet architecture into image classification tutorial

import torch
import torch.nn as nn
import torch.nn.functional as F

class Dense_Layer(nn.Module):
    def __init__(self, n_in, n_out):
        super(Dense_Layer, self).__init__()
        self.bndl = nn.BatchNorm2d(n_in)
        self.reludl = nn.ReLU(inplace=True)
        self.convdl = nn.Conv2d(n_in, n_out, kernel_size=3, stride=1, padding=1, bias=False)
    def forward(self,x):
        out = self.convdl(self.reludl(self.bndl(x)))
        x = torch.cat((x,out),1)
        return x

class Transition_Layer(nn.Module):
    def __init__(self, n_in, n_out):
        super(Transition_Layer, self).__init__()
        self.bntl = nn.BatchNorm2d(n_in)
        self.relutl = nn.ReLU(inplace=True)
        self.convtl = nn.Conv2d(n_in, n_out,kernel_size=1,stride=1,padding = 0,bias=False)
        self.pooltl = nn.AvgPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        x = self.pooltl(self.convtl(self.relutl(self.bntl(x))))
        return x

class Dense_Block(nn.Module):
    def __init__(self, n_in, growth):
        super(Dense_Block, self).__init__()
        self.layer = self.dense_block(n_in, growth)
    def dense_block(self, n_in, growth):
        layers = []
        for i in range(0,4):
            layers.append(Dense_Layer(n_in + i*growth, growth))
        return nn.Sequential(*layers)
    def forward(self,x):
        return self.layer(x)

class DenseNet3(nn.Module):
    def __init__(self):
        super(DenseNet3, self).__init__()
        #initial convolution
        self.conv0 = nn.Conv2d(3, 32, kernel_size=3, stride = 1, padding=1, bias=False)
        
        #first denseblock
        self.dense1 = Dense_Block(n_in = 32, growth = 32)
        #note since there are 4 layers the output of denseblock will have 6 +4*16 channels
        self.trans1 = Transition_Layer(n_in = 160, n_out = 80)

        #second denseblock
        self.dense2 = Dense_Block(n_in = 80, growth = 32)
        #note since there are 4 layers the output of denseblock will have 35 + 4*16
        self.trans2 = Transition_Layer(n_in = 208, n_out=104)

        #third denseblock
        self.dense3 = Dense_Block(n_in=104,growth=32)

        #global average pooling, classifier
        self.bnend =  nn.BatchNorm2d(232)
        self.classifier = nn.Linear(232, 10)

    def forward(self,x):
        out = self.conv0(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)
        out = F.relu(self.bnend(out))
        out = F.adaptive_avg_pool2d(out, (1,1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return out