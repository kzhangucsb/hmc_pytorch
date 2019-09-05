"""
Created on Mon Aug 26 20:52:58 2019

@author: zkq
"""


"""
Created on Wed Aug  7 10:47:36 2019

@author: zkq
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter, ParameterList
import numpy as np
from tensor_layer import TTConv2d, TTlinear





cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}





class vgg11_TT(nn.Module):
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    def __init__(self, rank=[[64], [64], [64], [32, 32], [32, 32], [32, 32], [32, 32], 
                             [64, 64], [64, 64], [64]]):
        super(vgg11_TT, self).__init__()
        assert(len(rank) == 10)
        beta_conv = 1
        beta_fc = 1
        self.conv0  = nn.Conv2d(3, 64, 3, padding=1)
        self.conv1  = TTConv2d((8, 8), (16, 8), 3, rank[0], padding=1, beta=beta_conv)
        self.conv2  = TTConv2d((16, 8), (16, 16), 3, rank[1], padding=1, beta=beta_conv)
        self.conv3  = TTConv2d((16, 16), (16, 16), 3, rank[2], padding=1, beta=beta_conv)
        self.conv4  = TTConv2d((8, 8, 4), (8, 8, 8), 3, rank[3], padding=1, beta=beta_conv)
        self.conv5  = TTConv2d((8, 8, 8), (8, 8, 8), 3, rank[4], padding=1, beta=beta_conv)
        self.conv6  = TTConv2d((8, 8, 8), (8, 8, 8), 3, rank[5], padding=1, beta=beta_conv)
        self.conv7  = TTConv2d((8, 8, 8), (8, 8, 8), 3, rank[6], padding=1, beta=beta_conv)
        
        self.bn0    = nn.BatchNorm2d(64)
        self.bn1    = nn.BatchNorm2d(128)
        self.bn2    = nn.BatchNorm2d(256)
        self.bn3    = nn.BatchNorm2d(256)
        self.bn4    = nn.BatchNorm2d(512)
        self.bn5    = nn.BatchNorm2d(512)
        self.bn6    = nn.BatchNorm2d(512)
        self.bn7    = nn.BatchNorm2d(512)
        
        self.fc0 = TTlinear((32, 16, 49), (16, 16, 16), rank[7], beta=beta_fc)
        self.fc1 = TTlinear((16, 16, 16), (16, 16, 16), rank[8], beta=beta_fc)
        self.fc2 = TTlinear((64, 64), (2, 5), rank[9], beta=beta_fc)
        
        self.dropout0 = nn.Dropout()
        self.dropout1 = nn.Dropout()
        
class vggBC(nn.Module):
    
    def __init__(self):
        super(vggBC, self).__init__()
        beta_conv = 5
#        beta_fc = 5
        self.conv0  = nn.Conv2d(3, 128, 3, padding=1)
        self.conv1  = nn.Conv2d(128, 256, 3, padding=1)
        self.conv2  = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3  = nn.Conv2d(256, 256, 3, padding=1)
        
            
        self.bn0    = nn.BatchNorm2d(128)
        self.bn1    = nn.BatchNorm2d(256)
        self.bn2    = nn.BatchNorm2d(256)
#        self.bn3    = nn.BatchNorm2d(256)
        self.bn3    = nn.BatchNorm1d(256*8*8)
        self.bn4    = nn.BatchNorm1d(512)
        
        self.fc0 = nn.Linear(256*8*8, 512)
        self.fc1 = nn.Linear(512, 10)
        
#        self.dropout0 = nn.Dropout()
#        self.dropout1 = nn.Dropout()
        
        
        
    
    def forward(self, x):
        x = F.relu(self.bn0(self.conv0(x)))
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        x = x.reshape((x.shape[0], -1)) 
        x = self.bn3(x)
        x = F.relu(self.bn4(self.fc0(x)))
#        x = self.dropout0(x)
        x = self.fc1(x)
        return x

    
        
class vggBC_TT(nn.Module):
    
    def __init__(self, rank=[[16, 32, 16], [16, 32, 16], [16, 32, 16], [64, 64], [32]]):
        super(vggBC_TT, self).__init__()
        assert(len(rank) == 5)
        beta_conv = 5
#        beta_fc = 5
        self.conv0  = nn.Conv2d(3, 128, 3, padding=1)
        self.conv1  = TTConv2d((16, 1, 8, 1), (1, 16, 1, 16), 3, rank[0], padding=1, beta=beta_conv)
        self.conv2  = TTConv2d((16, 1, 16, 1), (1, 16, 1, 16), 3, rank[1], padding=1, beta=beta_conv)
        self.conv3  = TTConv2d((16, 1, 16, 1), (1, 16, 1, 16), 3, rank[2], padding=1, beta=beta_conv)
        
            
        self.bn0    = nn.BatchNorm2d(128)
        self.bn1    = nn.BatchNorm2d(256)
        self.bn2    = nn.BatchNorm2d(256)
#        self.bn3    = nn.BatchNorm2d(256)
        self.bn3    = nn.BatchNorm1d(256*8*8)
        self.bn4    = nn.BatchNorm1d(512)
        
        self.fc0 = TTlinear((16, 16, 64), (4, 8, 16), rank[3], beta=0.1)
        self.fc1 = TTlinear((32, 16), (5, 2), rank[4], beta=1)
        
#        self.dropout0 = nn.Dropout()
#        self.dropout1 = nn.Dropout()
        
        
        
    
    def forward(self, x):
        x = F.relu(self.bn0(self.conv0(x)))
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        x = x.reshape((x.shape[0], -1)) 
        x = self.bn3(x)
        x = F.relu(self.bn4(self.fc0(x)))
#        x = self.dropout0(x)
        x = self.fc1(x)
        return x

    def regularizer(self, exp=True):
        ret = 0
        for i in range(1, 4):
            ret += getattr(self, 'conv{}'.format(i)).regularizer(exp=exp)
        for i in range(2):
            ret += getattr(self, 'fc{}'.format(i)).regularizer(exp=exp)
        return ret