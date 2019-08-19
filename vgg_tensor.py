#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 10:47:36 2019

@author: zkq
"""

import torch
import torch.nn as nn
import tensorly as tl
from torch.nn import Parameter, ParameterList
import numpy as np
tl.set_backend('pytorch')


class tensorizedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, in_rank, out_rank, alpha = 1, beta = 5, **kwargs):
        super(tensorizedConv2d, self).__init__()
        self.linear_pre = nn.Linear(in_channels, in_rank)
        self.conv = nn.Conv2d(in_rank, out_rank, **kwargs)
        self.linear_post = nn.Linear(out_rank, out_channels)
        self.lamb_in  = Parameter(torch.ones(in_rank))
        self.lamb_out = Parameter(torch.ones(out_rank))
        self.alpha = alpha
        self.beta = beta
        
        
        self._initialize_weights()
        
    def forward(self, x):
        x = torch.transpose(x, 1, 3)
        x = self.linear_pre(x)
        x = torch.transpose(x, 1, 3)
        
        x = self.conv(x)
        
        x = torch.transpose(x, 1, 3)
        x = self.linear_post(x)
        x = torch.transpose(x, 1, 3)
        return x
    
    def _initialize_weights(self):
        pass
    
    def regularizer(self):
        self.lamb_in.data.clamp_min_(1e-6)
        self.lamb_out.data.clamp_min_(1e-6)
        ret = 0
        ret += torch.sum(torch.sum(self.linear_pre.weight**2, dim=1) / self.lamb_in)
        ret += torch.sum(torch.log(self.lamb_in)) / 2
        ret += torch.sum(torch.sum(self.linear_post.weight**2, dim=0) / self.lamb_out)
        ret += torch.sum(torch.log(self.lamb_out)) / 2
        ret += torch.sum(self.conv.weight**2)
        
        ret += self.beta * torch.sum(self.lamb_in)
        ret += self.beta * torch.sum(self.lamb_out)
        ret += (self.alpha + 1) * torch.sum(torch.log(self.lamb_in))
        ret += (self.alpha + 1) * torch.sum(torch.log(self.lamb_out))
        return ret
        
class tensorizedlinear(nn.Module):
    def __init__(self, in_size, out_size, in_rank, out_rank,  alpha = 1, beta = 5, **kwargs):
        super(tensorizedlinear, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.in_rank = in_rank
        self.out_rank = out_rank
        self.factors_in  = ParameterList([Parameter(torch.Tensor(r, s)) for (r, s) in zip(in_rank, in_size)])
        self.factors_out = ParameterList([Parameter(torch.Tensor(s, r)) for (r, s) in zip(out_rank, out_size)])
        self.core = Parameter(torch.Tensor(np.prod(out_rank), np.prod(in_rank)))
        self.bias = Parameter(torch.Tensor(np.prod(out_rank)))
        self.lamb_in  = ParameterList([Parameter(torch.ones(r)) for r in in_rank])
        self.lamb_out = ParameterList([Parameter(torch.ones(r)) for r in out_rank])
        self.alpha = alpha
        self.beta = beta
        self._initialize_weights()
        
    def forward(self, x):
        x = x.reshape((x.shape[0], *self.in_size))
        for i in range(len(self.factors_in)):
            x = tl.tenalg.mode_dot(x, self.factors_in[i], i+1)
        x = x.reshape((x.shape[0], -1))
        x = torch.nn.functional.linear(x, self.core, self.bias)
        x = x.reshape((x.shape[0], *self.out_rank))
        for i in range(len(self.factors_out)):
            x = tl.tenalg.mode_dot(x, self.factors_out[i], i+1)
        x = x.reshape((x.shape[0], -1))
        x /= np.prod(self.out_rank) **0.5
        return x
    
    def _initialize_weights(self):
        for f in self.factors_in:
            nn.init.kaiming_uniform_(f)
        for f in self.factors_out:
            nn.init.kaiming_uniform_(f)
        nn.init.kaiming_uniform_(self.core)
#        self.core.data /= np.prod(self.out_rank) **0.5
        nn.init.constant_(self.bias, 0)
        
    def regularizer(self):
        ret = 0
        for l, f in zip(self.lamb_in, self.factors_in):
            l.data.clamp_min_(1e-6)
            ret += torch.sum(torch.sum(f**2, dim=1) / l)
            ret += torch.sum(torch.log(l)) / 2
            ret += self.beta * torch.sum(l)
            ret += (self.alpha + 1) * torch.sum(torch.log(l))
        for l, f in zip(self.lamb_out, self.factors_out):
            l.data.clamp_min_(1e-6)
            ret += torch.sum(torch.sum(f**2, dim=0) / l)
            ret += torch.sum(torch.log(l)) / 2
            ret += self.beta * torch.sum(l)
            ret += (self.alpha + 1) * torch.sum(torch.log(l))
        ret += torch.sum(self.core ** 2) 
        return ret

class VGG(nn.Module):

    def __init__(self, features, tensorized, num_classes=10, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        if tensorized is None:
            tensorized = [None]*3
        if tensorized[0] is None:
            l1 = nn.Linear(512 * 7 * 7, 4096)
        else:
            l1 = tensorizedlinear((512, 49), (64, 64), tensorized[0][0], tensorized[0][1])
        if tensorized[1] is None:
            l2 = nn.Linear(4096, 4096)
        else:
            l2 = tensorizedlinear((64, 64), (64, 64), tensorized[1][0], tensorized[1][1])
        if tensorized[2] is None:
            l3 = nn.Linear(4096, num_classes)
        else:
            l3 = tensorizedlinear((64, 64), (num_classes,), tensorized[2][0], tensorized[2][1])
            
        
        self.classifier = nn.Sequential(
            l1,
            nn.ReLU(True),
            nn.Dropout(),
            l2,
            nn.ReLU(True),
            nn.Dropout(),
            l3,
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


    def regularizer(self):
        ret = 0
        for m in self.modules():
            if isinstance(m, tensorizedlinear) or isinstance(m, tensorizedConv2d):
                ret += m.regularizer()
                
        return ret
                

def make_layers(cfg, tensorized, batch_norm=False):
    layers = []
    in_channels = 3
    for i, v in enumerate(cfg):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            if tensorized is not None and tensorized[i] is not None:
                (in_rank, out_rank) = tensorized[i]
                conv2d = tensorizedConv2d(in_channels, v, in_rank, out_rank, kernel_size=3, padding=1)
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch, cfg, batch_norm, tensorized, **kwargs):
    if tensorized is None:
        t1 = None
        t2 = None
    else:
        assert len(tensorized) == len(cfgs[cfg])+3
        t1 = tensorized[:len(cfgs[cfg])]
        t2 = tensorized[len(cfgs[cfg]):]
    model = VGG(make_layers(cfgs[cfg], t1, batch_norm=batch_norm), t2, **kwargs)
    return model


def vgg11(tensorized, **kwargs):
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg11', 'A', False, tensorized, **kwargs)


def vgg11_bn(tensorized, **kwargs):
    r"""VGG 11-layer model (configuration "A") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg11_bn', 'A', True, tensorized, **kwargs)


def vgg13(tensorized, **kwargs):
    r"""VGG 13-layer model (configuration "B")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg13', 'B', False, tensorized, **kwargs)


def vgg13_bn(tensorized, **kwargs):
    r"""VGG 13-layer model (configuration "B") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg13_bn', 'B', True, tensorized, **kwargs)


def vgg16(tensorized, **kwargs):
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16', 'D', False, tensorized, **kwargs)


def vgg16_bn(tensorized, **kwargs):
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16_bn', 'D', True, tensorized, **kwargs)


def vgg19(tensorized, **kwargs):
    r"""VGG 19-layer model (configuration "E")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg19', 'E', False, tensorized, **kwargs)


def vgg19_bn(tensorized, **kwargs):
    r"""VGG 19-layer model (configuration 'E') with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg19_bn', 'E', True, tensorized, **kwargs)