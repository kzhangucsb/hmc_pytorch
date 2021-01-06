"""
Created on Fri Dec  6 15:29:07 2019

@author: zkq
"""

import torch 

def poisson_exp_loss(input, target, reduction='mean'):
    assert(reduction in ['mean', 'sum', 'none'])
    l = torch.exp(input) - target * input
    if reduction == 'mean':
        return torch.mean(l)
    elif reduction == 'sum':
        return torch.sum(l)
    else:
        return l
     
