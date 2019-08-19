#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 12:10:21 2019

@author: zkq
"""

from __future__ import print_function
import torch
import math
import torch.nn.functional as F
from torch.nn import init
from torch.nn import Module, Parameter, ParameterList

import tensorly
tensorly.set_backend('pytorch')

class cp(Module):
    def __init__(self, size, rank, update_lambda=False):
        super(cp, self).__init__()
        self.size = size
        self.rank = rank
        self.dim = len(size)
        if update_lambda:
            self.lamb = Parameter(torch.ones(self.rank))
        else:
            self.lamb = torch.ones(self.rank)
            
        self.factors = ParameterList([Parameter(torch.Tensor(s, rank)) for s in size])
        self.reset_parameters()

    def reset_parameters(self):
        for f in self.factors:
#            init.kaiming_uniform_(f, a=math.sqrt(5))
            init.uniform_(f)
#            f.data.copy_(torch.rand_like(f.data))


#    @weak_script_method
    def forward(self, input):
        #input: indexes, batch*dim, torch.long
        vsz = input.shape[0]
        vals = torch.zeros(vsz)
        for j in range(self.rank):
            tmpvals = self.lamb[j] * torch.ones(vsz)
            for k in range(self.dim):
                akvals = self.factors[k][input[:,k],j]
                tmpvals = tmpvals * akvals

            vals = vals + tmpvals;
        return vals

    def extra_repr(self):
        return 'size={}, rank={}'.format(
            self.size, self.rank
        )
        
    def full(self):
        return tensorly.kruskal_to_tensor(self.factors, self.lamb)




