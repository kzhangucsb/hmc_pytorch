#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 15:52:59 2019

@author: zkq
"""

from __future__ import print_function
import torch
import math
import torch.nn.functional as F
from torch.nn import init
from torch.nn import Module, Parameter, ParameterList
import numpy as np

import tensorly
tensorly.set_backend('pytorch')

class bfcp(Module):
    def __init__(self, size, rank, alpha = 1, beta = 0.2, c = 1, d = 1e6):
        super(bfcp, self).__init__()
        self.size = size
        self.rank = rank
        self.dim = len(size)
        self.lamb = Parameter(torch.Tensor(self.rank))
        self.tau  = Parameter(torch.tensor(1.0))
        self.alpha = Parameter(torch.tensor(alpha), requires_grad=False)
        self.beta = Parameter(torch.tensor(beta), requires_grad=False)
        self.c = Parameter(torch.tensor(c), requires_grad=False)
        self.d = Parameter(torch.tensor(d), requires_grad=False)
#        self.lamb = torch.Tensor(self.rank)
        
        self.factors = ParameterList([Parameter(torch.Tensor(s, rank)) for s in size])
        self.reset_parameters()

    def reset_parameters(self):
#        init.uniform_(self.lamb)
        init.constant_(self.lamb, 1)
        for f in self.factors:
            init.uniform_(f)


#    @weak_script_method
    def forward(self, input):
        #input: indexes, batch*dim, torch.long
        vsz = input.shape[0]
        vals = torch.zeros(vsz, device=input.device)
        for j in range(self.rank):
#            tmpvals = self.lamb[j] * torch.ones(vsz)
            tmpvals = torch.ones(vsz, device=input.device)
            for k in range(self.dim):
                akvals = self.factors[k][input[:,k],j]
                tmpvals = tmpvals * akvals

            vals = vals + tmpvals
        return vals
    
    def prior_theta(self):
        self.lamb.data.clamp_min_(1e-6)
        
               
        ret = 0
        for f in self.factors:
            ret += torch.sum(torch.sum(f**2, dim=0) / self.lamb / 2)
        ret += torch.sum(torch.log(self.lamb))*sum(self.size)/ 2
        
#        for f in self.factors:
#            ret += torch.sum(torch.sum(f**2, dim=0) * self.lamb)
#        ret -= torch.sum(torch.log(self.lamb))*sum(self.size)/ 2
         
        ret += torch.sum(self.beta / self.lamb)
        ret += (self.alpha + 1) * torch.sum(torch.log(self.lamb))
    

        return ret 
    def prior_tau_exp(self):
        return -self.c * self.tau + torch.exp(self.tau) / self.d
        
#    def prior_theta_exp(self):        
#               
#        ret = 0
#        for f in self.factors:
#            ret += torch.sum(torch.sum(f**2, dim=0) * torch.exp(self.lamb))
#        ret += torch.sum(self.lamb)*sum(self.size)/ 2
#        
#
#        ret += -self.alpha * torch.sum(self.lamb)
#        ret += torch.sum(torch.exp(self.lamb)) / self.beta
#      
#        return ret 
#    
#    def prior_tau(self):
#        self.tau.data.clamp_min_(1e-6)
#        
#        return (self.c+1) * torch.log(self.tau) + self.d / self.tau
    
    
        
    def extra_repr(self):
        return 'size={}, rank={}, alpha={}, beta={}, c={}, d={}'.format(
            self.size, self.rank, self.alpha.item(), self.beta.item(), 
            self.c.item(), self.d.item()
        )
        
    def full(self):
        return tensorly.kruskal_to_tensor(
                (torch.ones(self.rank, device=self.lamb.device), self.factors))


def regulized_loss(loss, model, len_dataset):
    loss_n = loss * (len_dataset * torch.exp(model.tau)/2)
    loss_n -= 0.5 * len_dataset * model.tau
        
    loss_n += model.prior_theta() #/ 1e2  #/ len(data)
    loss_n += model.prior_tau_exp() #/ 1e2 # 1e-2
    return loss_n
        

