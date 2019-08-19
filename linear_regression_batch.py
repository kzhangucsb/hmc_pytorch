#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 13:52:54 2019

@author: zkq
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 11:26:57 2019

@author: zkq
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, TensorDataset, DataLoader
import warnings

class linear_regression:
    def __init__(self, ndim, bias=True, lamb = 0, batchsize = 64, shuffle = True):
        self.dim = ndim
        self.bias = bias
        self.beta = None
        self.beta0 = 0
        self.lamb = lamb
        self.batchsize = batchsize
        self.shuffle = shuffle
        
    def generate_data(self, npoint, scale =1, noise_db = -np.inf):
        if self.beta is None:
            self.beta = torch.rand(self.dim)
            if self.bias:
                self.beta0 = np.random.normal() / 4
        
        datax = (torch.rand(npoint, self.dim) * 2 - 1) * scale
        datax = datax + torch.zeros_like(datax).normal_(0, np.exp(noise_db/20.))
        datay = torch.matmul(datax, self.beta) + self.beta0
        datay = torch.where(datay > 0, torch.tensor(1), torch.tensor(0))
        
        self.ds = TensorDataset(datax, datay)
        self.dl = DataLoader(self.ds, batch_size=self.batchsize, shuffle=self.shuffle)
        
    def get_loss(self, w, b=None):
        if b is None:
            if self.bias:
                b = w[-1]
                w = w[:-1]
            else:
                b = 0    
#        y_est = (torch.matmul(self.x, w) + b).sigmoid()
#        loss = F.binary_cross_entropy(y_est, self.y.float()) 
        y_est = torch.matmul(self.ds.tensors[0], w) + b
        loss = F.binary_cross_entropy_with_logits(y_est, self.ds.tensors[1].float()) 
        
        return loss
    
    def get_batch_loss(self, w, b=None):
        if b is None:
            if self.bias:
                b = w[-1]
                w = w[:-1]
            else:
                b = 0    
#        y_est = (torch.matmul(self.x, w) + b).sigmoid()
#        loss = F.binary_cross_entropy(y_est, self.y.float()) 
        x, y = next(iter(self.dl))
        y_est = torch.matmul(x, w) + b
        loss = F.binary_cross_entropy_with_logits(y_est, y.float()) 
        
        return loss
    
    def get_regularized_loss(self, w, b=None):
        if b is None:
            if self.bias:
                b = w[-1]
                w = w[:-1]
            else:
                b = 0    
        return self.get_loss(w, b) + self.lamb * torch.sum(w*w)
    
    def get_batch_regularized_loss(self, w, b=None):
        if b is None:
            if self.bias:
                b = w[-1]
                w = w[:-1]
            else:
                b = 0    
        return self.get_batch_loss(w, b) + self.lamb * torch.sum(w*w)
    
    def get_error(self, w, b=None):
        if b is None:
            if self.bias:
                b = w[-1]
                w = w[:-1]
            else:
                b = 0    
        y_est = torch.where(torch.matmul(self.ds.tensors[0], w) + b > 0, torch.tensor(1), torch.tensor(0))
        nerr = torch.sum(y_est != self.ds.tensors[1])
        return float(nerr) / float(len(self.ds.tensors[0]))
    
    def get_ground_truth(self, tor = 1e-8, max_iter = 1000, lr= 1):

#        w = Variable(torch.cat((self.beta, torch.tensor([self.beta0]))), requires_grad=True)
        w = Variable(torch.Tensor(np.random.normal(0, 1, self.dim + (1 if self.bias else 0))), requires_grad=True)
#        optimizer = torch.optim.SGD([w], lr=0.1) 
        l_old = np.inf
        stop_reached = False
        for niter in range(max_iter):
#            lr_n = lr
#            optimizer.zero_grad() 
            l = self.get_regularized_loss(w)
            l.backward()
            w.data -= w.grad * lr #+ torch.Tensor(np.random.normal(0, 1e-6, self.dim+1))
            
#            optimizer.step() 
#            print(l_old, l_old-l.item(), torch.norm(w.grad) , flush=True)
            if (l.item() > l_old):
                lr /= 2
                warnings.warn('Loss is increasing. Reducing stepsize to {}'.format(lr))
            if (l_old - l.item() < tor):
#                w.detach_()
#                return w[:-1], w[-1].item()
                stop_reached = True
                break
            l_old = l.item()
            w.grad.zero_()
        if not stop_reached:  
            warnings.warn('Maxiter reached. Result may be inaccurate')
        w.detach_()
        if self.bias:
            return w[:-1], w[-1].item() 
        else:
            return w, 0
        
#     a = self.get_ground_truth()
        
        