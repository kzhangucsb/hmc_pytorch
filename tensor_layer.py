"""
Created on Sat Aug 24 12:50:37 2019

@author: zkq
"""

import torch
import torch.nn as nn
import tensorly as tl
from torch.nn import Parameter, ParameterList
#from torch.nn.modules.utils import _pair
import numpy as np

tl.set_backend('pytorch')

class tensorizedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, in_rank, out_rank, alpha = 1, beta = 0.1, **kwargs):
        super(tensorizedConv2d, self).__init__()
        self.linear_pre = nn.Linear(in_channels, in_rank)
        self.conv = nn.Conv2d(in_rank, out_rank, **kwargs)
        self.linear_post = nn.Linear(out_rank, out_channels)
        self.lamb_in  = Parameter(torch.ones(in_rank))
        self.lamb_out = Parameter(torch.ones(out_rank))
        self.alpha = Parameter(torch.tensor(alpha), requires_grad=False)
        self.beta = Parameter(torch.tensor(beta), requires_grad=False)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        
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
    
    def regularizer(self, exp=True):
        ret = 0
        if exp:
            self.lamb_in.data.clamp_min_(1e-6)
            lamb_in_exp  = torch.exp(self.lamb_in)
            lamb_out_exp = torch.exp(self.lamb_out)
            ret += torch.sum(torch.sum(self.linear_pre.weight**2, dim=1) 
                * lamb_in_exp / 2)
            ret -= torch.sum(self.lamb_in) * self.in_channels / 2
            ret += torch.sum(torch.sum(self.linear_post.weight**2, dim=0) 
                * lamb_out_exp / 2)
            ret -= torch.sum(self.lamb_out) * self.out_channels / 2
            #ret += torch.sum(self.conv.weight**2 / 2)
            conv2 = self.conv.weight**2
            conv2 *= lamb_out_exp.reshape([-1, 1, 1, 1])
            conv2 *= lamb_in_exp.reshape([1, -1, 1, 1])
            ret += torch.sum(conv2) / 2
            
            ret -= torch.sum(self.alpha * self.lamb_in)
            ret -= torch.sum(self.alpha * self.lamb_out)
            ret += torch.sum(lamb_in_exp) / self.beta
            ret += torch.sum(lamb_out_exp) / self.beta
        else:
            self.lamb_in.data.clamp_min_(1e-6)
            self.lamb_out.data.clamp_min_(1e-6)
            ret += torch.sum(torch.sum(self.linear_pre.weight**2, dim=1) / self.lamb_in / 2)
            ret += torch.sum(torch.log(self.lamb_in)) * self.in_channels / 2
            ret += torch.sum(torch.sum(self.linear_post.weight**2, dim=0) / self.lamb_out / 2)
            ret += torch.sum(torch.log(self.lamb_out)) * self.out_channels / 2
            #ret += torch.sum(self.conv.weight**2 / 2)
            conv2 = self.conv.weight**2
            conv2 /= self.lamb_out.reshape([-1, 1, 1, 1])
            conv2 /= self.lamb_in.reshape([1, -1, 1, 1])
            ret += torch.sum(conv2) / 2
            
            ret += torch.sum(self.beta / self.lamb_in)
            ret += torch.sum(self.beta / self.lamb_out)
            ret += (self.alpha + 1) * torch.sum(torch.log(self.lamb_in))
            ret += (self.alpha + 1) * torch.sum(torch.log(self.lamb_out))
        return ret

        
class tensorizedlinear(nn.Module):
    def __init__(self, in_size, out_size, in_rank, out_rank,  alpha = 1, beta = 0.1, c=1e-3, **kwargs):
        super(tensorizedlinear, self).__init__()
        self.in_size = list(in_size)
        self.out_size = list(out_size)
        self.in_rank = list(in_rank)
        self.out_rank = list(out_rank)
        self.factors_in  = ParameterList([Parameter(torch.Tensor(r, s)) for (r, s) in zip(in_rank, in_size)])
        self.factors_out = ParameterList([Parameter(torch.Tensor(s, r)) for (r, s) in zip(out_rank, out_size)])
        self.core = Parameter(torch.Tensor(np.prod(out_rank), np.prod(in_rank)))
        self.bias = Parameter(torch.Tensor(np.prod(out_size)))
        self.lamb_in  = ParameterList([Parameter(torch.ones(r)) for r in in_rank])
        self.lamb_out = ParameterList([Parameter(torch.ones(r)) for r in out_rank])
        self.alpha = Parameter(torch.tensor(alpha), requires_grad=False)
        self.beta = Parameter(torch.tensor(beta), requires_grad=False)
        self.c = Parameter(torch.tensor(c), requires_grad=False)
        self._initialize_weights()
        
    def forward(self, x):
        x = x.reshape((x.shape[0], *self.in_size))
        for i in range(len(self.factors_in)):
            x = tl.tenalg.mode_dot(x, self.factors_in[i], i+1)
        x = x.reshape((x.shape[0], -1))
        x = torch.nn.functional.linear(x, self.core)
        x = x.reshape((x.shape[0], *self.out_rank))
        for i in range(len(self.factors_out)):
            x = tl.tenalg.mode_dot(x, self.factors_out[i], i+1)
        x = x.reshape((x.shape[0], -1))
        x = x + self.bias
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
        
    def regularizer(self, exp=True):
        ret = 0
        if exp:
            for l, f, s in zip(self.lamb_in, self.factors_in, self.in_size):
                ret += torch.sum(torch.sum(f**2, dim=1) * torch.exp(l) / 2)
                ret -= s * torch.sum(l) / 2
                ret -= torch.sum(self.alpha * l)
                ret += torch.sum(torch.exp(l)) / self.beta
            for l, f, s in zip(self.lamb_out, self.factors_out, self.out_size):
                ret += torch.sum(torch.sum(f**2, dim=0) * torch.exp(l) / 2)
                ret -= s * torch.sum(l) / 2
                ret -= torch.sum(self.alpha * l)
                ret += torch.sum(torch.exp(l)) / self.beta
            ret += torch.sum(self.core ** 2 / 2) 
            core_shape = list(self.out_rank) + list(self.in_rank)
            core = self.core.reshape(core_shape)
            core2 = core ** 2
            for d, l in enumerate(list(self.lamb_out) + list(self.lamb_in)):
                s = [1] * len(core_shape)
                s[d] = -1
                l = l.reshape(s)
                core2 = core2 * torch.exp(l)
                ret -= core2.numel() / l.numel() * torch.sum(l) / 2
#            core2 = self.core ** 2
            ret += torch.sum(core2) * self.c / 2
        else:
            for l, f, s in zip(self.lamb_in, self.factors_in, self.in_size):
                l.data.clamp_min_(1e-6)
                ret += torch.sum(torch.sum(f**2, dim=1) / l / 2)
                ret += s * torch.sum(torch.log(l)) / 2
                ret += torch.sum(self.beta / l)
                ret += (self.alpha + 1) * torch.sum(torch.log(l))
            for l, f, s in zip(self.lamb_out, self.factors_out, self.out_size):
                l.data.clamp_min_(1e-6)
                ret += torch.sum(torch.sum(f**2, dim=0) / l / 2)
                ret += s * torch.sum(torch.log(l)) / 2
                ret += torch.sum(self.beta / l)
                ret += (self.alpha + 1) * torch.sum(torch.log(l))
    #        ret += torch.sum(self.core ** 2 / 2) 
            core_shape = list(self.out_rank) + list(self.in_rank)
            core = self.core.reshape(core_shape)
            core2 = core ** 2
            for d, l in enumerate(list(self.lamb_out) + list(self.lamb_in)):
                s = [1] * len(core_shape)
                s[d] = -1
                l = l.reshape(s)
                core2 = core2 / l
            ret += torch.sum(core2) / 2
        return ret
    
    
    def get_lamb_ths(self, exp=True):
        if exp:
            ths_in = [np.log((
                    (s + self.core.numel() / self.in_rank[ind])/2 
                    + self.alpha.item()) * self.beta.item()) 
                for (ind, s) in enumerate(self.in_size)]
            ths_out = [np.log((
                    (s + self.core.numel() / self.out_rank[ind])/2 
                    + self.alpha.item()) * self.beta.item()) 
                for (ind, s) in enumerate(self.out_size)]
        else:
            ths_in = [self.beta.item() / (s / 2 + self.alpha.item() + 1) 
                for s in self.in_size]
            ths_out = [self.beta.item() / (s / 2 + self.alpha.item() + 1) 
                for s in self.out_size]
        return (ths_in, ths_out)
  
    
    
class TTlinear(nn.Module):
    def __init__(self, in_size, out_size, rank,  alpha = 1, beta = 0.1, **kwargs):
        # increase beta to decrease rank
        super(TTlinear, self).__init__()
        assert(len(in_size) == len(out_size))
        assert(len(rank) == len(in_size) - 1)
        self.in_size = list(in_size)
        self.out_size = list(out_size)
        self.rank = list(rank)
        self.factors = ParameterList()
        r1 =[1] + list(rank)
        r2 = list(rank) + [1]
        for ri, ro, si, so in zip(r1, r2, in_size, out_size):
            p = Parameter(torch.Tensor(ri, si, so, ro))
            self.factors.append(p)
        self.bias = Parameter(torch.Tensor(np.prod(out_size)))
        self.lamb = ParameterList([Parameter(torch.ones(r)) for r in rank])
        self.alpha = Parameter(torch.tensor(alpha), requires_grad=False)
        self.beta = Parameter(torch.tensor(beta), requires_grad=False)
        
        
        
        self._initialize_weights()
        
    def forward(self, x):
        def mode2_dot(tensor, matrix, mode):
            ms = matrix.shape
            matrix = matrix.reshape(ms[0]*ms[1], ms[2]*ms[3])
            
            sp = list(tensor.shape)
            sp[mode:mode+2] = [sp[mode]*sp[mode+1], 1]
            
            sn = list(tensor.shape)
            sn[mode:mode+2] = ms[2:4]
            
            tensor = tensor.reshape(sp)
            tensor = tl.tenalg.mode_dot(tensor, matrix.t(), mode)
            return tensor.reshape(sn)

            
            
        
        x = x.reshape((x.shape[0], 1, *self.in_size))
        for (i, f) in enumerate(self.factors):
            x = mode2_dot(x, f, i+1)
        x = x.reshape((x.shape[0], -1))
        x = x + self.bias
        return x
    
    def _initialize_weights(self):
        for f in self.factors:
            nn.init.kaiming_uniform_(f)
        nn.init.constant_(self.bias, 0)
        
    def regularizer(self, exp=True):
        ret = 0
        if exp:
            for i in range(len(self.rank)):
                # ret += torch.sum(torch.sum(self.factors[i]**2, dim=[0, 1, 2]) 
                    # * torch.exp(self.lamb[i]) / 2)
                ret -= np.prod(self.factors[i].shape[:-1]) \
                    * torch.sum(self.lamb[i]) / 2
                # ret += torch.sum(torch.sum(self.factors[i+1]**2, dim=[1, 2, 3]) 
                    # * torch.exp(self.lamb[i] / 2))
                ret -= np.prod(self.factors[i+1].shape[1:]) \
                     * torch.sum(self.lamb[i]) / 2
                ret -= torch.sum(self.alpha * self.lamb[i])
                ret += torch.sum(torch.exp(self.lamb[i])) / self.beta

            for i in range(len(self.rank)+1):
                m = torch.sum(self.factors[i]**2, dim=[1, 2])
                if i > 0:
                    m = m * torch.exp(self.lamb[i-1]).reshape([-1, 1])  
                if i < len(self.rank):
                    m = m * torch.exp(self.lamb[i]).reshape([1, -1]) 
                ret += torch.sum(m, dim=[0, 1]) / 2
               
                
                
        else:
            for i in range(len(self.rank)):
                self.lamb[i].data.clamp_min_(1e-6)
                ret += torch.sum(torch.sum(self.factors[i]**2, dim=[0, 1, 2]) 
                    / self.lamb[i] / 2)
                ret += np.prod(self.factors[i].shape[:-1]) \
                    * torch.sum(torch.log(self.lamb[i])) / 2
                ret += torch.sum(torch.sum(self.factors[i+1]**2, dim=[1, 2, 3]) 
                    /self.lamb[i] / 2)
                ret += np.prod(self.factors[i+1].shape[1:]) \
                    * torch.sum(torch.log(self.lamb[i])) / 2
                
                ret += torch.sum(self.beta / self.lamb[i])
                ret += (self.alpha + 1) * torch.sum(torch.log(self.lamb[i]))
            
        return ret
    def get_lamb_ths(self, exp=True):
        if (exp):
            lamb_ths = [
                np.log((np.prod(self.factors[i].shape[:-1]) / 2
                           + np.prod(self.factors[i+1].shape[1:]) / 2 
                           + self.alpha.item()) 
                        * self.beta.item()) for i in range(len(self.lamb))]
        else:
            lamb_ths = [self.beta.item() / (np.prod(self.factors[i].shape[:-1]) / 2
                           + np.prod(self.factors[i+1].shape[1:]) / 2 
                           + self.alpha.item() + 1) for i in range(len(self.lamb))]
        return lamb_ths
    
    
class TTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, rank, 
                 stride=1, padding=0, dilation=1,
                 alpha = 1, beta = 0.1, **kwargs):
        # increase beta to decrease rank
        super(TTConv2d, self).__init__()
        assert(len(in_channels) == len(in_channels))
        assert(len(rank) == len(in_channels) - 1)
        self.in_channels = list(in_channels)
        self.out_channels = list(out_channels)
        self.rank = list(rank)
        self.factors = ParameterList()
        
        r1 = [1] + self.rank[:-1]
        r2 = self.rank
        for ri, ro, si, so in zip(r1, r2, in_channels[:-1], out_channels[:-1]):
            p = Parameter(torch.Tensor(ri, si, so, ro))
            self.factors.append(p)
        self.bias = Parameter(torch.Tensor(np.prod(out_channels)))
        
        self.conv = nn.Conv2d(
                in_channels=self.rank[-1] * in_channels[-1],
                out_channels=out_channels[-1],
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False)
                
                        
        
        self.lamb = ParameterList([Parameter(torch.ones(r)) for r in rank])
        self.alpha = Parameter(torch.tensor(alpha), requires_grad=False)
        self.beta = Parameter(torch.tensor(beta), requires_grad=False)
        
        
        
        self._initialize_weights()
        
    def forward(self, x):
        def mode2_dot(tensor, matrix, mode):
            ms = matrix.shape
            matrix = matrix.reshape(ms[0]*ms[1], ms[2]*ms[3])
            
            sp = list(tensor.shape)
            sp[mode:mode+2] = [sp[mode]*sp[mode+1], 1]
            
            sn = list(tensor.shape)
            sn[mode:mode+2] = ms[2:4]
            
            tensor = tensor.reshape(sp)
            tensor = tl.tenalg.mode_dot(tensor, matrix.t(), mode)
            return tensor.reshape(sn)

            
            
        (b, c, h, w) = x.shape
        x = x.reshape((x.shape[0], 1, *self.in_channels, h, w))
        for (i, f) in enumerate(self.factors):
            x = mode2_dot(x, f, i+1)
        x = x.reshape((b * np.prod(self.out_channels[:-1]), 
                       self.rank[-1] * self.in_channels[-1], h, w))
        x = self.conv(x)
        x = x.reshape((b, np.prod(self.out_channels), h, w))
        x = x + self.bias.reshape((1, -1, 1, 1))
        return x
    
    def _initialize_weights(self):
        for f in self.factors:
            nn.init.kaiming_uniform_(f)
        nn.init.constant_(self.bias, 0)
        
    def regularizer(self, exp=True):
        ret = 0
        if exp:
            for i in range(len(self.rank)):
                # ret += torch.sum(torch.sum(self.factors[i]**2, dim=[0, 1, 2]) 
                #     * torch.exp(self.lamb[i]) / 2)
                m = torch.sum(self.factors[i]**2, dim=[1, 2])
                if i > 0:
                    m = m * torch.exp(self.lamb[i-1]).reshape([-1, 1]) \
                        / np.exp(self.get_lamb_ths(exp)[i-1])
                m = m * torch.exp(self.lamb[i]).reshape([1, -1]) 
                ret += torch.sum(m, dim=[0, 1]) / 2
                ret -= np.prod(self.factors[i].shape[:-1]) \
                    * torch.sum(self.lamb[i]) / 2
                if i != len(self.rank) -1:
                    # ret += torch.sum(torch.sum(self.factors[i+1]**2, dim=[1, 2, 3]) 
                    #     * torch.exp(self.lamb[i] / 2))
                    ret -= np.prod(self.factors[i+1].shape[1:]) \
                        * torch.sum(self.lamb[i]) / 2
                else:
                    w = self.conv.weight.transpose(0, 1)
                    w = w.reshape(self.rank[i], -1)
                    ret += torch.sum(torch.sum(w**2, dim=1) 
                        * torch.exp(self.lamb[i]) / 2)
                    ret -= w.shape[1] * torch.sum(self.lamb[i]) / 2
            
                
                ret -= torch.sum(self.alpha * self.lamb[i])
                ret += torch.sum(torch.exp(self.lamb[i])) / self.beta
                
                
        else:
            for i in range(len(self.rank)-1):
                self.lamb[i].data.clamp_min_(1e-6)
                ret += torch.sum(torch.sum(self.factors[i]**2, dim=[0, 1, 2]) 
                    / self.lamb[i] / 2)
                ret += np.prod(self.factors[i].shape[:-1]) \
                    * torch.sum(torch.log(self.lamb[i])) / 2
                if i != len(self.rank) -1:
                    ret += torch.sum(torch.sum(self.factors[i+1]**2, dim=[1, 2, 3]) 
                        /self.lamb[i] / 2)
                    ret += np.prod(self.factors[i+1].shape[1:]) \
                        * torch.sum(torch.log(self.lamb[i])) / 2
                else:
                    w = self.conv.weight.transpose(0, 1)
                    w = w.reshape(self.rank[i], -1)
                    ret += torch.sum(torch.sum(w**2, dim=1) /self.lamb[i] / 2)
                    ret += w.shape[1] * torch.sum(torch.log(self.lamb[i])) / 2
                    
                ret += torch.sum(self.beta / self.lamb[i])
                ret += (self.alpha + 1) * torch.sum(torch.log(self.lamb[i]))
                
        return ret
    def get_lamb_ths(self, exp=True):
        if (exp):
            lamb_ths = [
                np.log((np.prod(self.factors[i].shape[:-1]) / 2
                           + np.prod(self.factors[i+1].shape[1:]) / 2 
                           + self.alpha.item()) 
                        * self.beta.item()) for i in range(len(self.lamb)-1)]
            lamb_ths.append(
                np.log((np.prod(self.factors[-1].shape[:-1]) / 2
                           + (self.out_channels[-1]*self.in_channels[-1]
                               *self.conv.weight.shape[2]*self.conv.weight.shape[3]) / 2 
                           + self.alpha.item()) 
                        * self.beta.item()))
        else:
            lamb_ths = [self.beta.item() / (np.prod(self.factors[i].shape[:-1]) / 2
                           + np.prod(self.factors[i+1].shape[1:]) / 2 
                           + self.alpha.item() + 1) for i in range(len(self.lamb)-1)]
            lamb_ths.append(self.beta.item() / (np.prod(self.factors[-1].shape[:-1]) / 2
                           + (self.out_channels[-1]*self.in_channels[-1]
                               *self.conv.weight.shape[2]*self.conv.weight.shape[3]) / 2 
                           + self.alpha.item() + 1))
        return lamb_ths