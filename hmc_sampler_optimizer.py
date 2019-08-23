
"""
Created on Thu Aug  8 18:14:55 2019

@author: zkq
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 11:28:02 2019

@author: zkq
"""


import torch
from torch.optim.optimizer import Optimizer
import numpy as np
import warnings
from copy import deepcopy
import os
import threading
# from default_potential import vanilla_potential as df_poten
# from binary_potential import   Binary_potnetial as bp_poten


class hmcsampler(Optimizer):
    
    def __init__(self, params, num_steps_in_leap = 32, max_step_size=0.05, samples_dir=None, **kwargs):
        defaults = {'B': 0, 
                  'frac_adj': 0, 
                  'max_length': 1, 
                  'mass': 1,
                  'sample': None}
        for key, value in kwargs.items():
            if key in defaults.keys():
                defaults[key] = value
            else:
                raise TypeError('Unexpected key {}'.format(key))
        super(hmcsampler, self).__init__(params, defaults)
        
        
        self.max_step_size = max_step_size
        self.num_steps_in_leap = num_steps_in_leap
        self.samples_dir = samples_dir
        if self.samples_dir is not None and not os.path.exists(self.samples_dir):
            os.makedirs(self.samples_dir)
        self.samples =[]
        for group in self.param_groups:
            group['velocity'] = []
            for p in group['params']:
                group['velocity'].append(p.clone().detach())
        self._reset_v()
        self.step_eclipsed = 0
        self._step_size = 0
        
        self.save_thread = None
        
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        if (self.step_eclipsed > 0) and (self.step_eclipsed % self.num_steps_in_leap == 0):
            # keep a sample
            self._step_v(self._step_size / 2)
            self._sample()
            # reset potential
            self._reset_v()
            # next step
            step_size = self._get_step_size()
            self._step_v(step_size / 2)
            
        else:
            # combine two steps
            step_size = self._get_step_size()
            self._step_v((self._step_size +  step_size)/ 2)
        self._step_size = step_size
        self._update_B(self._step_size)
        self._step_t(self._step_size)
                
        self.step_eclipsed += 1
        return loss
 
        
    def _sample(self):
        for group in self.param_groups:
            if group['sample'] is not None and not group['sample'](group['params']):
                return
            
        s = []
        for group in self.param_groups:
            s.append([p.clone().detach().cpu() for p in group['params']])
        if self.samples_dir is None:
            self.samples.append(s)
        else:
            fname = 'sample_{}.pth'.format(len(self.samples))
            fname_full = os.path.join(self.samples_dir, fname)
#            torch.save(s, fname_full)
            self.samples.append(fname_full)
            if self.save_thread is not None:
                self.save_thread.join()
            self.save_thread = threading.Thread(target=torch.save, args=(s, fname_full))
            self.save_thread.start()
            
                
            


    def _reset_v(self):
        for group in self.param_groups:
            m = group['mass']
            for p in group['velocity']:
                p.normal_(0, m ** 0.5)

    def _get_step_size(self):
        step_size = self.max_step_size
        for group in self.param_groups:
            norm_v = 0
            norm_g = 0
            for (v, p) in zip(group['velocity'], group['params']):
                norm_v += torch.norm(v)**2
                if p.grad is not None:
                    norm_g += torch.norm(p.grad)**2
            norm_v = norm_v.item() ** 0.5
            norm_g = norm_g.item() ** 0.5
            s = np.inf
            if norm_v > 0:
                s = min(s, group['max_length'] / norm_v * group['mass'])
            if norm_g > 0:
                s = min(s, (group['max_length'] / norm_g * group['mass']) ** 0.5)
            step_size = min(step_size, s)
        return step_size
               
                    
    
    def _update_B(self, step_size): 
        for group in self.param_groups:
            norm_v = 0
            n_params = 0
            for v in group['velocity']:
                norm_v += torch.norm(v)**2
                n_params += v.numel()
            group['B'] += step_size * (norm_v / n_params / group['mass'] - 1) * group['frac_adj']
            group['B'] = max(group['B'] , 0)
            group['B'] = min(group['B'] , 1 / step_size)
   
        
    def _step_v(self, step_size):
        for group in self.param_groups:
            for [v, p] in zip(group['velocity'], group['params']):
                v.add_(-v * (step_size * group['B']))
                if p.grad is not None:
                    v.add_(-p.grad * step_size)  
                
    def _step_t(self, step_size):
        for group in self.param_groups:
            for [v, p] in zip(group['velocity'], group['params']):
                p.data.add_(v * (step_size / group['mass']))
        
    def load_sample_index(self, index):
        self.load_sample(self.samples[index])
    
    def load_sample(self, sample):
        if isinstance(sample, str):
            sample = torch.load(sample)
            
        for group_src, group_dst in zip(sample, self.param_groups):
            for p_src, p_dst in zip(group_src, group_dst['params']):
                p_dst.data.copy_(p_src.to(p_dst.device))
        
        
   
        
            

    
