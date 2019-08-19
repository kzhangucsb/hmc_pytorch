#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 11:28:02 2019

@author: zkq
"""


import torch
import numpy as np
#from  Hamiltonian import hamilton_operator as hamilton
from torch.autograd import Variable
from torch.nn.functional import conv1d, interpolate
#import sys
import warnings
import copy
# from default_potential import vanilla_potential as df_poten
# from binary_potential import   Binary_potnetial as bp_poten


class hmcsampler:

    def __init__(self, model, dataloader, criterion, **kwargs):
        options = {'dataloader_test': None, 
                  'dataloader_index': False, 
                  'g':lambda x: torch.tensor(1), 
                  'phi': lambda x: torch.tensor(0), 
                  'do_vr': False, 
                  'T': 1, 'B': 0, 'm': 1, 'Bt': 0, 
                  'frac_adj': 0, 'lb': 0, 
                  'max_step_size':0.05, 
                  'max_length': 1, 
                  'max_length_t': 1e-3, 
                  'num_steps_in_leap': 64,
                  'mass': {},
                  'regularizer': None}
        for key, value in kwargs.items():
            if key in options.keys():
                options[key] = value
            else:
                raise TypeError('Unexpected key {}'.format(key))
        for k, v in options.items():
            setattr(self, k, v)    
        self.model = model   
        self.device = next(self.model.parameters()).device
        self.dataloader = dataloader  
        self.criterion = criterion
        
        self.position = dict(self.model.named_parameters())
        self.n_params = 0
        for key, value in self.position.items():
            self.n_params += value.numel()
        self.t = torch.tensor(0.0).to(self.device)
        print(self.device)



    def sample(self, sample_size):
        # H = g(t)U(position) + 0.5*velocity**2 + phi(t) + 0.5*v**2
        steps = 0
        sample_array = []#[copy.deepcopy(self.model.state_dict())]
        t_array = []
        
        velocity = dict()
        for key, para in self.position.items():
            velocity[key] = torch.zeros_like(para)
            
        
        v = torch.tensor(0.0).to(self.device)
    
        while len(sample_array) < sample_size:
#            velocity = [torch.randn_like(p) * self.T for p in self.model.parameters()]
            for key, item in velocity.items():
                if key in self.mass.keys():
                    m = self.mass[key]
                else:
                    m = 1    
                item.normal_(0, m ** 0.5)
            v.normal_(0, self.m ** 0.5)
            
            
#            p_old = copy.deepcopy(self.model.state_dict())
#            t_old = copy.deepcopy(t)
            self._leap_frog_step(velocity, v)

            
            if self.dataloader_test is not None:
                with torch.no_grad():
                    loss = self.get_batch_loss(self.dataloader_test)
            else:
                loss = None

            if self.g(self.t).item() == 1:
                sample_array.append(copy.deepcopy(self.model.state_dict()))
            steps += 1
            t_array.append(self.t.item())
            
            print('\rGenerated {}/{} samples, total steps {}, loss {:04f}, t {:04f}'.format(
                    len(sample_array), sample_size, steps, loss, self.t.item()), end='')
        print('')
        

        return sample_array, {'steps': steps, 't_array': t_array}
        

    def _leap_frog_step(self, velocity, v):
        t = Variable(self.t, requires_grad=True)
        
        position = self.position
        loss = self.get_batch_loss(self.dataloader)
        H = (loss - self.lb) * self.g(t) + self.phi(t)
        H /= self.T
        if self.regularizer is not None:
            H += self.regularizer(self.model)
        H.backward()

        for step in range(self.num_steps_in_leap):
            with torch.no_grad():
                velocity_norm = 0
                grad_norm = 0
                for key in position:
                    if key in self.mass.keys():
                        m = self.mass[key]
                    else:
                        m = 1     
                    velocity_norm += torch.norm(velocity[key]).item()**2 / m
                    grad_norm += torch.norm(position[key].grad).item()**2
                velocity_norm = velocity_norm ** 0.5
                grad_norm = grad_norm ** 0.5
#            
            stepsize_max = [self.max_step_size, self.max_length / velocity_norm, 
                            (self.max_length / grad_norm)**0.5, 
                            self.max_length_t * self.m/ (abs(v.item())+1e-6) if t.grad is not None else np.inf,
                            (self.max_length_t * self.m/ (abs(t.grad.item())+1e-6)) **0.5 if t.grad is not None else np.inf]
            step_size = min(stepsize_max)
#            print(step_size, t.item(), t.grad.item(), v.item())
#            print(step_size, t)
            # friction
            self.B += step_size * (velocity_norm ** 2 / self.n_params - 1) * self.frac_adj
            self.Bt += step_size * (v.item() ** 2 / self.m - 1) * self.frac_adj
            self.B = max(self.B, 0)
            self.Bt = max(self.Bt, 0)
            self.B = min(self.B, 1/step_size)
            self.Bt = min(self.Bt, 1/step_size)
            # leapfrog
            for key in position:
                velocity[key] -= step_size / 2 * (position[key].grad + self.B * velocity[key]) 
            if t.grad is not None:
                v -= step_size / 2 * (t.grad + self.Bt * v)
            else:
                v -= step_size / 2 * (self.Bt * v)
                
                
#            print(t.item(), t.grad.item() , v.item())
            
            for key in position:
                if key in self.mass.keys():
                    m = self.mass[key]
                else:
                    m = 1    
                position[key].data.add_(step_size * velocity[key] / m)
                if position[key].grad is not None:
                    position[key].grad.detach_()
                    position[key].grad.zero_()
            t.data.add_(step_size * v / self.m)
            assert(abs(t)<20)
            if t.grad is not None:
                t.grad.detach_()
                t.grad.zero_()
            
            
            
            loss = self.get_batch_loss(self.dataloader)
            H = (loss - self.lb) * self.g(t) + self.phi(t)
            H /= self.T
            if self.regularizer is not None:
                H += self.regularizer(self.model)
            H.backward()
            
            
            for key in position:
                velocity[key] -= step_size / 2 * (position[key].grad + self.B * velocity[key]) 
            if t.grad is not None:
                v -= step_size / 2 * (t.grad + self.Bt * v)
            else:
                v -= step_size / 2 * (self.Bt * v)
                

        
    def get_batch_loss(self, dataset):
        if self.dataloader_index:
            data, labels, index = next(iter(dataset))
            data, labels, index = data.to(self.device), labels.to(self.device), index.to(self.device)
        else:
            data, labels = next(iter(dataset))
            data, labels = data.to(self.device), labels.to(self.device)
        output = self.model(data)
        loss = self.criterion(output, labels)
#        if self.regularizer is not None:
#            loss += self.regularizer(self.model)
        return loss
        


