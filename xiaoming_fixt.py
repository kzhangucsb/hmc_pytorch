"""
Created on Mon Nov 18 13:24:55 2019

@author: zkq
"""

import torch
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
from tqdm import tqdm
# from default_potential import vanilla_potential as df_poten
# from binary_potential import   Binary_potnetial as bp_poten


class hmcsampler:

    def __init__(self, potential_fcn, position_dim=None, init_position=None,  init_velocity=None):

        self.potential_fcn = potential_fcn
        self.init_velocity =None
        
        if init_velocity is not None:
            self.pos_dim = len(init_velocity)
        elif init_position is not None:
            self.pos_dim = len(init_position)
        elif position_dim is not None:
            self.pos_dim = position_dim
        else:
            raise ValueError("Neither velocity nor position and nor the dimension has been given.")


        if init_velocity is None:
            velocity = np.random.multivariate_normal(
                    np.zeros(self.pos_dim),np.eye(self.pos_dim,self.pos_dim))
        else:
            velocity = init_velocity
        
        if init_position is None :
            position = np.random.multivariate_normal(
                    np.zeros(self.pos_dim), np.eye(self.pos_dim, self.pos_dim))
        else:    
            position = init_position
        
            
        self.velocity = Variable(torch.from_numpy(velocity), requires_grad=False)
        self.position = Variable(torch.from_numpy(position), requires_grad=True)
        

        if not len(self.position) == len(self.velocity):
            raise ValueError("Lengths of init position and init velocity are not equal. fix them please.")
            '''Lenghts are give an equalt'''
            
        self._update_grad()
        return

    def reset_velocity(self):
        velocity = np.random.multivariate_normal(
                    np.zeros(self.pos_dim),np.eye(self.pos_dim,self.pos_dim))
        self.velocity.copy_(torch.from_numpy(velocity))
        
    def _pos_zero_grad(self):
        if self.position.grad is not None:
            self.position.grad.detach_()
            self.position.grad.zero_()

    def _update_grad(self):
         self._pos_zero_grad()
         potential = self.potential_fcn(self.position)
         potential.backward()

    def leap_frog_step(self, step_size, num_step = 1, skip_grad=False):
        if not skip_grad:
            self._update_grad()
#        phase_grad = position.grad
#        print(orig_hamitlonian.data)
        self.velocity -= step_size / 2 * (self.position.grad)
            
        for step in range(num_step - 1):
            self.position.data.add_(step_size * self.velocity)
            self._update_grad()
            self.velocity -= step_size * (self.position.grad)
        
        self.position.data.add_(step_size * self.velocity)
        self._update_grad()
        self.velocity -= step_size / 2 * (self.position.grad)

    
if __name__ == "__main__":
    def potential_fcn(x):
        if x < 0:
            return -x
        else:
            return 3*x
        
    num_samples = 5000
    base_step_size = 0.02
    sample_interval = 1
    sampler = hmcsampler(potential_fcn, 1)
    samples = np.zeros(num_samples)
    
    n_got = 0
    t_since_prev = 0
    pbar = tqdm(total=num_samples)
    while n_got < num_samples:
        sample_flag = False
        step_size = base_step_size / np.abs(sampler.position.grad.detach().item())
#        step_size *= 1-np.random.random()*0.1
        if step_size >= sample_interval - t_since_prev:
             step_size = sample_interval - t_since_prev
             sample_flag = True
        sampler.leap_frog_step(step_size)
        t_since_prev += step_size
        if sample_flag:
            samples[n_got] = sampler.position.item()
            n_got += 1
            t_since_prev = 0
            sampler.reset_velocity()
            pbar.update()
    pbar.close()
            
    hist_x = np.arange(-3, 3, 0.2)
    hist_samples, hist_bar = np.histogram(samples, bins=hist_x)
    prob_samples = np.array(hist_samples, float) / num_samples
    
    
    hist_y = np.zeros_like(hist_x)
    for i in range(len(hist_x)):
        hist_y[i] = np.exp(-potential_fcn(hist_x[i]))
    hist_y /= np.sum(hist_y)
    
    plt.bar(hist_bar[:-1]+0.1, prob_samples, 0.18, align='center')
    plt.plot(hist_x, hist_y, 'r--')
    plt.show()
        
        
    
    
    
    

    