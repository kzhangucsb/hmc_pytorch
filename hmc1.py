#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 15:33:04 2019

@author: zkq
"""

from HMC_sampler import sampler
import torch
from   torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from linear_regression import linear_regression


#Regular run
#hmc = sampler(sample_size=100,position_dim=5)
#sample,_= hmc.main_hmc_loop()

#print("Init place")
#hmc = sampler(sample_size=100,init_position=np.array([1.0,2.1,3.1,-1.]))
#sample,_= hmc.main_hmc_loop()
#print(sample)

class test_potential:

    def __init__(self, bias,  weight_matrix):
        self.weight_matrix = weight_matrix
        self.bias = bias
        
    def __call__(self, xx):
        return self.calc_potential_energy(xx)

    def calc_potential_energy (self, xx):
        xx = xx - self.bias
        potential_energy=torch.dot(xx,torch.matmul(self.weight_matrix,xx))
        return potential_energy
#Regular run
#print("Potential")

#
Bmat = [torch.FloatTensor([2,2]), torch.FloatTensor([-2,-2]), torch.FloatTensor([1,-3])]
init_pos = [[0.1,0.1], [-0.1,-0.1],[0.1,-0.1]]
Amat = [torch.FloatTensor([[-2, -1], [-1, -2]]) , torch.FloatTensor([[-2, 0.5], [0.5, -2]]), torch.FloatTensor([[-1, 1], [1, -2]])]
#                          [0, 0, -2, 0], 
#                          [0, 0, 0, -2]])
    
#c = linear_regression(3, lamb=0.1)
#c.generate_data(200, scale=1, noise_db=-40)
#ps = lambda x: -200*c.get_loss(x[:-1], x[-1])    
sample = [None]*3

tp = [test_potential(Bmat[i], Amat[i]) for i in range(3)]

for i in range(3):
    hmc = sampler(sample_size=50, position_dim=2, step_size=0.02,init_position=init_pos[i], potential_struct=tp[i])
    sample[i],_ = hmc.main_hmc_loop()
#print(sample)


plt.plot(sample[0][:, 0], sample[0][:, 1], 'b', label='chain 1')
plt.plot(sample[1][:, 0], sample[1][:, 1], 'r', label='chain 2')
plt.plot(sample[2][:, 0], sample[2][:, 1], 'g', label='chain 3')

xaxis = np.arange(-4, 4, 1e-1)
yaxis = np.arange(-4, 4, 1e-1)

x, y = np.meshgrid(xaxis, yaxis)
p = np.zeros_like(x)

 


for ii in range(len(x)):
    for jj in range(len(x[ii])): 
        tmp = [np.exp(tp[i](torch.tensor([x[ii, jj], y[ii, jj]])).item()) for i in range(3)]
        p[ii, jj] = sum(tmp)

plt.contour(x, y, p,levels= np.exp(np.arange(-7, 0, 2)))
plt.xticks([])
plt.yticks([])
#z = np.exp(-2 * dist_x**2)
#plt.plot(dist_x, len(sample)*z/sum(z))

#print("Init vel")
#hmc = sampler(sample_size=100,position_dim=4,init_velocity=np.array([1.0,2.1,3.1,-1.]))
#sample,_= hmc.main_hmc_loop()
#print(sample)

