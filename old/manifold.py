#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 16:41:58 2019

@author: zkq
"""

import torch
from HMC_sampler import sampler
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plane2z(x, y):
    return (-x**2-y**2)/20+(x*y)/10+(y-x)/2

x, y = np.meshgrid(np.arange(-4, 4, 0.1), np.arange(-4, 4, 0.2))
z = plane2z(x, y)

fig = plt.figure()
#ax = plt.axes(projection='3d')
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, color=[0, 1, 1], alpha=0.5)

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

Bmat = torch.FloatTensor([1,-1])
Amat = torch.FloatTensor([[-0.5, 0], [0, -0.3]]) 

hmc = sampler(sample_size=100, position_dim=2, step_size=0.02, potential_struct=test_potential(Bmat, Amat))
trace,_ = hmc.main_hmc_loop()
tracez = plane2z(trace[:, 0], trace[:, 1])
ax.plot(trace[:, 0], trace[:, 1], tracez, 'r')
ax.plot(trace[:, 0], trace[:, 1], -8)
ax.axis([-4, 4, -4, 4])
ax.set_zlim(-8, 0)
