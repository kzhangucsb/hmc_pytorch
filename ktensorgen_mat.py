"""
Created on Tue Aug 27 10:52:49 2019

@author: zkq
"""



import numpy as np
import tensorly as tl
import pickle
import scipy.io as io



d = io.loadmat('../data/aperiodic_pincat2.mat')

full_nn = abs(d['f']) 
factors = None
rank = None
full = full_nn / 167.6


size = full.shape
filename = '../data/ktensor_noise_pincat.pickle'

samples_train = 80000
samples_test = 1000



subs = np.random.choice(np.prod(size), samples_train+samples_test, False)
ind_train = np.unravel_index(subs[:samples_train], size)
ind_test = np.unravel_index(subs[samples_train:], size)

values_train = full[ind_train]
#values_train += np.random.randn(*values_train.shape)*1e-1
values_test  = full[ind_test]

with open(filename, 'wb') as f:
    pickle.dump({'size': size, 'rank': rank,
                 'factors': factors, 'full': full, 'full_nn': full_nn,
                 'train': {'indexes': ind_train, 'values': values_train}, 
                 'test': {'indexes': ind_test, 'values': values_test}}, f)
