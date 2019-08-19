
"""
Created on Sun Jul 28 15:55:28 2019

@author: zkq
"""

import numpy as np
import tensorly as tl
import pickle
import scipy.io as io

#filename = 'invivo.pickle'
#
#d = io.loadmat('invivo_perfusion4.mat')
#
#full = abs(d['f'])
#factors = None



#size = full.shape
filename = 'ktensor_noise_1em1.pickle'
size = (20, 20, 20)

rank = 5
samples_train = 800#100000
samples_test = 800#100000



factors = [np.random.rand(s, rank) for s in size]
full    = tl.kruskal_to_tensor(factors)

subs = np.random.choice(np.prod(size), samples_train+samples_test, False)
ind_train = np.unravel_index(subs[:samples_train], size)
ind_test = np.unravel_index(subs[samples_train:], size)

values_train = full[ind_train]
values_train += np.random.randn(*values_train.shape)*1e-1
values_test  = full[ind_test]

with open(filename, 'wb') as f:
    pickle.dump({'size': size, 'rank': rank,
                 'factors': factors, 'full': full, 
                 'train': {'indexes': ind_train, 'values': values_train}, 
                 'test': {'indexes': ind_test, 'values': values_test}}, f)
