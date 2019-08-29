
"""
Created on Sun Jul 28 15:55:28 2019

@author: zkq
"""

import numpy as np
import tensorly as tl
import pickle
import scipy.io as io



#d = io.loadmat('../data/invivo_perfusion4.mat')
#
#full = abs(d['f'])
#factors = None



#size = full.shape
nl = '1em2'

nl_num = float(nl[0]) * 10 ** (-float(nl[3]))


size = (20, 20, 20)

rank = 5
samples_train = 1600
samples_test = 800



factors = [np.random.randn(s, rank) for s in size]
full_nn = tl.kruskal_to_tensor((np.ones(rank), factors) )
full    = full_nn + np.random.randn(*size) * nl_num

subs = np.random.choice(np.prod(size), samples_train+samples_test, False)
ind_train = np.unravel_index(subs[:samples_train], size)
ind_test = np.unravel_index(subs[samples_train:], size)

values_train = full[ind_train]
#values_train += np.random.randn(*values_train.shape)*1e-1
values_test  = full[ind_test]

p = {'size': size, 'rank': rank,
                 'factors': factors, 'full': full, 'full_nn': full_nn,
                 'train': {'indexes': ind_train, 'values': values_train}, 
                 'test': {'indexes': ind_test, 'values': values_test}}

with open('../data/ktensor_noise_normal_{}.pickle'.format(nl), 'wb') as f:
    pickle.dump(p, f)
    
io.savemat('../data/ktensor_noise_normal_{}.mat'.format(nl), p)
