"""
Created on Sat Dec  7 14:16:37 2019

@author: zkq
"""

import pickle
import gzip
import numpy as np
dim = 3
training_precent = 0.8

indexes = [[] for _ in range(dim)]
values = []

# year = set()

with gzip.open('../data/nips/nips.tns.gz', 'rt') as f:
    while(1):
        line = f.readline()
        if len(line) == 0:
            break ## EOF
        line = line.rstrip().split(' ')
        assert(len(line) == dim + 2)
        if int(line[dim]) != 17:
            continue
        
        for ii in range(dim):
            indexes[ii].append(int(line[ii]))
        values.append(int(line[-1]))
        
    
    values = np.array(values)
    indexes = [np.array(ind) for ind in indexes]
    size = [0] * dim
    for ii in range(dim):
        indexes_set = sorted(set(indexes[ii])) 
        size[ii] = len(indexes_set)
        indexed_map = {key: num for (num, key) in enumerate(indexes_set)} 
        indexes[ii] = np.array([indexed_map[ind] for ind in indexes[ii]])
        
    
    nsamples  = len(values)
    ntrain    = int(np.round(nsamples * training_precent))
    
    ind_all   = np.random.permutation(np.arange(nsamples, dtype=int))
    ind_train = ind_all[:ntrain]
    ind_test  = ind_all[ntrain:]
    
    train_set = {'indexes': tuple([index[ind_train] for index in indexes]), 
                 'values': values[ind_train]}
    test_set = {'indexes': tuple([index[ind_test] for index in indexes]), 
                 'values': values[ind_test]}
    
    with open('../data/frostt_nips_17.pickle', 'wb') as f:
        pickle.dump({'train': train_set, 'test': test_set, 
                     'size': tuple(size)}, f)
    
    
        
    
            