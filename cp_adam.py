
"""
Created on Sun Jul 28 17:20:21 2019

@author: zkq
"""

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
from torch.utils.data import TensorDataset, DataLoader
#from cp import cp
from bfcp import bfcp as cp, regulized_loss
import tensorly as tl
device = torch.device('cpu')
args = dict()#{'num_workers': 1, 'pin_memory': True}
batchsize = 40
batchsize_test = 5000
nepoch = 5000

nl = 'invivo2'
svd_init = False
fix_tau = False

with open('../data/ktensor_noise_{}.pickle'.format(nl), 'rb') as f:
#with open('../data/invivo.pickle', 'rb') as f:
    p = pickle.load(f)
    
size = p['size']
rank = 80
train_input = torch.LongTensor(p['train']['indexes']).t()
train_value = torch.Tensor(p['train']['values'])
#train_norm = torch.norm(train_value).item()
test_input = torch.LongTensor(p['test']['indexes']).t()
test_value = torch.Tensor(p['test']['values'])
#test_norm = torch.norm(test_value).item()

model = cp(size, rank, beta=0.9, d=1e4)#1, beta = 1, c = 1)#5, 1e4) # beta=0.2, c=10)


if svd_init:
    t = torch.zeros(size)
    for ind, value in zip(train_input, train_value):
        t[tuple(ind)] = value
    for r in range(len(size)):
        t1 = tl.unfold(t, r)
        u, s, v = torch.svd(t1)
        model.factors[r].data = u[:,:rank] * (s[:rank] ** 0.5)
    print("factors initialized")
        
if fix_tau:
    model.tau.data = torch.tensor(2.6)
    model.tau.requires_grad = False
    
model.to(device)    



train_loader = DataLoader(TensorDataset(train_input, train_value),
        batch_size = batchsize, shuffle=True, **args)
test_loader = DataLoader(TensorDataset(test_input, test_value),
        batch_size = batchsize_test, shuffle=False, **args)
#for ii in range(3):
#    model.factors[ii].data.copy_(torch.tensor(p['factors'][ii]))

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=2e-3)
#optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.5)
schedular = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: (epoch+1)**(-0.5))
loss_log = [np.zeros(nepoch+1), np.zeros(nepoch+1)]

with torch.no_grad():
    loss_train = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        out = model(data)
        loss_train += criterion(out, target).item() ** 0.5 
    loss_train /= len(train_loader) 
    loss_test = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        out = model(data)
        loss_test += criterion(out, target).item() ** 0.5 
    loss_test /= len(test_loader) 
    loss_log[0][0] = loss_train
    loss_log[1][0] = loss_test

for epoch in range(nepoch):
    loss_train = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, target)
        
        loss_train += loss.item() 
#        loss *= (len(train_loader.dataset) * torch.exp(model.tau)/2)
#        loss -= 0.5 * len(train_loader.dataset) * model.tau
#        
#        loss += model.prior_theta() #/ 1e2  #/ len(data)
#        loss += model.prior_tau_exp() #/ 1e2 # 1e-2
        loss = regulized_loss(loss, model, len(train_loader.dataset))
        
        loss.backward()
        optimizer.step()
        
    loss_train /= len(train_loader)
    loss_train = loss_train ** 0.5
    if (epoch+1) % 500 == 0:
        schedular.step()
    with torch.no_grad():
        loss_test = 0
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            out = model(data)
            loss_test += criterion(out, target).item() 
        loss_test /= len(test_loader) 
        loss_test = loss_test ** 0.5
    loss_log[0][epoch+1] = loss_train
    loss_log[1][epoch+1] = loss_test
    print('Epoch {} training err {:4.4f} testing err {:4.4f} training loss {}'.format(
            epoch, loss_train, loss_test, loss.item()))
    
plt.semilogy(np.arange(nepoch+1), loss_log[0], label='train')
plt.semilogy(np.arange(nepoch+1), loss_log[1], label='test')
plt.legend()
plt.show()
#with open('../models/cp_bayes_model_{}.pth'.format(nl), 'wb') as f:
#    torch.save(model.state_dict(), f)

#with open('cp_bayes_model_invivo.pth'.format(nl), 'wb') as f:
