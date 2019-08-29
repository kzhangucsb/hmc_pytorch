
"""
Created on Sun Aug 11 12:11:10 2019

@author: zkq
"""


from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from hmc_sampler_optimizer import hmcsampler
import pickle
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
#from cp import cp
from bfcp import bfcp as cp, regulized_loss

from tqdm import tqdm

#nl = '1em2'
nl = '1em3'

device = torch.device('cpu')
args = {'num_workers': 1, 'pin_memory': True}
with open('../data/ktensor_noise_{}.pickle'.format(nl), 'rb') as f:
    p = pickle.load(f)

#with open('../data/invivo.pickle', 'rb') as f:
#    p = pickle.load(f)
#    
    
    
size = p['size']

train_input = torch.LongTensor(p['train']['indexes']).t()
train_value = torch.Tensor(p['train']['values'])
#train_norm = torch.norm(train_value).item()
test_input = torch.LongTensor(p['test']['indexes']).t()
test_value = torch.Tensor(p['test']['values'])
#test_norm = torch.norm(test_value).item()

T = 1#e-2
batchsize = 40


train_loader = DataLoader(TensorDataset(train_input, train_value),
        batch_size = batchsize, shuffle=True, **args)
test_loader = DataLoader(TensorDataset(test_input, test_value),
        batch_size = batchsize, shuffle=False, **args)

criterion = nn.MSELoss()

with open('../models/cp_bayes_model_{}.pth'.format(nl), 'rb') as f:
    state_dict = torch.load(f)
    
ths = 0.2
ind = np.where(state_dict['lamb'].cpu().numpy() > ths)[0]
rank = len(ind)
state_dict['lamb'] = state_dict['lamb'][ind]
for i in range(3):
    state_dict['factors.{}'.format(i)] = state_dict['factors.{}'.format(i)][:, ind] 
    
print('rank={}'.format(rank))
model = cp(size, rank)
#model.to(device)
model.load_state_dict(state_dict)
    

    
sampler = hmcsampler([{'params':model.factors, 'max_length': 0.001}, # 'max_length': 1e-2
                    {'params':model.lamb, 'mass': 1e2, 'max_length': 0.01},#'mass': 1e2, 'max_length': 1e-2
                    {'params':model.tau, 'mass': 1}], #'mass': 1e2
    frac_adj= 1, max_step_size=0.01)

while (len(sampler.samples) <500):
    loss_train = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        sampler.zero_grad()
        out = model(data)
        loss = criterion(out, target)
        loss_train += loss.item()
#        loss *= len(train_loader.dataset)
##        loss *= len(data)
#        loss *= torch.exp(model.tau)
#        loss -= 0.5 * len(train_loader.dataset) * model.tau
        loss = regulized_loss(loss, model, len(train_loader.dataset))
        
        loss /= T

        loss += model.prior_theta() 
        loss += model.prior_tau_exp() #*100
        loss.backward()
        sampler.step()
        

        
    loss_train /= len(train_loader)
    loss_train = loss_train ** 0.5
   
    with torch.no_grad():
        loss_test = 0
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            out = model(data)
            loss_test += criterion(out, target).item()
        loss_test /= len(test_loader) 
        loss_test = loss_test ** 0.5
    print('Samples {} training loss {} testing loss {}'.format(len(sampler.samples), loss_train, loss_test))
    
    

#with torch.no_grad():
#    
#    bias = np.array([])
#    bias_avg = []
#    pbar = tqdm(total=len(test_loader))
#    for batch_idx, (data, target) in enumerate(test_loader):
#        data, target = data.to(device), target.to(device)
#        out_avg = torch.zeros(len(data)).to(device)
#        for i in range(300, 500):
#            sampler.load_sample_index(i)
#            out = model(data)
#            bias = np.concatenate((bias, (out-target).cpu().numpy()))
#            out_avg += out
#        out_avg /= 200
#        bias_avg = np.concatenate((bias_avg, (out_avg-target).cpu().numpy()))
#        loss += criterion(out_avg, target)** 0.5
#        pbar.update()
#    pbar.close()
##            correct += torch.sum(torch.argmax(out_avg, dim=1) == target)
#    loss = loss.item()
#    loss /= len(test_loader) 
##    error = float(error) / len(test_loader.dataset)
##    print(loss, error)
#    print('\nTest set: Average loss: {:.4f}'.format(
#        loss))
#    plt.hist(bias, 100)
#    plt.show()
#    plt.hist(bias_avg, 100)
#    plt.show()
#    
#    
results = [] 
tau = []
pbar = tqdm(total=450)
for i in range(50, 500):
    sampler.load_sample(i)
    results.append(model.full().detach().cpu().numpy())
    tau.append(model.tau.detach().cpu().numpy())
    pbar.update(1)
pbar.close()
results_array = np.array(results)
tau = np.array(tau)
mean = np.mean(results_array, axis=0)
var = np.mean((results-mean)**2, axis=0)**0.5

#
#points_per_sample = 1
#var2 = np.zeros_like(mean)
#for r, t in zip(results, tau):
#    for i in range(points_per_sample):
#        sample = (r + np.sqrt(T / np.exp(t)) * np.random.normal(size=r.shape))
#        var2 += (sample-mean)**2
#
#var2 /= len(results) / points_per_sample        
var2 = np.zeros_like(mean)
for r, t in zip(results, tau):
    var2 += (r-mean)**2 + (T / np.exp(t))
        

var2 /= len(results) 
        
#samples = np.array(samples)
#var2 = np.mean((samples-mean)**2, axis=0)**0.5
var2 = var2 ** 0.5

err = mean - p['full']
err_n = np.mean(err**2)**0.5
var_n = np.mean(var**2)**0.5
var2n = np.mean(var2**2)**0.5
print('Error {} var {} var2 {}'.format(err_n, var_n, var2n))

#for pos in [0, 10, 20, 30, 40]:
#    plt.imsave('../img/{}_{}_mean.png'.format(nl, pos), mean[:,:,pos])
#    plt.imsave('../img/{}_{}_var.png'.format(nl, pos), var2[:,:,pos])
#    plt.imsave('../img/{}_{}_truth.png'.format(nl, pos), p['full'][:,:,pos])