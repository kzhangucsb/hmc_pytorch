
"""
Created on Mon Jul 29 18:47:24 2019

@author: zkq
"""



from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
from torch.utils.data import TensorDataset, DataLoader
from HMC_sampler_bnn import hmcsampler
from cp import cp

d1 = 0.05
d2 = 0.01
#d3 = 5e-2
S = 0.9



    
    



if __name__ == '__main__':
    # Training settings


    batchsize = 28
    
    with open('ktensor.pickle', 'rb') as f:
        p = pickle.load(f)
    
    size = p['size']
    rank = p['rank']
    train_input = torch.LongTensor(p['train']['indexes']).t()
    train_value = torch.Tensor(p['train']['values'])
    #train_norm = torch.norm(train_value).item()
    test_input = torch.LongTensor(p['test']['indexes']).t()
    test_value = torch.Tensor(p['test']['values'])
    #test_norm = torch.norm(test_value).item()
    
    model = cp(size, rank)
    train_loader = DataLoader(TensorDataset(train_input, train_value),
            batch_size = batchsize, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_input, test_value),
            batch_size = batchsize, shuffle=False)
    
    use_cuda = False
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
#    class MNIST_index(datasets.MNIST):
#        def __getitem__(self, index):
#            img, target = super(MNIST_index, self).__getitem__(index)
#            return (img, target, index)
    
    


    criterion = nn.MSELoss()
    model = cp(size, rank)
    state_dict = torch.load('mnist_10.pth')
#    model.load_state_dict(state_dict)
    sampler = hmcsampler(model, train_loader, criterion, dataloader_test=test_loader, 
                          num_steps_in_leap=32)
#    samples_discard, status = sampler.sample(50)
#    sampler.m = 0.01
    samples, status = sampler.sample(200)
    
#    model = Net()
    
    loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            out_avg = torch.zeros(len(data))
            for s in samples:
                model.load_state_dict(s)
                out = model(data)
                out_avg += out
            out_avg /= len(samples)
            loss += criterion(out_avg, target)
            correct += torch.sum(torch.argmax(out_avg, dim=1) == target)
        loss = loss.item()
    loss /= len(test_loader)
#    error = float(error) / len(test_loader.dataset)
#    print(loss, error)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
        
    
        
# 0.06715278625488282 0.0205
# 0.12032724380493164 0.0198
# 0.4229989242553711 0.0193
# 0.062879638671875 0.019
# 0.06468958854675293 0.0197
# 0.06294259071350097 0.0187

