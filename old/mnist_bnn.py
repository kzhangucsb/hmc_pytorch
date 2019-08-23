#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 14:51:44 2019

@author: zkq
"""

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from HMC_sampler_bnn import hmcsampler

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
#        self.conv1 = nn.Conv2d(1, 20, 5, 1)
#        self.conv2 = nn.Conv2d(20, 50, 5, 1)
#        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc1 = nn.Linear(28*28, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
#        x = F.relu(self.conv1(x))
#        x = F.max_pool2d(x, 2, 2)
#        x = F.relu(self.conv2(x))
#        x = F.max_pool2d(x, 2, 2)
#        x = x.view(-1, 4*4*50)
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

if __name__ == '__main__':
    # Training settings
    batch_size = 100
    mh_batch_size = 100
    test_batch_size = 100
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    use_cuda = False
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    class MNIST_index(datasets.MNIST):
        def __getitem__(self, index):
            img, target = super(MNIST_index, self).__getitem__(index)
            return (img, target, index)
    
    train_loader = torch.utils.data.DataLoader(
        MNIST_index('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, **kwargs)
    mh_loader = torch.utils.data.DataLoader(
        MNIST_index('../data', train=True, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=mh_batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        MNIST_index('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=mh_batch_size, shuffle=False, **kwargs)


    criterion = nn.CrossEntropyLoss()
    sampler = hmcsampler(Net(), train_loader, criterion, mh_loader, test_loader, 
                         step_size = 1e-4, T = 1e-6, B = 0.6, C = 0,  dataloader_index=True, do_vr_mh=True)
    sampler.do_mh = False
    samples_discard, status = sampler.sample(50)
    sampler.init_hamitlonian_table()
    sampler.do_mh = True
    samples, status = sampler.sample(200)
    
    model = Net()
    loss = 0
    error = 0
    with torch.no_grad():
        for batch_idx, (data, target, index) in enumerate(test_loader):
            out_avg = torch.zeros(len(data), 10)
            for s in samples:
                model.load_state_dict(s)
                out = model(data)
                out_avg += out
            out_avg /= len(samples)
            loss += criterion(out_avg, target)
            error += torch.sum(torch.argmax(out_avg, dim=1) != target)
        loss = loss.item()
    loss /= len(test_loader)
    error = float(error) / len(test_loader.dataset)
    print(loss, error)
        
    
        
# Happy Birthday!!!

