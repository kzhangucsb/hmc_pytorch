#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 14:41:20 2019

@author: zkq
"""


from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from HMC_sampler_ct import hmcsampler

d1 = 5
d2 = 1
#d3 = 5e-2
S = 0.9


def g(x):
    x = abs(x)
    if x <= d2:
        return 1.0 + 0 * x
    elif x <= d1:
        z = (x-d2) / (d1-d2)
        return 1 - S * (3*(z**2) - 2*(z**3))
    else:
        return 1 - S + 0 * x
#    return 1 - S + 0 * x
    
def phi(x):
    x = abs(x)
    if x <= d1:
        return 0 * x
    else:
        return (x - d1)**2#1 - torch.exp(-((torch.abs(x) - d2) / d3)**2) 



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
#        self.fc1 = nn.Linear(32*32*3, 500)
        self.fc1 = nn.Linear(5*5*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
#        x = x.view(-1, 32*32*3)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    

if __name__ == '__main__':
    # Training settings
    batch_size = 100
    test_batch_size = 100
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    use_cuda = False
    device = torch.device('cuda:0') if use_cuda else torch.device('cpu')
  
#    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    kwargs = {}
#    class MNIST_index(datasets.MNIST):
#        def __getitem__(self, index):
#            img, target = super(MNIST_index, self).__getitem__(index)
#            return (img, target, index)
    
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, **kwargs)


    criterion = nn.CrossEntropyLoss()
    model = Net().to(device)
#    state_dict = torch.load('mnist_10.pth')
#    model.load_state_dict(state_dict)
    sampler = hmcsampler(model, train_loader, criterion, dataloader_test=test_loader,
                         m = 0.001, max_step_size = 1e-4, T = 1e-6, frac_adj=1, lb = 0.5)
    samples_discard, status = sampler.sample(100)
#    sampler.m = 0.01
    samples, status = sampler.sample(200)
    
#    model = Net()
    
    loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            out_avg = torch.zeros(len(data), 10)
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

