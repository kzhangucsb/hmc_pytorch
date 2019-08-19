"""
Created on Thu Aug  8 17:29:26 2019

@author: zkq
"""



from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from vgg_tensor import *
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



if __name__ == '__main__':
    # Training settings
    batch_size = 64
    test_batch_size = 1000
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0') if use_cuda else torch.device('cpu')
  
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
#    kwargs = {}
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
    rank = [(3, 32), None, (32, 32), None, (64, 64), (64, 64), None, 
            (128, 128), (128, 128), None, (128, 128), (128, 128), None, 
#            None, None, None]
            ((32, 32), (32, 32)), ((32, 32), (32, 32)), ((32, 32), (10,))]
    model = vgg11_bn(rank).to(device)
#    state_dict = torch.load('mnist_10.pth')
#    model.load_state_dict(state_dict)
    sampler = hmcsampler(model, train_loader, criterion, dataloader_test=test_loader, g=g, phi=phi,
                         m = 0.1, max_step_size = 1e-4, T = 1e-6, frac_adj=1, lb = 0.5)
    samples_discard, status = sampler.sample(200)
#    sampler.m = 0.01
    samples, status = sampler.sample(200)
    
#    model = Net()
    
    loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            out_avg = torch.zeros(len(data), 10).to(device)
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
    for (ind, s) in enumerate(samples):
        with open('models_vgg/model_{}.pth'.format(ind), 'wb'):
            torch.save(s)
            
        
    
        
# 0.06715278625488282 0.0205
# 0.12032724380493164 0.0198
# 0.4229989242553711 0.0193
# 0.062879638671875 0.019
# 0.06468958854675293 0.0197
# 0.06294259071350097 0.0187

