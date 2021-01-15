#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 14:42:03 2021

@author: zkq
"""


import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
from vgg_tensor_1 import vggBC_TT2
from copy import deepcopy
from hmc_sampler_optimizer import hmcsampler, modelsaver_test
import time
from svgd_base import SVGD_sampler
from tqdm import tqdm
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'

def evaluate_svgd_individual_models(sampler):
    for i,model in enumerate(sampler.models):
        model.eval()
        test_loss = torch.zeros([1])
        correct = torch.zeros([1])
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss+=criterian(output,target).item()
                pred= output.argmax(dim=1)
                correct+=pred.eq(target).sum().item()
            
        test_loss /= len(test_loader)

        print(i,test_loss,correct/10000)

        
        model.train()

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--stepsize', type=float, default=1e-8, metavar='N',
                        help='svgd stepsize')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                        help='input batch size for testing (default: 128)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='Number of epoches to train')
    parser.add_argument('--num-samples', type=int, default=10, metavar='N',
                        help='number of sampels to get (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-bf', action='store_true', default=False,
                        help='Don\'t Use Bayesian model')
    parser.add_argument('--save-result', action='store_true', default=True,
                        help='For Saving the current result')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()


#    torch.cuda.set_device(1)
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.RandomHorizontalFlip(),
                           transforms.RandomCrop(32, 4),
                           transforms.ToTensor(),
                           transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
                       ])),
        batch_size=args.test_batch_size, shuffle=False, **kwargs)


    criterian = nn.CrossEntropyLoss()
    sampler = SVGD_sampler(vggBC_TT2, args.num_samples).to(device)

    for i,model in enumerate(sampler.models):
        model.load_state_dict(torch.load('saved_models/svgd_%s.pth'%(2*i)))
        model.to(device)
    print("Evaluating initial model accuracy")

    evaluate_svgd_individual_models(sampler)

    for i_epoch in range(args.epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                    
                data, target = data.to(device), target.to(device)
                sampler.zero_grad()
                loss = sampler.getloss(data, target, criterian) 
                if not args.no_bf:
                    for model in sampler.models:
                        loss += 0.0*model.regularizer() / len(train_loader.dataset)

                
                loss.backward()
                sampler.step(args.stepsize)
                if batch_idx%20==0:
                    print(loss)
                    evaluate_svgd_individual_models(sampler)
    

    print("Final models accuracy")
    
    evaluate_svgd_individual_models(sampler)

    if args.save_result:
        torch.save([model.state_dict() for model in sampler.models],
                    __file__.split('/')[-1].split('.')[0]+'_sampler.pth')






