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
from vgg_tensor_1 import vggBC_TT
from copy import deepcopy
from hmc_sampler_optimizer import hmcsampler, modelsaver_test
import time
from svgd_base import SVGD_sampler
from tqdm import tqdm


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                        help='input batch size for testing (default: 128)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='Number of epoches to train')
    parser.add_argument('--num_samples', type=int, default=10, metavar='N',
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
    sampler = SVGD_sampler(vggBC_TT, args.num_samples).to(device)
        

    for i_epoch in range(args.epochs):
        with tqdm(total=len(train_loader.dataset), desc='Iter {}'.format(i_epoch)) as bar:
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                sampler.zero_grad()
                loss = sampler.getloss(data, target, criterian) 
                if not args.no_bf:
                    for model in sampler.models:
                        loss += model.regularizer() / len(train_loader.dataset)
                loss.backward()
                sampler.step()
                
                bar.set_postfix_str('loss: {:0.6f}'.format(loss.item()), refresh=False)
                bar.update(len(data))
                break

    if args.save_result:
        torch.save([model.state_dict() for model in sampler.models],
                    __file__.split('/')[-1].split('.')[0]+'_sampler.pth')






