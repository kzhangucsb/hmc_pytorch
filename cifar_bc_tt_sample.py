"""
Created on Wed Aug 28 10:32:29 2019

@author: zkq
"""


from __future__ import print_function
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


def test(model, test_loader, criterian):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterian(output, target).item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)), flush=True)
    

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                        help='input batch size for testing (default: 128)')
    parser.add_argument('--num_samples', type=int, default=500, metavar='N',
                        help='number of sampels to get (default: 500)')
    parser.add_argument('--samples_discarded', type=int, default=50, metavar='N',
                        help='number of sampels to discard (default: 50)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-bf', action='store_true', default=True,
                        help='Don\'t Use Bayesian model')
#    parser.add_argument('--seed', type=int, default=1, metavar='S',
#                        help='random seed (default: 1)')
    
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


#    model = Net().to(device)
#    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    criterian = nn.CrossEntropyLoss()
    
    if args.no_bf:
        model = vggBC_TT().to(device)
        state_dict = torch.load("../models/cifar_vggbc_nobf_TT.pth")
        model.load_state_dict(state_dict)
    else:
        model_o = vggBC_TT().to(device)
        state_dict = torch.load("../models/cifar_vggbc_TT.pth")
        model_o.load_state_dict(state_dict)
        test(model_o, test_loader, criterian)
    
        rank = []
        for (layer, ths_o) in zip(['conv1', 'conv2', 'conv3', 'fc0', 'fc1'],
                                  [   0.03,    0.03,    0.03,  0.01,  0.25]):
            ths = getattr(model_o, layer).get_lamb_ths()
            rank_i = []
            for i in range(len(ths)):
                ind = list(np.where(state_dict['{}.lamb.{}'.format(layer, i)].cpu().numpy() 
                    < ths[i] - ths_o)[0])
                rank_i.append(len(ind))
                state_dict['{}.lamb.{}'.format(layer, i)] = \
                    state_dict['{}.lamb.{}'.format(layer, i)][ind]
                state_dict['{}.factors.{}'.format(layer, i)] = \
                    state_dict['{}.factors.{}'.format(layer, i)][:,:,:,ind]
                if (i == len(ths)- 1) and layer.find('conv') == 0:
                    t = state_dict['{}.conv.weight'.format(layer)]
                    s = t.shape
                    t = t.reshape([-1, getattr(model_o, layer).rank[-1], s[2], s[3]])
                    t = t[:, ind, :, :]
                    t = t.reshape([s[0], len(ind), s[2], s[3]])
                    state_dict['{}.conv.weight'.format(layer)] = t
#                    state_dict['{}.conv.weight'.format(layer)] = \
#                        state_dict['{}.conv.weight'.format(layer)][:,ind,:,:]
                else:
                    state_dict['{}.factors.{}'.format(layer, i+1)] = \
                        state_dict['{}.factors.{}'.format(layer, i+1)][ind,:,:,:]
            rank.append(rank_i)
        print('rank={}'.format(rank), flush=True)
        
        model = vggBC_TT(rank).to(device)
        model.load_state_dict(state_dict)
        test(model, test_loader, criterian)        
        
    
    def forwardfcn(x):
        with torch.no_grad():
            x = x.to(device)
            x = model(x)
            x = x.cpu()
        return x
    
    
    modelsaver = modelsaver_test(forwardfcn, test_loader)
    
    sampler = hmcsampler(model.parameters(), sampler=modelsaver, max_length=1e-3)

    while(len(sampler.samples) < args.num_samples):
        model.eval()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            sampler.zero_grad()
            output = model(data)
            loss = criterian(output, target) * len(train_loader.dataset)
            loss += model.regularizer()
            loss.backward()
            sampler.step()
            
            
#        model.eval()
#        test_loss = 0
#        correct = 0
#        with torch.no_grad():
#            for data, target in test_loader:
#                data, target = data.to(device), target.to(device)
#                output = model(data)
#                test_loss += criterian(output, target).item() # sum up batch loss
#                pred = output.argmax(dim=1) # get the index of the max log-probability
#                correct += pred.eq(target).sum().item()
#    
#        test_loss /= len(test_loader)
#        if (epoch + 1) % 20 == 0:
#            scheduler.step()
    
        p = []        
    for s in sampler.samples[args.samples_discarded: args.num_samples]:
        s_softmax = F.log_softmax(s, dim=1)
        p.append(s_softmax)
    p = torch.stack(p, dim=2)
    p = torch.logsumexp(p, dim=2) - np.log(p.shape[2])
    
    target = sampler.sampler.target
    pred = torch.argmax(p, dim=1)
    correct = pred.eq(target).sum().item()
    LL = F.nll_loss(p, target)
    

    print('Overall prediction: Loss {:0.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        LL, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)), flush=True)

        

    

    if (args.save_result):
        if args.no_bf:
            torch.save({'target': modelsaver.target, 'samples': modelsaver.samples}, 
                   "../models/cifar_vggbc_TT_nobf_samples.pth")
        else:
            torch.save({'target': modelsaver.target, 'samples': modelsaver.samples}, 
                   "../models/cifar_vggbc_TT_samples.pth")
        

