"""
Created on Wed Aug 28 14:52:24 2019

@author: zkq
"""



from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
from vgg_tensor_1 import vggBC
from copy import deepcopy
from hmc_sampler_optimizer import hmcsampler, modelsaver_test
#from hmc_sampler_optimizer import hmcsampler



    

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                        help='input batch size for testing (default: 128)')
    parser.add_argument('--epochs', type=int, default=150, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--num_samples', type=int, default=500, metavar='N',
                        help='number of sampels to get (default: 500)')
    parser.add_argument('--samples_discarded', type=int, default=50, metavar='N',
                        help='number of sampels to discard (default: 50)')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--no-init', action='store_true', default=False,
                        help='Skip initialization')
    parser.add_argument('--no-sample', action='store_true', default=False,
                        help='Skip sampling')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()


#    torch.cuda.set_device(1)
    device = torch.device("cuda:1" if use_cuda else "cpu")

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
    model = vggBC().to(device)
    criterian = nn.CrossEntropyLoss()
    
    if args.no_init:
        model.load_state_dict(torch.load("../models/cifar_vggbc.pth"))
    else:
        optimizer = optim.Adam(model.parameters())
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: (epoch+1)**(-0.5))
        
    
        for epoch in range(1, args.epochs + 1):
            model.train()
            bar = tqdm(total=len(train_loader.dataset), desc='Iter {}'.format(epoch))
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterian(output, target)
                loss.backward()
                optimizer.step()
                
                bar.set_postfix_str('loss: {:0.6f}'.format(loss.item()), refresh=False)
                bar.update(len(data))
    
            bar.close()       
                
            model.eval()
            test_loss = 0
            correct = 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    test_loss += criterian(output, target).item() # sum up batch loss
                    pred = output.argmax(dim=1) # get the index of the max log-probability
                    correct += pred.eq(target).sum().item()
        
            test_loss /= len(test_loader)
            if (epoch + 1) % 20 == 0:
                scheduler.step()
        
            print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
                test_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)), flush=True)
            
    
        if (args.save_model):
            torch.save(model.state_dict(),"../models/cifar_vggbc.pth")
    
    if not args.no_sample:
        
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
                loss.backward()
                sampler.step()
            
        
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
            torch.save({'target': modelsaver.target, 'samples': modelsaver.samples}, 
                   "../models/cifar_vggbc_samples.pth")
       
