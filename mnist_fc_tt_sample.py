"""
Created on Sun Aug 25 15:53:10 2019

@author: zkq
"""



from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from hmc_sampler_optimizer import hmcsampler, modelsaver_test
from tensor_layer import TTlinear
from tqdm import tqdm



class Net(nn.Module):
    def __init__(self, rank=[(20, 20, 20), (20,)]):
        super(Net, self).__init__()
    
        self.fc1 = TTlinear((4,7,4,7), (4, 5, 5, 5), rank[0], beta=2)
        self.fc2 = TTlinear((20, 25), (2, 5), rank[1], beta=5)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def regularizer(self):
        ret = 0
        for m in self.modules():
            if isinstance(m, TTlinear):
                ret += m.regularizer()
                
        return ret

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
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--num_samples', type=int, default=500, metavar='N',
                        help='number of sampels to get (default: 500)')
    parser.add_argument('--samples_discarded', type=int, default=50, metavar='N',
                        help='number of sampels to discard (default: 50)')
    parser.add_argument('--lamb_ths', default=0.1,
                        help='threshold to reduce rank (default: 0.1)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-bf', action='store_true', default=False,
                        help='Don\'t Use Bayesian model')
#    parser.add_argument('--seed', type=int, default=1, metavar='S',
#                        help='random seed (default: 1)')
    
    parser.add_argument('--save-result', action='store_true', default=True,
                        help='For Saving the current result')
    
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

#    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=False, **kwargs)
    criterian = nn.CrossEntropyLoss()

    
    # reduce rank
    if args.no_bf:
        model = Net().to(device)
        state_dict = torch.load("../models/fashion_mnist_fc_tt_nobf.pth")
        model.load_state_dict(state_dict)
    else:
        model_o = Net().to(device)
        state_dict = torch.load("../models/fashion_mnist_fc_tt.pth")
        model_o.load_state_dict(state_dict)
        test(model_o, test_loader, criterian)
    
        rank = []
        for layer in ['fc1', 'fc2']:
            ths = getattr(model_o, layer).get_lamb_ths()
            rank_i = []
            for i in range(len(ths)):
                ind = list(np.where(state_dict['{}.lamb.{}'.format(layer, i)].cpu().numpy() 
                    < ths[i] - args.lamb_ths)[0])
                rank_i.append(len(ind))
                state_dict['{}.lamb.{}'.format(layer, i)] = \
                    state_dict['{}.lamb.{}'.format(layer, i)][ind]
                state_dict['{}.factors.{}'.format(layer, i)] = \
                    state_dict['{}.factors.{}'.format(layer, i)][:,:,:,ind]
                state_dict['{}.factors.{}'.format(layer, i+1)] = \
                    state_dict['{}.factors.{}'.format(layer, i+1)][ind,:,:,:]
            rank.append(rank_i)
        print('rank={}'.format(rank), flush=True)
        
        model = Net(rank).to(device)
        model.load_state_dict(state_dict)
        test(model, test_loader, criterian)        
        
    
    
    def forwardfcn(x):
        model.eval()
        with torch.no_grad():
            x = x.to(device)
            x = model(x)
            x = x.cpu()
        model.train()
        return x
    
    
#    pbar = tqdm(total=args.num_samples)
    
#    def pbar_update(correct):
#        total = len(test_loader.dataset)
#        pbar.set_postfix_str('Test set: Accuracy: {}/{} ({:.0f}%)'.format(
#                    correct, total,
#                    100. * correct / total), refresh=False)
#        pbar.update()
    
    modelsaver = modelsaver_test(forwardfcn, test_loader)
    
    sampler = hmcsampler(model.parameters(), sampler=modelsaver, max_length=1e-2)
    
    
    
    while(len(sampler.samples) < args.num_samples):
#        model.train()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            sampler.zero_grad()
            output = model(data)
            loss = criterian(output, target) * len(train_loader.dataset)
            loss += model.regularizer() 
            loss.backward()
            sampler.step()
#        test(model, test_loader) 
#    pbar.close()
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
                   "../models/fashion_mnist_fc_tt_nobf_samples.pth")
        else:
            torch.save({'target': modelsaver.target, 'samples': modelsaver.samples}, 
                   "../models/fashion_mnist_fc_tt_samples.pth")
        

