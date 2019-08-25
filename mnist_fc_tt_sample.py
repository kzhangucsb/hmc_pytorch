"""
Created on Sun Aug 25 15:53:10 2019

@author: zkq
"""



from __future__ import print_function
import argparse
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from hmc_sampler_optimizer import hmcsampler, modelsaver_test
from model_fashion_mnist_fc_0 import Net

from tqdm import tqdm

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--num_samples', type=int, default=500, metavar='N',
                        help='number of sampels to get (default: 500)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
#    parser.add_argument('--seed', type=int, default=1, metavar='S',
#                        help='random seed (default: 1)')
    
    parser.add_argument('--save-result', action='store_true', default=False,
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
        batch_size=args.test_batch_size, shuffle=True, **kwargs)


    model = Net().to(device)
    model.load_state_dict(torch.load("../models/fashion_mnist_fc_tt.pth"))
    
    def forwardfcn(x):
        model.eval()
        with torch.no_grad():
            x = x.to(device)
            x = model(x)
            x = x.cpu()
        model.train()
        return x
    
    
    pbar = tqdm(total=args.num_samples)
    
    def pbar_update(correct):
        total = len(test_loader.dataset)
        pbar.set_postfix_str('\nTest set: Accuracy: {}/{} ({:.0f}%)'.format(
                    correct, total,
                    100. * correct / total), refresh=False)
        pbar.update()
    
    modelsaver = modelsaver_test(forwardfcn, test_loader, post_handler=pbar_update)
    
    sampler = hmcsampler(model.parameters(), sampler=modelsaver_test(forwardfcn, test_loader))
    criterian = nn.CrossEntropyLoss()
    
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            sampler.zero_grad()
            output = model(data)
            loss = criterian(output, target)
            loss += model.regularizer() / len(train_loader.dataset)
            loss.backward()
            sampler.step()
            
    pbar.close()

    if (args.save_model):
        torch.save(sampler.samples,"../models/fashion_mnist_fc_tt_samples.pth")
        

