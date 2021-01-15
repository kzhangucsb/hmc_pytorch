"""
Created on Mon Aug 26 21:33:32 2019

@author: zkq
"""

from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from torchvision import datasets, transforms
from tqdm import tqdm
from vgg_tensor_1 import vggBC_TT2
from copy import deepcopy
#from hmc_sampler_optimizer import hmcsampler


import os
os.environ['CUDA_VISIBLE_DEVICES']='1'


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                        help='input batch size for testing (default: 128)')
    parser.add_argument('--epochs', type=int, default=150, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.04, metavar='LR',
                        help='learning rate (default: 0.04)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--no-bf', action='store_true', default=False,
                        help='Don\'t Use Bayesian model')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--warmup-epochs', type=int)
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    print("warmup epochs ",args.warmup_epochs)

#    torch.cuda.set_device(1)
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 5, 'pin_memory': True} if use_cuda else {}
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
    model = vggBC_TT2().to(device)
#    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = optim.Adam(model.parameters())
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: (epoch+1)**(-0.5))
    criterian = nn.CrossEntropyLoss()

   

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_scaling = torch.clamp(torch.tensor((epoch-args.warmup_epochs)/args.epochs),0.0,1.0)
        #bar = tqdm(total=len(train_loader.dataset), desc='Iter {}'.format(epoch))
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterian(output, target)
            if not args.no_bf:
                loss += epoch_scaling*model.regularizer() / len(train_loader.dataset)
            loss.backward()
            optimizer.step()

            #bar.set_postfix_str('loss: {:0.6f}'.format(loss.item()), refresh=False)
            #bar.update(len(data))

        #bar.close()

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

        print('Epoch:{:.3f} Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            epoch,test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)), flush=True)
        rank = []
        for layer in ['conv1', 'conv2', 'conv3', 'fc0', 'fc1']:
            ths = getattr(model, layer).get_lamb_ths()
            rank_i = []
            for i in range(len(ths)):
                ind = list(np.where(getattr(model, layer).lamb[i].detach().cpu().numpy()
                    < ths[i] - 0.1)[0])
                rank_i.append(len(ind))
            rank.append(rank_i)
        print('rank={}'.format(rank), flush=True)


        if (args.save_model) and args.epochs-epoch<15:

            i = 0
            while os.path.exists("saved_models/svgd_{}.pth".format(i)):
                i+=1

            torch.save(model.state_dict(),
                       "saved_models/svgd_{}.pth".format(i))



    if args.no_bf:
        if (args.save_model):
            torch.save(model.state_dict(),
                       "../models/cifar_vggbc_nobf_TT_{}.pth".format(int(np.round(time.time()))))
    else:
        if (args.save_model):

            i = 0
            while os.path.exists("saved_models/svgd_{}.pth".format(i)):
                i+=1

            torch.save(model.state_dict(),
                       "saved_models/svgd_{}.pth".format(i))

        state_dict = deepcopy(model.state_dict())
        rank = []
        for layer in ['conv1', 'conv2', 'conv3', 'fc0', 'fc1']:
            ths = getattr(model, layer).get_lamb_ths()
            rank_i = []
            for i in range(len(ths)):
                ind = list(np.where(state_dict['{}.lamb.{}'.format(layer, i)].cpu().numpy()
                    < ths[i] - 0.03)[0])
                rank_i.append(len(ind))
    #            state_dict['{}.lamb.{}'.format(layer, i)] = \
    #                state_dict['{}.lamb.{}'.format(layer, i)][ind]
    #            state_dict['{}.factors.{}'.format(layer, i)] = \
    #                state_dict['{}.factors.{}'.format(layer, i)][:,:,:,ind]
    #            state_dict['{}.factors.{}'.format(layer, i+1)] = \
    #                state_dict['{}.factors.{}'.format(layer, i+1)][ind,:,:,:]
            rank.append(rank_i)
        print('rank={}'.format(rank), flush=True)

