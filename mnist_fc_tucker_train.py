"""
Created on Mon Aug 26 22:33:25 2019

@author: zkq
"""





from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
from tensor_layer import tensorizedlinear, TTlinear




class Net(nn.Module):
    def __init__(self, rank=[[(25, 25),(20, 20)], [(20, 20), (10,)]]):
        super(Net, self).__init__()
    
        self.fc1 = tensorizedlinear((28, 28), (20, 25), *rank[0], beta=0.01, c=1e-5)
        self.fc2 = tensorizedlinear((20, 25), (10, ), *rank[1], beta=0.1, c=1e-3)
        for l in self.fc1.lamb_in:
            nn.init.constant_(l, 5)
        for l in self.fc1.lamb_out:
            nn.init.constant_(l, 5)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def regularizer(self):
        ret = 0
        for m in self.modules():
            if isinstance(m, TTlinear) or isinstance(m, tensorizedlinear):
                ret += m.regularizer()
                
        return ret

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
#    parser.add_argument('--seed', type=int, default=1, metavar='S',
#                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--no-bf', action='store_true', default=False,
                        help='Don\'t Use Bayesian model')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    
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
    
#    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer = optim.Adam(model.parameters())
    criterian = nn.CrossEntropyLoss()
    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(total=len(train_loader.dataset), desc='Iter {}'.format(epoch))
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterian(output, target)
            if not args.no_bf:
                loss += model.regularizer() / len(train_loader.dataset)
            loss.backward()
            optimizer.step()
            pbar.set_postfix_str('loss: {:0.4f}'.format(loss.item()), refresh=False)
            pbar.update(len(data))
        pbar.close()
                
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


    if (args.save_model):
        if args.no_bf:
            torch.save(model.state_dict(),"../models/fashion_mnist_fc_tucker_nobf.pth")
        else:
            torch.save(model.state_dict(),"../models/fashion_mnist_fc_tucker.pth")
        

