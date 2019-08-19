from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from tqdm import tqdm
from hmc_sampler_optimizer import hmcsampler




if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--temperature', type=float, default=1e-6, metavar='N',
                        help='HMC temperature')
    
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
#    if use_cuda:
#        torch.cuda.set_device(1)
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
        batch_size=args.test_batch_size, shuffle=True, **kwargs)


#    model = Net().to(device)
    model = models.vgg11_bn(num_classes=10).to(device)
    with open("../models/cifar_vgg11.pth", 'rb') as f:
        model.load_state_dict(torch.load(f, map_location=device))
    #optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    sampler = hmcsampler(model.parameters(), samples_dir="../models/cifar_vgg11_samples")

    epoch = 0
    while(len(sampler.samples) < 200):
        epoch += 1
        model.train()
        bar = tqdm(total=len(train_loader.dataset), desc='Iter {}'.format(epoch))
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            sampler.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss /= args.temperature
            loss.backward()
            sampler.step()
#            if batch_idx % args.log_interval == 0:
#                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                    epoch, batch_idx * len(data), len(train_loader.dataset),
#                    100. * batch_idx / len(train_loader), loss.item()))
            bar.set_postfix_str('loss: {}'.format(loss.item()), refresh=False)
            bar.update(len(data))
        bar.close()   
            
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
    
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
            
            
            
#    if (args.store_true):
#        torch.save(sampler.samples, '../model/cifar_vgg11_samples.pth')
        


