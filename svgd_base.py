import torch
import torch.distributions as td
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform
from copy import deepcopy

class SVGD:
    def __init__(self, particles_shape):
        self.historical_grad_square = torch.zeros(particles_shape)
        self.iter=0

    def SVGD_kernal(self, x, h=-1):
        init_dist = pdist(x.cpu())
        pairwise_dists = torch.tensor(squareform(init_dist),dtype=torch.float).to(x.device)

        if h < 0:  # if h < 0, using median trick
            h = torch.median(pairwise_dists)
            h = torch.tensor(h ** 2 / np.log(x.shape[0] + 1),dtype=torch.float).to(x.device)

        kernal_xj_xi = torch.exp(- pairwise_dists ** 2 / h)
        d_kernal_xi = torch.zeros(x.shape).to(x.device)

        for i_index in range(x.shape[0]):
            d_kernal_xi[i_index] = torch.matmul(kernal_xj_xi[i_index], x[i_index] - x) * 2 / h

        return kernal_xj_xi, d_kernal_xi

    def update(self, x0, grad_x0, stepsize=1e-4, bandwidth=-1, alpha=0.9, debug=False):

        x = x0

        # adagrad with momentum
        eps_factor = 1e-8

        kernal_xj_xi, d_kernal_xi = self.SVGD_kernal(x, h=-1)
        current_grad = (torch.matmul(kernal_xj_xi, grad_x0) + d_kernal_xi) / x.shape[0]
        if self.iter == 0:
            self.historical_grad_square += current_grad ** 2
        else:
            self.historical_grad_square = alpha * self.historical_grad_square + (1 - alpha) * (current_grad ** 2)
        adj_grad = current_grad / torch.sqrt(self.historical_grad_square + eps_factor)
        x += stepsize * adj_grad

        self.iter+=1

        return x
    
    
class SVGD_sampler:
    def __init__(self, Net, num_sample):
        
        self.models = [Net() for _ in range(num_sample)]
        num_paras = 0
        for p in self.models[0].parameters():
            if p.requires_grad:
                num_paras += p.numel()
            
        self.params = torch.zeros(num_sample, num_paras)
        self.grads = torch.zeros(num_sample, num_paras)
        self.svgd = SVGD((num_sample, num_paras))
        
    def to(self, device):
        self.models = [m.to(device) for m in self.models]
        self.params = self.params.to(device)
        self.grads = self.grads.to(device)
        self.svgd.historical_grad_square = self.svgd.historical_grad_square.to(device)
        return self
        
    def _model_to_params(self):
        for (i, model) in enumerate(self.models):
            offset = 0
            for p in model.parameters():
                if p.requires_grad:
                    self.params[i, offset : offset + p.numel()].copy_(p.data.view(-1))
                    self.grads[i, offset : offset + p.numel()].copy_(p.grad.view(-1))
                
            offset += p.numel()
                
    def _params_to_model(self):
        for (i, model) in enumerate(self.models):
            offset = 0
            for p in model.parameters():
                if p.requires_grad:
                    p.data.view(-1).copy_(self.params[i, offset : offset + p.numel()])
                
            offset += p.numel()
            
    def step(self, *args, **kwargs):
        self._model_to_params()
        self.svgd.update(self.params, -self.grads, *args, **kwargs)
        self._params_to_model()
        
    def getloss(self, data, target, criterian):
        loss = 0
        for model in self.models:
            output = model(data)
            loss += criterian(output, target)
            
        return loss
        
    def zero_grad(self):
        for (i, model) in enumerate(self.models):
             model.zero_grad()
             
    def train(self):
        for (i, model) in enumerate(self.models):
             model.train()
             
    def eval(self):
        for (i, model) in enumerate(self.models):
             model.eval()
        
        
                
            
            
            
        
            
        

"""
test_dist = td.MultivariateNormal(torch.tensor([0.0,0.0]),torch.eye(2))
init_dist = td.Normal(torch.tensor(-1.0),torch.tensor(1.0))

particles = torch.nn.Parameter(init_dist.sample([10,2]))


svgd = SVGD(particles.shape)

stepsize = 1e-2

for _ in range(100):
    loss = torch.sum(test_dist.log_prob(particles))

    loss.backward()

    particles.data.copy_(svgd.update(particles,particles.grad,stepsize=stepsize))

    particles.grad.zero_()

plt.scatter(particles.detach().numpy()[:,0],particles.detach().numpy()[:,1]) 
plt.xlim(-3,3)
plt.ylim(-3,3)
"""
