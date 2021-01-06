import torch
import torch.distributions as td
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform

class SVGD:
    def __init__(self,particles_shape):
        self.historical_grad_square = torch.zeros(particles_shape)
        self.iter=0

    def SVGD_kernal(self, x, h=-1):
        init_dist = pdist(x)
        pairwise_dists = torch.tensor(squareform(init_dist),dtype=torch.float)
        if h < 0:  # if h < 0, using median trick
            h = torch.median(pairwise_dists)
            h = torch.tensor(h ** 2 / np.log(x.shape[0] + 1),dtype=torch.float)

        kernal_xj_xi = torch.exp(- pairwise_dists ** 2 / h)
        d_kernal_xi = torch.zeros(x.shape)

        for i_index in range(x.shape[0]):
            d_kernal_xi[i_index] = torch.matmul(kernal_xj_xi[i_index], x[i_index] - x) * 2 / h

        return kernal_xj_xi, d_kernal_xi

    def update(self, x0, grad_x0, stepsize=1e-3, bandwidth=-1, alpha=0.9, debug=False):

        x = x0.clone().detach()

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