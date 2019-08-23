from HMC_sampler import sampler
import torch
from   torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from linear_regression_batch import linear_regression


#Regular run
#hmc = sampler(sample_size=100,position_dim=5)
#sample,_= hmc.main_hmc_loop()

#print("Init place")
#hmc = sampler(sample_size=100,init_position=np.array([1.0,2.1,3.1,-1.]))
#sample,_= hmc.main_hmc_loop()
#print(sample)

#class test_potential:
#
#    def __init__(self, weight_matrix):
#        self.weight_matrix = weight_matrix
#        
#    def __call__(self, xx):
#        return self.calc_potential_energy(xx)
#
#    def calc_potential_energy (self, xx):
#        potential_energy=torch.dot(xx,torch.matmul(self.weight_matrix,xx))
#        return potential_energy
#Regular run
#print("Potential")

#
#Amat = torch.FloatTensor([[-2, 0, 0, 0], 
#                          [0, -2, 0, 0], 
#                          [0, 0, -2, 0], 
#                          [0, 0, 0, -2]])
    
dim = 4
bias = True

c = linear_regression(dim, lamb=0.1, bias=bias)
c.generate_data(20000, scale=1, noise_db=-np.inf)
ps = lambda x: -20*c.get_batch_regularized_loss(x) 

w, b = c.get_ground_truth(lr=2)
if bias:
    init_position = np.append(w.numpy(), b)
else:
    init_position = w 
   
hmc = sampler(position_dim = dim + bias, step_size = 0.02, potential_struct = ps, T = 0.1, B = 0)
#sample,rej_cnt = hmc.main_hmc_loop(1000)
sample,rej_cnt = hmc.main_hmc_loop(200)
#print(sample)
dist_x = np.linspace(-2, 2, 100)
#plt.hist(sample[:, 0], dist_x)

loss = np.array([c.get_regularized_loss(torch.FloatTensor(sample[i])) for i in range(len(sample))])
plt.hist(loss, 50)
plt.show()

err = np.array([c.get_error(torch.FloatTensor(sample[i])) for i in range(len(sample))])
plt.hist(err, 50)
plt.show()



#print(np.append(c.beta.numpy(), c.beta0))
#print(np.mean(sample, 0))
#print(np.mean(sample**2, 0) - np.mean(sample, 0)**2)
#z = np.exp(-2 * dist_x**2)
#plt.plot(dist_x, len(sample)*z/sum(z))

#print("Init vel")
#hmc = sampler(sample_size=100,position_dim=4,init_velocity=np.array([1.0,2.1,3.1,-1.]))
#sample,_= hmc.main_hmc_loop()
#print(sample)

