import torch
import numpy as np
from  Hamiltonian import hamilton_operator as hamilton
from torch.autograd import Variable
from torch.nn.functional import conv1d, interpolate
#import sys
import warnings
# from default_potential import vanilla_potential as df_poten
# from binary_potential import   Binary_potnetial as bp_poten


class sampler:

    def __init__(self, potential_struct, T = 1, B = 0,  init_position=None, 
                 init_velocity=None, position_dim=None, step_size=0.05,
                 step_size_min = 1e-3, step_size_max = 1,
                 accept_rate_target=None, accept_avg=0.9, stepsize_scale = 0.9,
                 num_steps_in_leap=20, acceptance_thr=None, 
                 duplicate_samples=False, output_tensor = False):

        self.step_size =step_size
        self.T = T
        self.B = B
        self.output_tensor = output_tensor
        self.init_velocity =None
        self.half_step = 0.5*self.step_size
        self.accept_rate_target = accept_rate_target
        self.accept_avg = accept_avg
        self.stepsize_scale = stepsize_scale
        self.step_size_min = step_size_min
        self.step_size_max = step_size_max
        
        self.num_steps_in_leap = num_steps_in_leap
        self.acceptance_thr = acceptance_thr
        self.duplicate_samples = duplicate_samples
#        if potential_struct is None :
#            self.hamiltonian_obj = hamilton()
#        else :
#            self.hamiltonian_obj = hamilton(potential_struct)
        self.hamiltonian_obj = hamilton(potential_struct)
        if init_velocity is None and  init_position is None and ((position_dim is None) or (position_dim<=0)  ):
            raise ValueError("Neither velocity nor position and nor the dimension has been given.")


        if init_velocity is None and init_position is None and (position_dim>0):
            self.pos_dim = position_dim
            # self.init_velocity = np.random.multivariate_normal(np.zeros(self.pos_dim),np.eye(self.pos_dim,self.pos_dim))
            self.init_position = np.random.multivariate_normal(np.zeros(self.pos_dim),np.eye(self.pos_dim, self.pos_dim))
        if init_velocity is None and init_position is not None :
                self.pos_dim = len(init_position)
                self.init_position =init_position
                # self.init_velocity = np.random.multivariate_normal(np.zeros(self.pos_dim), np.eye(self.pos_dim, self.pos_dim))
        if init_velocity is not None and init_position is  None:
            self.pos_dim = len(init_velocity)
            self.init_velocity = init_velocity
            self.init_position = np.random.multivariate_normal(np.zeros(self.pos_dim),
                                                               np.eye(self.pos_dim, self.pos_dim))

        if init_velocity is not None and init_position is not None:
            ll_pos= len(init_position)
            if not (ll_pos==len(init_velocity)):
                raise ValueError("Lengths of init position and init velocity are not equal. fix them please.")
            '''Lenghts are give an equalt'''
            self.init_position = init_position
            self.init_velocity = init_velocity
            self.pos_dim = ll_pos
        self.gradient = torch.ones(2 * self.pos_dim)
        return


    def main_hmc_loop(self, sample_size):
        bad_decline_cntr = 0
        sample_array= np.array([self.init_position],dtype=np.float64)
        
#        for sample in range (self.sample_size):
        accept_rate = self.accept_rate_target
        while len(sample_array) < sample_size:
            if len(sample_array) == 1 and self.init_velocity is not None:
                tmp_tensor = np.concatenate((sample_array[-1], self.init_velocity), 0)
            else :
                rand_init_velocity = np.random.multivariate_normal(np.zeros(self.pos_dim), np.eye(self.pos_dim)*self.T)
#                rand_init_velocity = np.zeros(self.pos_dim)
                tmp_tensor = np.concatenate((sample_array[-1], rand_init_velocity),0) # position, velocity

#            phase_tensor= Variable(torch.FloatTensor(tmp_tensor), requires_grad=True)
#            new_sample = self.leap_frog_step(phase_tensor)
            new_sample = self.leap_frog_step(tmp_tensor)

            if self.duplicate_samples or not(np.array_equal(new_sample, sample_array[-1])):
                sample_array = np.vstack((sample_array, new_sample))
                accept_rate = accept_rate * self.accept_avg + (1.0 - self.accept_avg)
#                if len(sample_array) % (self.sample_size // 100) == 0:
            else :
                bad_decline_cntr+=1
                accept_rate = accept_rate * self.accept_avg
                
            if self.accept_rate_target is not None:
                self.step_size *= (accept_rate / self.accept_rate_target)
                self.step_size = max(self.step_size, self.step_size_min)
                self.step_size = min(self.step_size, self.step_size_max)
            print('\rGenerated {}/{} samples, declined {} samples, accept rate {}'.format(
                    len(sample_array), sample_size, bad_decline_cntr, accept_rate), end='')
        print('')
        
        self.init_position = new_sample
#        self.init_velocity = new_sample[self.pos_dim:]
        if  bad_decline_cntr > 100:
            warnings.warn("Look out: Many Metropolis Hastings declines.")
        if self.output_tensor:
            sample_array =torch.FloatTensor(sample_array)
        return sample_array, bad_decline_cntr
        
#    def leap_frog_iter(self, tmp_array, tor = 1e-4, max_iter = 100):
##        position = Variable(torch.Tensor(tmp_array[:self.pos_dim]), requires_grad=True)
##        velocity = Variable(torch.Tensor(tmp_array[self.pos_dim:]), requires_grad=False)
#        
#        pv_list = np.tile(tmp_array, (self.num_steps_in_leap+1, 1))
#        pv_list = torch.Tensor(pv_list)
#        conv_weight = torch.Tensor([[[1.0, 2.0, 1.0]]])
#        for it in range(max_iter):
#            # error calculation
#            dpv  = pv_list[1:, :] - pv_list[:-1, :]
#            v    = pv_list[1:, self.pos_dim:]
#            dedp = torch.zeros_like(dpv)
#            for tau in range(self.num_steps_in_leap):
#                position = Variable(pv_list[tau+1, :self.pos_dim], requires_grad=True)
#                velocity = Variable(pv_list[tau+1, self.pos_dim:], requires_grad=False)
#                on_goingrig_hamitlonian = self.hamiltonian_obj(position, velocity) #todo: velocity is not needed here
#                on_goingrig_hamitlonian.backward()
#                dedp[tau, :] = position.grad
#                
#            err_fine   = torch.cat(v, dedp, axis=0) - dpv / self.step_size
#            err_fine   = torch.unsqueeze(torch.t(err_fine), 1)
#            # restriction
#            err_coarse = conv1d(err_fine, conv_weight, stride=2)
#            #correction
#            cor_coarse = torch.cumsum(err_coarse, dim=0)
#            #prolongation
            
        

    def leap_frog_step(self,tmp_array):
        position = Variable(torch.Tensor(tmp_array[:self.pos_dim]), requires_grad=True)
        velocity = Variable(torch.Tensor(tmp_array[self.pos_dim:]), requires_grad=False)
#        on_going_phase = phase_tensor
        orig_hamitlonian = self.hamiltonian_obj(position, velocity)
        orig_hamitlonian.backward()
#        phase_grad = position.grad
#        print(orig_hamitlonian.data)

        for step in range(self.num_steps_in_leap):
            # print "step=",step
#            tmp_array = torch.cat((phase_tensor[:self.pos_dim] + self.step_size * phase_grad[self.pos_dim:],
#                                   phase_tensor[self.pos_dim:] - self.half_step * phase_grad[:self.pos_dim]), 0)
#            xx = Variable(torch.FloatTensor(tmp_array[:self.pos_dim].data), requires_grad=True)
#
#            potential= self.hamiltonian_obj.potential(xx)
#            # potential = binary_potential(xx, weight_mat, bias_array)
#            potential.backward(self.gradient[:self.pos_dim])
#            tmp_array[self.pos_dim:] = tmp_array[self.pos_dim:] - self.half_step * xx.grad
#
#            velocity = Variable(tmp_array[self.pos_dim:].data, requires_grad=True)
#            on_goingrig_hamitlonian = self.hamiltonian_obj.hamiltonian_measure(tmp_array[:self.pos_dim], velocity,   pot_val=potential.data)
            velocity += self.half_step * (position.grad - self.B * velocity)
#            position = Variable(position + self.step_size * velocity, requires_grad=True)
            position.data.add_(self.step_size * velocity)
            position.grad = None
            
            on_goingrig_hamitlonian = self.hamiltonian_obj(position, velocity)
            on_goingrig_hamitlonian.backward()
            velocity += self.half_step * (position.grad - self.B * velocity)
            "Prepare Hamiltonian for next iteration"
            
            position.grad = None
            on_goingrig_hamitlonian = self.hamiltonian_obj(position, velocity)
            on_goingrig_hamitlonian.backward()
#        print(on_goingrig_hamitlonian.data)

        # current_hamiltonian = hamiltonian_measure(tmp_array[:pos_dim], tmp_array[pos_dim:], potential_function, weight_mat, bias_array, pot_val= potential.data[0])
        p_accept = min(1.0, np.exp(- orig_hamitlonian.data + on_goingrig_hamitlonian.data))
        if np.isnan(p_accept) or np.isinf(p_accept):
            raise ValueError('Nan or inf encounted when generating data')

        if self.acceptance_thr is None :
            thr = np.random.uniform()
        else:
            thr = self.acceptance_thr
        if p_accept > thr: # accept
            termination_val= position.data.numpy()
#            print('accept')
        else: # decline
            termination_val= tmp_array[:self.pos_dim]
#            print('decline')


        return termination_val
