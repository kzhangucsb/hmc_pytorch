
import torch
import numpy as np
from  Hamiltonian import hamilton_operator as hamilton
from torch.autograd import Variable
from torch.nn.functional import conv1d, interpolate
#import sys
import warnings
import copy
# from default_potential import vanilla_potential as df_poten
# from binary_potential import   Binary_potnetial as bp_poten


class hmcsampler:

    def __init__(self, model, dataloader, criterion, dataloader_mh=None, dataloader_test = None, 
                 dataloader_index = False, do_mh = False, do_vr=False, do_vr_mh = False, do_test=True, 
                 T = 1, B = 0, C = 0, step_size=0.05, step_size_min = 1e-3, step_size_max = 1,
                 accept_rate_target=None, accept_avg=0.9, stepsize_scale = 0.9,
                 num_steps_in_leap=64, acceptance_thr=None, 
                 duplicate_samples=False):

        self.model = model
#        self.velocity = [torch.zeros_like(p) for p in params]        
            
        self.dataloader = dataloader
        if dataloader_mh is None:
            self.dataloader_mh = dataloader
        else:
            self.dataloader_mh = dataloader_mh
        if dataloader_test is None:
            self.dataloader_test = dataloader
        else:
            self.dataloader_test = dataloader_test
        self.dataloader_mh2 = copy.deepcopy(self.dataloader_mh)
            
        self.criterion = criterion
        self.step_size =step_size
        
        if (not dataloader_index) and (do_vr or do_vr_mh):
            raise ValueError('Index is needed for variance reduction')
        
        self.dataloader_index = dataloader_index
        self.do_mh = do_mh # Metropolis–Hastings
        self.do_vr = do_vr # variance reduce
        self.do_vr_mh = do_vr_mh # variance reduce for Metropolis–Hastings
        self.do_test = True
        self.T = T
        self.B = B
        self.C = C 
        
        
        if self.do_vr_mh:
            self.mh_loss_list = torch.zeros(len(self.dataloader_mh.dataset))
           
#        self.half_step = 0.5*self.step_size
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
        
        return


    def sample(self, sample_size):
        bad_decline_cntr = 0
        sample_array = [copy.deepcopy(self.model.state_dict())]
        
#        for sample in range (self.sample_size):
        accept_rate = self.accept_rate_target if self.accept_rate_target is not None else 0
        
        velocity = dict()
        for key, para in sample_array[0].items():
            velocity[key] = torch.zeros_like(para)
    
        while len(sample_array) < sample_size:
#            velocity = [torch.randn_like(p) * self.T for p in self.model.parameters()]
            for key, item in velocity.items():
                item.normal_()
            

            h_old = self.get_hamitlonian(velocity) if self.do_mh else 0
            
            if self.dataloader_index:
                data, labels, index = next(iter(self.dataloader_mh))
            else:
                data, labels = next(iter(self.dataloader_mh))
            output = self.model(data)
            loss = self.criterion(output, labels) / self.T
            for p in velocity.values():
                loss += (torch.norm(p)**2) / 2
            h_old_est = loss.item()     
#            phase_tensor= Variable(torch.FloatTensor(tmp_tensor), requires_grad=True)
#            new_sample = self.leap_frog_step(phase_tensor)
            new_velocity = self.leap_frog_step(velocity)
            h_new = self.get_hamitlonian(new_velocity) if self.do_mh else 0
            
            output = self.model(data)
            loss = self.criterion(output, labels) / self.T
            for p in new_velocity.values():
                loss += (torch.norm(p)**2) / 2
            h_new_est = loss.item() 
            
            print (h_new - h_old, h_new_est - h_old_est)
            
            if self.do_test:
                with torch.no_grad():
    #            self.model.load_state_dict(params)
                    if self.dataloader_index:
                        data, labels, index = next(iter(self.dataloader_test))
                    else:
                        data, labels = next(iter(self.dataloader_test))
                    output = self.model(data)
                    loss = self.criterion(output, labels)
            else:
                loss = None

            if not self.do_mh or self.mh(h_old, h_new):
                sample_array.append(copy.deepcopy(self.model.state_dict()))
                accept_rate = accept_rate * self.accept_avg + (1.0 - self.accept_avg)
#                if len(sample_array) % (self.sample_size // 100) == 0:
            else :
                bad_decline_cntr+=1
                self.model.load_state_dict(copy.deepcopy(sample_array[-1]))
                accept_rate = accept_rate * self.accept_avg
                
            if self.accept_rate_target is not None:
                self.step_size *= (accept_rate / self.accept_rate_target)
                self.step_size = max(self.step_size, self.step_size_min)
                self.step_size = min(self.step_size, self.step_size_max)
            print('Generated {}/{} samples, declined {} samples, loss {}, h rate {}'.format(
                    len(sample_array), sample_size, bad_decline_cntr, loss, h_old - h_new))
        print('')
        
#        self.init_position = sample_array[-1]
#        self.init_velocity = new_sample[self.pos_dim:]
        if  bad_decline_cntr > 100:
            warnings.warn("Look out: Many Metropolis Hastings declines.")
        return sample_array, {'bad_decline_cntr': bad_decline_cntr}
        

    def leap_frog_step(self, velocity):
#        on_going_phase = phase_tensor
#        params   = copy.deepcopy(params)
#        velocity = copy.deepcopy(velocity)
#        self.model.load_state_dict(params)
        position = dict(self.model.named_parameters())
        if self.dataloader_index:
            data, labels, index = next(iter(self.dataloader))
        else:
            data, labels = next(iter(self.dataloader))
        output = self.model(data)
        loss = self.criterion(output, labels)
        loss.backward()
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
            for key in position:
                velocity[key] -= self.step_size / 2 * (position[key].grad + self.B * velocity[key]) \
                    + torch.randn_like(velocity[key]) * (self.C * self.step_size) ** 0.5
            
#            position = Variable(position + self.step_size * velocity, requires_grad=True)
            for key in position:
                position[key].data.add_(self.step_size * velocity[key])
                position[key].grad = None
            if self.dataloader_index:
                data, labels, index = next(iter(self.dataloader))
            else:
                data, labels = next(iter(self.dataloader)) 
            output = self.model(data)
            loss = self.criterion(output, labels) / self.T
            loss.backward()
            for key in position:
                velocity[key] -= self.step_size / 2 * (position[key].grad + self.B * velocity[key]) \
                    + torch.randn_like(velocity[key]) * (self.C * self.step_size) ** 0.5
                
        return velocity
            
#        print(on_goingrig_hamitlonian.data)
        
    def get_hamitlonian(self, velocity):
        with torch.no_grad():
#            self.model.load_state_dict(params)
#            if self.dataloader_index:
#                data, labels, index = next(iter(self.dataloader_mh))
#            else:
#                data, labels = next(iter(self.dataloader_mh))
            
#            output = self.model(data)
#            if self.do_vr_mh:
#                h_old = torch.mean(self.mh_loss_list)
#                h_diff = 0
#                for i, j in enumerate(index):
#                     h = self.criterion(output[i:i+1], labels[i:i+1])
#                     h_diff += (h - self.mh_loss_list[j]) 
##                         self.mh_loss_list[j] = h
##                self.mh_loss_list[index] = self.criterion(output[i:i+1], labels[i:i+1])
#                h_diff /= len(index)
#                loss = h_old + h_diff#torch.mean(self.mh_loss_list)
#            else:
#                loss = self.criterion(output, labels)
#            
            loss2 = 0
            for batch_idx, (data, target, index) in enumerate(self.dataloader_mh2):
                out = self.model(data)
                loss2 += self.criterion(out, target)
            loss2 /= len(self.dataloader_mh2)
#            print(loss.item(), loss2.item())
        loss = loss2
        loss /= self.T
        for p in velocity.values():
            loss += (torch.norm(p)**2) / 2
        return loss.item() 

    def mh(self, h_original, h_new):
        # Metropolis–Hastings
        
#        h_original = self.get_hamitlonian(orig_params, orig_velocity)
#        h_new      = self.get_hamitlonian(new_params, new_velocity)
        p_accept = min(1.0, np.exp(h_original - h_new))
        if np.isnan(p_accept) or np.isinf(p_accept):
            raise ValueError('Nan or inf encounted when generating data')

        if self.acceptance_thr is None :
            thr = np.random.uniform()
        else:
            thr = self.acceptance_thr
        return p_accept > thr
    def init_hamitlonian_table(self):
        self.mh_loss_list = torch.zeros(len(self.dataloader_mh.dataset))
        with torch.no_grad():
            for data, labels, index in self.dataloader_mh:
                output = self.model(data)
                for i, j in enumerate(index):
                    self.mh_loss_list[j] = self.criterion(output[i:i+1], labels[i:i+1]) 

