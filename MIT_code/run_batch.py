import unittest
import torch
from torch.autograd import Variable
import numpy as np
from unitary_optimizer import unitary_optimizer
from unitary_optimizer import batch_matrix_exp, batch_matrix_multiply
import math
import utils
import config
from scipy.stats import unitary_group

# selecting float32 for data type (can also be float64)
dtype = config.dtype

# selecting device for pytorch (either gpu or cpu)
gpuid = config.gpuid
device = config.device
# device = torch.device("cpu")

# default grad_enabled
torch.set_grad_enabled(config.grad_enabled_bool)

n = 32			# dimension
lr = 0.0001		# learning rate for gd
n_repeats = 10  # number of time to randomly repeat for given set of parameters

save_results = True 		# set to True to save outputs as csv files
print_freq = 1				# prints to csv file at every print_freq steps

# list of dimensions for target unitary relative to number of parameters
dimensions_raw = [0.1, 0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 2.0]

dimensions = [int(p*n*n/2.) for p in dimensions_raw]
print(dimensions)

# list of parameters in target unitary (set to None to have Haar Random)
n_targets = [None]					# number of parameters in target unitary (set to None to have Haar Random)
# n_targets = [int(n*n)]			# number of parameters in target unitary (set to None to have Haar Random)



for i, ii in zip(dimensions, dimensions_raw):


	for j in n_targets:
		for k in range(n_repeats):

			A = utils.create_GUE(n)	# unitary 1
			B = utils.create_GUE(n)	# unitary 2
			random_A_B = torch.zeros( (2,2,n,n), dtype = dtype, device = device )		#formatting
			random_A_B[0,0,:,:] = torch.tensor(A.real, dtype = dtype, device = device )	#formatting
			random_A_B[0,1,:,:] = torch.tensor(A.imag, dtype = dtype, device = device )	#formatting
			random_A_B[1,0,:,:] = torch.tensor(B.real, dtype = dtype, device = device )	#formatting
			random_A_B[1,1,:,:] = torch.tensor(B.imag, dtype = dtype, device = device )	#formatting

			# if number of targets is set to None, then we select a random Haar unitary as target
			if j is None:			
				target_unitary = unitary_group.rvs(n)
				target_formatted = torch.zeros( (1,2,n,n), dtype = dtype, device = device )		#formatting
				target_formatted[0,0,:,:] = torch.tensor(target_unitary.real, dtype = dtype, device = device )	#formatting
				target_formatted[0,1,:,:] = torch.tensor(target_unitary.imag, dtype = dtype, device = device )	#formatting
				a = unitary_optimizer(control_matrices = random_A_B, target = target_formatted )
			else:
				a = unitary_optimizer(control_matrices = random_A_B, random_time_target = j )	
			


			# example setup for vanilla gradient descent optimizer
			a.optimize(n_epochs = 10000, lr = lr/ii, weight_decay = 0.0, manual_grad_calc = True,
						n_times = i, optim_type = 'SGD', save_results = save_results, print_every = print_freq,
						absolute_stop_rate = 1e-7*n*n, momentum = 0 ) # n_times = n_layers

			# example setup for adam optimizer
			# a.optimize(n_epochs = 10000, lr = 0.001/ii, weight_decay = 0.0, manual_grad_calc = True,
			# 			n_times = i, optim_type = 'Adam', save_results = save_results, print_every = print_freq,
			# 			absolute_stop_rate = 1e-7*n*n, amsgrad = True ) # n_times = n_layers