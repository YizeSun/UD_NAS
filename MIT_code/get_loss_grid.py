import unittest
import torch
from torch.autograd import Variable
import numpy as np
from unitary_optimizer import unitary_optimizer
from unitary_optimizer import batch_matrix_exp, batch_matrix_multiply
import math
import utils
import config


# selecting float32 for data type (can also be float64)
dtype = config.dtype

# selecting device for pytorch (either gpu or cpu)
gpuid = config.gpuid
device = config.device
# device = torch.device("cpu")

# default grad_enabled
torch.set_grad_enabled(config.grad_enabled_bool)





n = 32		# dimension
A = utils.create_GUE(n)	# unitary 1
B = utils.create_GUE(n)	# unitary 2


random_A_B = torch.zeros( (2,2,n,n), dtype = dtype, device = device )		#formatting
random_A_B[0,0,:,:] = torch.tensor(A.real, dtype = dtype, device = device )	#formatting
random_A_B[0,1,:,:] = torch.tensor(A.imag, dtype = dtype, device = device )	#formatting
random_A_B[1,0,:,:] = torch.tensor(B.real, dtype = dtype, device = device )	#formatting
random_A_B[1,1,:,:] = torch.tensor(B.imag, dtype = dtype, device = device )	#formatting

# getting random time parameters for target
time_params = np.random.uniform(-3,3, size = [2, 1])
time_params = torch.tensor(time_params, dtype = dtype, device = device)

# initialize your optimizer
a = unitary_optimizer(control_matrices = random_A_B, time_params = time_params)


# setup optimization (but don't run any steps)
a.optimize(n_epochs = 0, lr = 0.000, weight_decay = 0.0, manual_grad_calc = True, track_times = True,
           n_times = 1, optim_type = 'SGD', save_results = False, print_every = 1 ) # n_times = n_layers

# get directions to plot grid for
dir1 = torch.tensor([[1],[0]], dtype = dtype, device = device)
dir2 = torch.tensor([[0],[1]], dtype = dtype, device = device)
x,y,l = a.get_loss_grid(dir1, dir2,
	n_grid_steps = 451, grid_size = 7)

x = np.asarray(x).reshape(-1,1)
y = np.asarray(y).reshape(-1,1)
l = np.asarray(l).reshape(-1,1)

out_data = np.concatenate((x,y,l), axis = 1)
np.savetxt('saved_data/2d_loss_grid.csv', out_data, delimiter = ',', header = 'x,y,l')