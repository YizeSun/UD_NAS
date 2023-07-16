import torch
from torch.autograd import Variable
import torch.optim as optim
import gc
import numpy as np
import math
from expm_module import torch_expm
import datetime
import pandas as pd
import config

# selecting float32 for data type (can also be float64)
dtype = config.dtype

# selecting device for pytorch (either gpu or cpu)
gpuid = config.gpuid
device = config.device
# device = torch.device("cpu")

# default grad_enabled
torch.set_grad_enabled(config.grad_enabled_bool)

# directory for saving any matrices or other variables
save_dir = '/hd2/research/ML_Theory/unitary_opt/pytorch_save/'
save_csv = './csv_files/'

class unitary_optimizer():

	def __init__(self, control_matrices, time_params = None, target = None, random_time_target = None):
		'''Initializes a unitary optimizer.
		Inputs:
			Required:
				control_matrices: batched list of pytorch matrices multiplied in order
			One of the following:
				time_params: batched list of time vectors for the control matrices to construct the target
				target: target matrix in pytorch format
				random_time_target = integer of number of random multiplications of control matrices to create target
		'''
		super(unitary_optimizer, self).__init__()

		self.manual_grad_calc = False # default to False, can be set to True during optimize phase

		if len(control_matrices.size()) == 4:
			self.control_matrices_real = complex_matrix_setup(control_matrices)
			self.control_matrices = control_matrices
		else:
			self.control_matrices_real = control_matrices
			self.control_matrices = convert_to_4d_batch(control_matrices)

		self.n_control_matrices = self.control_matrices.size()[0]
		self.dim_matrix = self.control_matrices_real.size()[1]

		if target is not None:
			self.target = target
			if len(self.target.size()) == 2:
				self.target = self.target.reshape([1]+list(self.target.size()))
			if len(self.target.size()) == 4:
				self.target = complex_matrix_setup(self.target)
			self.n_target_times = -1
		elif time_params is not None:
			time_params = self.format_time_tensor(time_params, grad_bool = True)
			self.target = self.construct_matrix(time_params)
			self.n_target_times = time_params.size()[1]
		elif random_time_target is not None:
			self.target = self.construct_random_matrix(random_time_target)
			self.n_target_times = random_time_target
		else:
			raise ValueError('either target, random_time_target, or time_params must be inputted when initializing unitary_optimizer')

		self.output = real_identity(self.target.size())

		

	def full_loss_calc(self):
		self.output = self.construct_matrix(self.times)
		loss = frobenius_norm(self.output, self.target)
		return loss

	def format_time_tensor(self, time_tensor, grad_bool = True):
		depth = time_tensor.size()[1]
		times_tiled = time_tensor.t().reshape((time_tensor.size()[0]*depth, 1))
		return torch.tensor(times_tiled, device = device, dtype = dtype, requires_grad = grad_bool)

	def construct_matrix(self, time_params, multiply_forward = True, forward_pass = True):
		if self.manual_grad_calc:
			self.matrix_exp, self.imaginary_matrices = batch_matrix_exp(self.control_matrices_real, time_params, return_imaginary_matrices = True)
		else:
			self.matrix_exp = batch_matrix_exp(self.control_matrices_real, time_params)
		if self.manual_grad_calc and forward_pass:
			out_mat = self.ordered_matrix_multiply(self.matrix_exp, multiply_forward = multiply_forward)
		else:
			out_mat = batch_matrix_multiply(self.matrix_exp)
		return out_mat

	def construct_random_matrix(self, n_times = 10, grad_bool = True):
		times = self.randomly_initialize_times(n_times, grad_bool = grad_bool)
		return self.construct_matrix(times)

	def randomly_initialize_times(self, n_times, grad_bool = True, uniform_range = 2.):
		rand_times = ( torch.rand( (self.n_control_matrices, n_times), 
			device = device, dtype = dtype) - 0.5 ) * 2*uniform_range
		return self.format_time_tensor(rand_times, grad_bool = grad_bool)

	def greedy_optimize(self, max_n_times = 100, epochs_per = 1000, print_statistics = True, 
		manual_grad_calc = True, init_type = 'zeros', optim_type = 'Adam', relative_stop_rate = 1e-8,
		absolute_stop_rate = 1e-7,	print_every = 100, min_epochs = 15, init_range = 0.01, normalize_learning_rate = 1.0,
		 **kwargs):
		'''optimizes by adding a layer of times each round and optimizing until convergence

		Inputs:
			max_n_times: (int) maximum depth (e.g. 10 means there will be 10 parameters per control matrix)
			epochs_per: (int) epochs of training per round
			print_statistics: (bool) prints error values if set to True
			print_every: (int) print error values every given iteration
			manual_grad_calc: (bool) determines whether pytorch automatically calculates gradients or manually calculated (set to True for now)
			init_type: (string) choose from 'zeros', 'random', or 'search'
			optim_type: (string) choose 'Adam', 'SGD', or 'LBFGS' for optimization method
			relative_stop_rate: (double) value of relative stop rate to move onto next round
			absolute_stop_rate: (double) value to stop after convergence reached
			**kwargs: arguments passed into the chosen pytorch optimizer
		'''
		
		# set values for inputs
		self.relative_stop_rate = relative_stop_rate
		self.absolute_stop_rate = absolute_stop_rate
		self.max_epochs = epochs_per
		self.min_epochs = min_epochs

		self.optim_type = optim_type

		if init_type == 'random':
			self.times = self.randomly_initialize_times(1, grad_bool = True)
		elif init_type == 'search':
			self.times = self.search_times_append(grid_range = init_range)
		elif init_type == 'zeros':
			init_times = torch.zeros([self.n_control_matrices, 1], device = device, dtype = dtype)
			self.times = self.format_time_tensor(init_times)
		else:
			raise ValueError('Please input valid initialization type')

		self.n_times = 1

		self.manual_grad_calc = manual_grad_calc

		for round_i in range(max_n_times):
			print()
			print('Beginning round {}'.format(round_i+1))
			if print_statistics:
				print('Starting times for round:')
				print(self.times)
			
			kwargs['lr'] = kwargs['lr']/ normalize_learning_rate

			if optim_type == 'Adam':
				self.optimizer = optim.Adam([self.times], **kwargs)
			elif optim_type == 'SGD':
				self.optimizer = optim.SGD([self.times], **kwargs)
			elif optim_type == 'LBFGS':
				self.optimizer = optim.LBFGS([self.times], **kwargs)
			else:
				raise ValueError('Please input valid optimization type')

			loss = self.optimize_round(print_statistics = True, print_every = print_every)

			if loss < self.absolute_stop_rate:
				print('Final loss: {}'.format(loss))
				print('Times:')
				print(self.times)
				return round_i

			print('Loss at end of round: {}'.format(loss))
			if print_statistics:
				print('New Times:')
				print(self.times)

			# if not achieve absolute_stop_rate, append new times
			self.append_times(init_type, init_range)
		return -1	


	def append_times(self, init_type, init_range):
		'''method used in greedy_optimize to append time to self.times

		Inputs:
			init_type: (string) set to given strings to specify type of initialization
			init_range: (double) range of initialization for randomization or search
		'''
		if init_type == 'random':
			new_times = self.randomly_initialize_times(1, grad_bool = True, uniform_range = init_range)
		elif init_type == 'search':
			new_times = self.search_times_append(grid_range = init_range)
		elif init_type == 'zeros':
			init_times = torch.zeros([self.n_control_matrices, 1], device = device, dtype = dtype)
			new_times = self.format_time_tensor(init_times)
		else:
			raise ValueError('Please input valid initialization type')

		self.times = torch.cat( (self.times, new_times) )
		self.n_times += 1


	def optimize_round(self, print_statistics = True, print_every = 10):
		''' method used in greedy_optimize to optimize each round
		'''
		# intialize epoch number and loss values
		epoch_i = 0
		change_loss = self.relative_stop_rate*10 # initialize to something larger than the relative_stop rate
		prior_loss = 99999999

		while (epoch_i < self.max_epochs and change_loss > self.relative_stop_rate) \
		or epoch_i < self.min_epochs:
			if not self.manual_grad_calc:
				self.optimizer.zero_grad()
			
			loss = self.full_loss_calc()

			if print_statistics:
				if epoch_i%print_every == 0:
					print('[%d] loss: %.3E' % (epoch_i , loss ))
			
			if self.manual_grad_calc:
				self.times.grad = self.manual_gradients()
			else:
				loss.backward()

			if self.optim_type == 'LBFGS':
				self.optimizer.step(self.full_loss_calc)
			else:
				self.optimizer.step()

			change_loss = (prior_loss - loss) / prior_loss
			prior_loss = loss		
			epoch_i += 1

		if print_statistics:
			print('Total number of epochs: {}'.format(epoch_i))

		return loss


	def search_times_append(self, n_grid = 1001, grid_range = math.pi, appending = True):
		# create grid of times
		grid_times = torch.linspace(-1*grid_range,grid_range,steps=n_grid, device = device, dtype = dtype)
		grid_times = torch.reshape(grid_times, [n_grid,1])
		
		# initialize vector containing new times
		new_times = torch.zeros([self.n_control_matrices, 1], device = device, dtype = dtype)

		for i, matrix_i in enumerate(self.control_matrices_real):
			# find optimal new time and matrix
			optim_time, optim_matrix = self.grid_loss_search(torch.reshape(matrix_i, [1,self.dim_matrix, self.dim_matrix]), 
															 grid_times, appending = appending)
			new_times[i] = optim_time
			# update output to take into account change
			self.output = torch.reshape(optim_matrix, [1,self.dim_matrix, self.dim_matrix])
		
		return new_times

	def propogate_round(self, n_grid = 1001, grid_range = math.pi, appending = False):
		
		# getting a backwards pass 		
		self.construct_matrix(self.times, multiply_forward = False)

		# start back_matrix at identity
		forward_matrix = torch.eye(self.dim_matrix, dtype = dtype, device = device).reshape(-1,self.dim_matrix, self.dim_matrix)

		# loop backwards in forward pass (start from second to last matrix being multiplied)
		for i, back_matrix_i in enumerate(self.backward_pass):
			# print(i)
			# print('back_matrix')
			# print(back_matrix_i)
			# print('forward_matrix')
			# print(forward_matrix)

			# create grid of times
			grid_times = self.times[i] + torch.linspace(-1*grid_range,grid_range,steps=n_grid, device = device, dtype = dtype)
			grid_times = torch.reshape(grid_times, [n_grid,1])
			# print(grid_times)

			matrix_i = self.control_matrices_real[i%self.n_control_matrices]
			# print('matrix')
			# print(matrix_i)


			# find optimal new time and matrix
			optim_time, optim_matrix, optim_exp = self.grid_loss_search(torch.reshape(matrix_i, [1,self.dim_matrix, self.dim_matrix]), 
															 grid_times, appending = appending,
															 back_matrix = torch.reshape(back_matrix_i, [1,self.dim_matrix, self.dim_matrix]),
															 forward_matrix = torch.reshape(forward_matrix, [1,self.dim_matrix, self.dim_matrix])
															 )
			# update output to take into account change
			self.output = torch.reshape(optim_matrix, [1,self.dim_matrix, self.dim_matrix])

			# update time
			self.times[i] = optim_time

			# update forward_matrix
			forward_matrix = torch.matmul(optim_exp ,forward_matrix)
			# print(self.output)

			# print(self.full_loss_calc())
		
		# return new_times

	def grid_loss_search(self, matrix, times, appending = True, back_matrix = None, forward_matrix = None):
		'''performs matrix multiplications on matrix and determines which has lowest loss
		'''
		# print(back_matrix)


		matrix = batch_matrix_exp(matrix, times)
		if appending:
			new_outputs = torch.matmul(matrix,self.output.expand(matrix.size()))
		else:
			new_outputs = torch.matmul(matrix, forward_matrix.expand(matrix.size()))
			new_outputs = torch.matmul(back_matrix.expand(matrix.size()), new_outputs)

		loss_values = batch_frobenius_norm(new_outputs, self.target.expand(new_outputs.size()))
		min_ind = torch.argmin(loss_values)
		# print(loss_values)
		# print(loss_values[min_ind])
		# print(matrix[min_ind])

		# print(forward_matrix)
		if appending:
			return times[min_ind], new_outputs[min_ind]
		else:
			return times[min_ind], new_outputs[min_ind], matrix[min_ind]
		

	def optimize(self, n_times = 2, n_epochs = 2000, print_statistics = True, 
		manual_grad_calc = True, init_times = None, optim_type = 'Adam',
		print_every = 100, save_results = False, absolute_stop_rate = 1e-7,
		include_propogation = False, track_times = False,
		propogate_every = 100, propogate_n_grid = 1001, propogate_grid_range = 5.0,
		**kwargs):
		'''
		Inputs:
			**kwargs = variable list of arguments for the pytorch adam optimizer
		'''

		if save_results:
			loss_tracking = []
			epoch_list = []
			l2_dist_list = []
			l1_dist_list = []

		if track_times:
			self.time_tracker = []

		if init_times is None:
			self.times = self.randomly_initialize_times(n_times, grad_bool = True)
			self.n_times = n_times
		else:
			self.n_times = int(init_times.size()[1])
			self.times = self.format_time_tensor(init_times)

		self.init_times = self.times.clone().detach()

		if optim_type == 'Adam':
			self.optimizer = optim.Adam([self.times], **kwargs)
		elif optim_type == 'SGD':
			self.optimizer = optim.SGD([self.times], **kwargs)
		elif optim_type == 'LBFGS':
			self.optimizer = optim.LBFGS([self.times], **kwargs)
		else:
			raise ValueError('Please input valid optimization type')
		self.manual_grad_calc = manual_grad_calc

		for epoch_i in range(n_epochs):
			if include_propogation:
				if epoch_i%propogate_every == 0:
					print('propogating at epoch {}'.format(epoch_i))
					self.propogate_round(n_grid = propogate_n_grid, 
										 grid_range = propogate_grid_range, 
										 appending = False)


			if not self.manual_grad_calc:
				self.optimizer.zero_grad()
			
			loss = self.full_loss_calc()
			if print_statistics:
				if epoch_i%print_every == 0:
					print('[%d] loss: %.3E' % (epoch_i + 1, loss ))
					if save_results:
						epoch_list.append(epoch_i)
						loss_tracking.append(loss.data.cpu().numpy())

						diff_times = self.times - self.init_times
						l1_norm = torch.sum( torch.abs(diff_times) )
						l1_dist_list.append(l1_norm.data.cpu().numpy())
						l2_norm = torch.sqrt( torch.sum( diff_times*diff_times ) )
						l2_dist_list.append(l2_norm.data.cpu().numpy())
			
			if manual_grad_calc:
				self.times.grad = self.manual_gradients()
			else:
				loss.backward()

			if loss < absolute_stop_rate:
				print('Final loss: {}'.format(loss))
				break

			if optim_type == 'LBFGS':
				self.optimizer.step(self.full_loss_calc)
			else:
				self.optimizer.step()

			if track_times:
				self.time_tracker.append(self.times.clone().detach())



		if save_results:
			epoch_list.append(epoch_i+1)
			loss_tracking.append(loss.data.cpu().numpy())
			diff_times = self.times - self.init_times
			l1_norm = torch.sum( torch.abs(diff_times) )
			l1_dist_list.append(l1_norm.data.cpu().numpy())
			l2_norm = torch.sqrt( torch.sum( diff_times*diff_times ) )
			l2_dist_list.append(l2_norm.data.cpu().numpy())

			pd_cols = {'number of time parameters': n_times*self.n_control_matrices,
				'dimension of unitary matrix': self.dim_matrix/2,
				'number of target parameters': self.n_target_times*self.n_control_matrices,
				'gradient descent step': epoch_list,
				'l1 distance traversed': l1_dist_list,
				'l2 distance traversed': l2_dist_list,
				'loss': loss_tracking
				}

			df = pd.DataFrame(data=pd_cols)
			currentDT = datetime.datetime.now()
			df.to_csv(save_csv+str(currentDT)+'.csv')
			print(df)
			

	def manual_gradients(self):
		''' manaully calculates gradients using backwards pass
		'''
		# initialize gradients

		self.times.grad = torch.zeros([self.n_times*self.n_control_matrices, 1], 
					device = device, dtype = dtype)

		self.ordered_matrix_multiply(self.matrix_exp, multiply_forward = False)

		grad_mats = torch.matmul( self.backward_pass, torch.matmul(self.imaginary_matrices,self.forward_pass) )
		grad_mats = torch.matmul(torch.transpose(self.target, 1, 2).expand(grad_mats.size()), grad_mats)
		grads = -1*torch.sum(torch.diagonal(grad_mats, dim1 = 1, dim2 = 2), 1)
		return grads.reshape([len(grads),1])


	def ordered_matrix_multiply(self, matrices_in, multiply_forward = True):
		matrices = torch.clone(matrices_in)
		n_mats = matrices.size()[0]
		dim_mat = matrices.size()[1]
		if multiply_forward:
			for i in range(n_mats - 1):
				matrices[i+1] = torch.matmul(matrices[i+1], matrices[i])
		else:
			shifted_matrices = torch.zeros(matrices.size(), device = device, dtype = dtype)
			shifted_matrices[-1,:,:] = torch.eye(matrices.size()[1], device = device, dtype = dtype)
			for i in range(n_mats - 1):
				shifted_matrices[-(i+2)] = torch.matmul(shifted_matrices[-(i+1)], matrices[-(i+1)])
				

		if self.manual_grad_calc:
			if multiply_forward:
				self.forward_pass = matrices
			else:
				self.backward_pass = shifted_matrices 
				return None 					# no return if backwards pass

		return matrices[-1].view([1] + list(matrices.size()[1:]))


	def get_loss_grid(self,
					  direction1 = None, direction2 = None, start_point = None, 
					  n_grid_steps = 5, grid_size = 0.25):
		'''returns 2d grid of loss values
		useful for plotting the loss function as a contour
		'''

		if direction1 is None:
			direction1 = torch.randn(*self.times.size(), 
				device = device, dtype = dtype)

		if direction2 is None:
			direction2 = torch.randn(*self.times.size(), 
				device = device, dtype = dtype)

		if start_point is None:
			start_point = self.times.clone().detach()


		# normalizing directions
		l1 = torch.sqrt(torch.sum( direction1**2 ))
		l2 = torch.sqrt(torch.sum( direction2**2 ))
		d1 = direction1 / l1
		d2 = direction2 / l2

		# get points in grid
		x_points = torch.linspace(-1*grid_size, grid_size, steps = n_grid_steps )
		y_points = torch.linspace(-1*grid_size, grid_size, steps = n_grid_steps )

		# initialize essentials
		x_vals = []
		y_vals = []
		loss_vals = []
		temp_store_times = self.times.clone().detach()

		# loop through and calculate loss
		for x in x_points:
			print(x)
			for y in y_points:
				self.times = start_point + x*d1 + y*d2
				loss = self.full_loss_calc()
				x_vals.append(float(x.data.cpu()))
				y_vals.append(float(y.data.cpu()))
				loss_vals.append(float(loss.data.cpu()))

		self.times = temp_store_times
		return x_vals, y_vals, loss_vals




def batch_matrix_multiply(matrices):
	n_mats = matrices.size()[0]
	# if only one matrix left, recursion is done
	if n_mats == 1:
		return matrices
	# if odd number, multiply last matrices to get even number
	if n_mats % 2 == 1:
		matrices[1,:,:] = torch.matmul(matrices[1,:,:], matrices[0,:,:])
		matrices = matrices[1:, :, :]
		n_mats = n_mats-1

	even_mats = list(range(0,n_mats,2))
	odd_mats = [i+1 for i in even_mats]
	matrices = torch.matmul(matrices[odd_mats], matrices[even_mats])
	return batch_matrix_multiply(matrices)

def batch_matrix_exp(matrices, times, is_pauli = False, return_imaginary_matrices = False):
	depth = int(times.size()[0]/matrices.size()[0])

	if is_pauli:
		expanded_matrices = matrices.repeat([depth, 1, 1])
		imaginary_matrices = torch.matmul(imaginary_identity( expanded_matrices.size() ), expanded_matrices )
		expanded_matrices = batch_matrix_scalar(torch.cos(times), real_identity( expanded_matrices.size() )) + \
							batch_matrix_scalar(torch.sin(times), imaginary_matrices )
	else:
		imaginary_matrices = torch.matmul(imaginary_identity( matrices.size() ), matrices )
		imaginary_matrices = imaginary_matrices.repeat([depth, 1, 1])
		expanded_matrices = batch_matrix_scalar(times, imaginary_matrices)
		expanded_matrices = torch_expm(expanded_matrices)

	if return_imaginary_matrices:
		return [expanded_matrices, imaginary_matrices]
	else:
		return expanded_matrices

def batch_matrix_scalar(scalars, matrices):
	n, r, c = list(matrices.size())
	return (matrices.view((n,r*c))*scalars).view((n,r,c))

def complex_matrix_setup(batch_in):
	'''Converts nx2xdxd matrix to a nx2*dx2*d matrix to compute imaginary
	matrix calculations using real numbers
	'''
	size_in = batch_in.size()
	n = size_in[0]
	d = size_in[2]

	# initialize output matrix
	batch_out = torch.zeros( (n, d*2, d*2), dtype = dtype, device = device )

	# first setup real numbers
	batch_out[:, :d, :d] = batch_in[:,0,:,:]
	batch_out[:, d:, d:] = batch_in[:,0,:,:]

	# next, setup imaginary numbers
	batch_out[:, :d, d:] = batch_in[:,1,:,:] * (-1)
	batch_out[:, d:, :d] = batch_in[:,1,:,:]

	return batch_out


def convert_to_4d_batch(batch_in):
	'''Converts nx2*dx2*d matrix to nx2xdxd matrix to simplify viewing matrices
	'''
	size_in = batch_in.size()
	n = int(size_in[0])
	d = int(size_in[1]/2)

	# initialize output matrix
	batch_out = torch.zeros( (n, 2, d, d), dtype = dtype, device = device )

	# first setup real numbers
	batch_out[:,0,:,:] = batch_in[:,:d,:d]

	# next, setup imaginary numbers
	batch_out[:,1,:,:] = batch_in[:, d:, :d]

	return batch_out

def frobenius_norm(A,B):
	''' Calculates Frobenius Norm of A-B, assuming format of complex structured matrix
	'''
	mat_norm = torch.matmul(torch.transpose(A, 1, 2).expand(B.size()), B)
	trace = torch.sum(torch.diagonal(mat_norm, dim1 = 1, dim2 = 2), 1)

	return mat_norm.size()[1]*trace.size()[0] - torch.sum(trace)

def batch_frobenius_norm(A,B):
	''' Calculates Frobenius Norm of A-B, assuming format of complex structured matrix
	Performed in batches where multiple As and Bs can be inputted
	'''
	mat_norm = torch.matmul(torch.transpose(A, 1, 2).expand(B.size()), B)
	trace = torch.sum(torch.diagonal(mat_norm, dim1 = 1, dim2 = 2), 1)

	return mat_norm.size()[1] - trace


def imaginary_identity(shape_out):
	'''Creates identity matrix that has i as value on all diagonal entries
	'''
	shape_out = list(shape_out)
	imaginary_id = torch.zeros([1]+shape_out[1:], dtype = dtype, device = device)
	half_point = int(shape_out[1]/2)
	imaginary_id[:, :half_point, half_point:] = -torch.eye(half_point, dtype = dtype, device = device)
	imaginary_id[:, half_point:, :half_point] = torch.eye(half_point, dtype = dtype, device = device)
	return imaginary_id.expand(shape_out)

def real_identity(shape_out):
	'''Create batched identity matrix
	'''
	shape_out = list(shape_out)
	real_id = torch.eye(shape_out[1], dtype = dtype, device = device).view([1]+shape_out[1:])
	return real_id.expand(shape_out)