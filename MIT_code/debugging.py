# import torch
import math
# from unitary_optimizer import unitary_optimizer

# # selecting float32 for data type (can also be float64)
# dtype = torch.float32

# # selecting device for pytorch (either gpu or cpu)
# gpuid = 0
# # device = torch.device("cuda:"+ str(gpuid))
# device = torch.device("cpu")

# # directory for saving any matrices or other variables
# save_dir = '/hd2/research/ML_Theory/unitary_opt/pytorch_save/'

# # default grad_enabled to False
# torch.set_grad_enabled(True)




# a = torch.tensor([5], dtype = torch.float32)
# b = a*2
# a = a+1
# print(a)

print(str(1552))


# # construct A and B as simple pauli matrices
# pauli_X_and_Y = torch.zeros( (2,2,2,2), dtype = dtype, device = device )
# # construct pauli X as first matrix element (note, no imaginary)
# pauli_X_and_Y[0,0,:,:] = torch.tensor( [[0,1], [1,0]], dtype = dtype, device = device ) 
# # construct pauli Y as second matrix element (note, no real)
# pauli_X_and_Y[1,1,:,:] = torch.tensor( [[0,-1], [1,0]], dtype = dtype, device = device )

# time_params = torch.tensor([ [0.99, 0], [0, 0] ], dtype = dtype, device = device)

# a = unitary_optimizer(pauli_X_and_Y, time_params = time_params)
# # a = unitary_optimizer(pauli_X_and_Y, random_time_target = 2)

# print(a.output)

# grid_times = torch.linspace(-1,1,steps=10, device = device, dtype = dtype)
# grid_times = torch.reshape(grid_times, [10,1])
# print(grid_times)
# print(torch.reshape(a.control_matrices_real[0,:,:], [1,4,4]))
# print(a.grid_search_times())
# print(a.output)
# def full_function(t, A, B):
# 	a_exp = torch.cos(t)*re_id+torch.sin(t)*im_id
# 	print(a_exp)
# 	return torch.sqrt(torch.sum((a_exp-B)**2)) / 2

# def partial_function(A,B):
# 	return torch.sqrt(torch.sum((A-B)**2)) / 2

# def imaginary_identity(shape_out):
# 	'''Creates identity matrix that has i as value on all diagonal entries
# 	'''
# 	shape_out = list(shape_out)
# 	imaginary_id = torch.zeros([1]+shape_out[1:], dtype = dtype, device = device)
# 	half_point = int(shape_out[1]/2)
# 	imaginary_id[:, :half_point, half_point:] = -torch.eye(half_point, dtype = dtype, device = device)
# 	imaginary_id[:, half_point:, :half_point] = torch.eye(half_point, dtype = dtype, device = device)
# 	return imaginary_id.expand(shape_out)

# def real_identity(shape_out):
# 	'''Create batched identity matrix
# 	'''
# 	shape_out = list(shape_out)
# 	real_id = torch.eye(shape_out[1], dtype = dtype, device = device).view([1]+shape_out[1:])
# 	return real_id.expand(shape_out)

# def construct_matrix(controls, time_params):
# 	print('time_params in step construct_matrix:')
# 	print(time_params)
# 	matrix_exp = batch_matrix_exp(controls, time_params)
# 	out_mat = batch_matrix_multiply(matrix_exp)
# 	return out_mat

# def batch_matrix_multiply(matrices):
# 	n_mats = matrices.size()[0]
# 	# if only one matrix left, recursion is done
# 	if n_mats == 1:
# 		return matrices
# 	# if odd number, multiply last matrices to get even number
# 	if n_mats % 2 == 1:
# 		matrices[1,:,:] = torch.matmul(matrices[1,:,:], matrices[0,:,:])
# 		matrices = matrices[1:, :, :]
# 		n_mats = n_mats-1

# 	even_mats = list(range(0,n_mats,2))
# 	odd_mats = [i+1 for i in even_mats]
# 	print('batch_matrix_multiply (pre):')
# 	print(matrices)
# 	matrices = torch.matmul(matrices[odd_mats], matrices[even_mats])
# 	print('batch_matrix_multiply (post):')
# 	print(matrices)
# 	return batch_matrix_multiply(matrices)

# def batch_matrix_scalar(scalars, matrices):
# 	n, r, c = list(matrices.size())
# 	print('times in batch multiply:')
# 	print(scalars)
# 	return (matrices.view((n,r*c))*scalars).view((n,r,c))

# def batch_matrix_exp(matrices, times):
# 	depth = int(times.size()[0]/matrices.size()[0])
# 	print('times in exponentiation:')
# 	print(times)
# 	print('times in cos exponentiation:')
# 	test = torch.cos(times)
# 	print(test)
	
# 	expanded_matrices = matrices.repeat([depth, 1, 1])
# 	expanded_matrices = batch_matrix_scalar(torch.cos(times), real_identity( expanded_matrices.size() )) + \
# 						batch_matrix_scalar(torch.sin(times), 
# 											torch.matmul(imaginary_identity( expanded_matrices.size() ), expanded_matrices ) )
	
# 	print('expanded_matrices in exponentiation:')
# 	print(expanded_matrices)
# 	return expanded_matrices

# def format_time_tensor(time_tensor, grad_bool = True):
# 	depth = time_tensor.size()[1]
# 	times_tiled = time_tensor.t().reshape((time_tensor.size()[0]*depth, 1))
# 	return torch.tensor(times_tiled, device = device, dtype = dtype, requires_grad = grad_bool)

# def randomly_initialize_times(n_times, grad_bool = True):
# 	rand_times = ( torch.rand( (2, n_times), 
# 		device = device, dtype = dtype) - 0.5 ) * 2*math.pi
# 	return format_time_tensor(rand_times, grad_bool = grad_bool)



# a = torch.tensor([[[1,2,0,1],[0,0,1,0],[0,1,0,0], [1,0,0,0]], [[2,2,0,1],[0,0,1,0],[0,1,0,0], [1,0,0,0]]], 
# 	device = device, dtype = dtype)

# b = torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,1,0], [0,0,0,1]], 
# 	device = device, dtype = dtype)

# re_id = torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,1,0], [0,0,0,1]], 
# 	device = device, dtype = dtype)

# im_id = torch.tensor([[0,0,-1,0],[0,0,0,-1],[1,0,0,0], [0,1,0,0]], 
# 	device = device, dtype = dtype)

# print(torch.transpose(a, 1, 2))
# print(torch.sum(torch.diagonal(a, dim1 = 1, dim2 = 2), 1))
