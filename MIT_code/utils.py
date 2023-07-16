import torch
from torch.autograd import Variable
import gc
import numpy as np

def create_GUE(n, save_matrix = None):
	matrix = np.random.normal(0,0.5, [n,n]).astype(np.complex_) + np.array([1j])*np.random.normal(0,0.5, [n,n]).astype(np.complex_)
	diag_entries = np.random.normal(0,1,[n,1])
	for i in range(n):
		matrix[i,i] = diag_entries[i]
	# matrix = np.tril(matrix) + np.triu(matrix.T, 1)
	matrix = ( matrix + np.conjugate(matrix.T) ) / 2

	if save_matrix is not None:
		with open(save_matrix, 'wb') as fname:
			pickle.dump(matrix, fname)
	return matrix

def create_qr_random(n):
	matrix = np.random.normal(0,1, [n,n]).astype(np.complex_) + np.array([1j])*np.random.normal(0,1, [n,n]).astype(np.complex_)
	matrix, R = np.linalg.qr(matrix)
	matrix = (matrix + np.conjugate(matrix.T)) / 2
	return matrix

def load_GUE(save_matrix):
	with open(save_matrix, 'rb') as fname:
		return pickle.load(fname)
