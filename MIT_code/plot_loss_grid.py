import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.colors as colors

def mesh_data(data):
	n_x = len(np.unique(data[:,0]))
	n_y = len(np.unique(data[:,1]))

	x = data[:,0].reshape(n_y, n_x)
	y = data[:,1].reshape(n_y, n_x)
	z = data[:,2].reshape(n_y, n_x)
	return x.T,y.T,z.T

def get_shrunken_levels(z, n_levels = 20):
	levels_raw = np.linspace(0,1,n_levels)
	levels_raw = levels_raw**(1/3)
	z_max = np.max(z)
	z_min = np.min(z)
	levels = levels_raw*(z_max - z_min) + z_min
	print(levels) 
	return levels


def build_and_save_plot(data, filename):

	# style
	plt.style.use('seaborn-darkgrid')
	 
	# create a color palette
	# palette = plt.get_cmap('Set1')
	cmap = plt.get_cmap('PiYG')


	levels = get_shrunken_levels(data[2])

	Z = data[2]

	plt.pcolor(data[0], data[1], data[2], norm=colors.DivergingNorm(vmin=Z.min(), vcenter=65., vmax=Z.max()), cmap='RdPu') 
	# plt.pcolormesh(data[0], data[1], data[2]**5, cmap='RdPu') #vmin
	# plt.contourf(data[0], data[1], np.log(data[2]), cmap=cmap)

	cbr = plt.colorbar()	
	cbr.set_label('Loss')

	plt.contour(data[0], data[1], data[2], levels, colors = 'gray', linewidths = 0.5)

	# plt.subplots_adjust(right=0.75)

	plt.ylabel(r"parameter 1 ($t_1$)")
	plt.xlabel(r'parameter 2 ($\tau _1$)')

	fig = plt.gcf()
	fig.set_size_inches(5,4)

	# plt.savefig("./figures/"+filename)	
	plt.savefig("./figures/"+filename, dpi = 300)	

	plt.close()






data = pd.read_csv('saved_data/2d_loss_grid.csv', sep=',')
print(data.values)
x,y,z = mesh_data(data.values)
# build_and_save_plot( [x,y,z], 'loss_grid.pdf')
build_and_save_plot( [x,y,z], 'loss_grid.jpg')
