import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import OrderedDict
from scipy.optimize import curve_fit


def return_steps_and_loss_vecs(df):
	filenames = pd.unique(df['filename'])
	x_vec = []
	y_vec = []
	for file_i in filenames:
		rows = df['filename'] == file_i
		x_vec.append(df[rows]['gradient descent step'].to_numpy())
		y_vec.append(df[rows]['loss'].to_numpy())
	return filenames, x_vec, y_vec

def build_param_colormap(df, dim):
	params = pd.unique(df.sort_values(by=['number of time parameters'])['number of time parameters'])
	print(params)
	colormap = {}
	n = [0,0,0]
	for i in params:
		if abs(i/dim/dim - 1.) < 0.01:
			colormap[i] = n[1]
			n[1] += 1
		elif i < dim*dim:
			colormap[i] = n[0]
			n[0] += 1
		elif i > dim*dim:
			colormap[i] = n[2]
			n[2] += 1
	return colormap

def format_param_str(params, dim):
	new_params = []
	for i in params:
		# new_params.append('{}: ${:0.2}d^2$'.format(i,float(i)/dim/dim))
		new_params.append('${:0.2f}d^2$'.format(float(i)/dim/dim))

	return new_params

def fit_values(x,y):
	def func_powerlaw(x, m, c, x0):
		return (x)**m * c 

	C_guess = 0
	x = np.asarray(x).astype(np.float)
	y = np.asarray(y).astype(np.float)
	popt, _ = curve_fit(func_powerlaw, x[20:200], y[20:200], p0 = np.asarray([-3.,1e6, -10.]))
	# print(popt)
	# print(y[1:])

	return func_powerlaw, popt

def fit_values_exp(x,y):
	def func_exp(x,m,c,k,x0):
		return np.exp(m*(x-x0)) * k + c

	C_guess = 0
	x = np.asarray(x).astype(np.float)
	y = np.asarray(y).astype(np.float)
	popt, _ = curve_fit(func_exp, x, y, p0 = np.asarray([-0.0001,y[-1],y[0],x[0]]), maxfev = 1000)
	if popt[1] < 0.01:
		popt[1] = y[-1]
	print(popt)
	# print(y[1:])

	return func_exp, popt


def plot_line_and_fit(x, y, ax, n_params, dim, fit_values = fit_values, alpha_exp = 1.):

	ax.loglog(x,y, marker='.', markersize = 0.6, color='black', 
		linestyle = '', linewidth=0, alpha = 1., label=n_params)
	fit_fun, fit_data = fit_values(x,y)
	ax.loglog(x[x<(x[-1]/2)], fit_fun(x[x<(x[-1]/2)], *fit_data), marker='', markersize = 0.0, color='red', 
		linestyle = '--', linewidth=1.0, alpha = 1., label=n_params)

	print(n_params)
	print(dim)

	if n_params <= dim*dim*1.01:
		alpha_mult = 0.
	else:
		alpha_mult = 1.
	fit_fun, fit_data = fit_values_exp(x[x>(x[-1]/4)],y[x>(x[-1]/4)])
	ax.loglog(x[x>(x[-1]/20)], fit_fun(x[x>(x[-1]/20)], *fit_data), marker='', markersize = 0.0, color='blue', 
		linestyle = '--', linewidth=1.0, alpha = alpha_exp*alpha_mult, label=n_params)
	param_str = format_param_str([n_params], dim)
	# ax.set_title(param_str[0], loc = 'center', pad = -10, fontdict = {'fontsize': 10})
	ax.text(.05,.1,param_str[0],
		horizontalalignment='left',
		transform=ax.transAxes)
	ax.set_ylim([10E-6, 90])

def build_and_save_plot(df, filename, 
	x_axis_label = "GD Steps (log axis)"):

	filenames, x_vec, y_vec = return_steps_and_loss_vecs(df)
	dim = pd.unique(df['dimension of unitary matrix'])
	dim = dim[0]
	colormap = build_param_colormap(df, dim)

	# style
	plt.style.use('seaborn-darkgrid')
	 
	# create a color palette
	palette = plt.get_cmap('Set1')

	# fig, axes = plt.subplots( 1, 3 , sharex = True, sharey = True, gridspec_kw={'wspace': 0.05})
	fig, axes = plt.subplots( int( (len(filenames) - 1)/2 ), 2 , sharex = True, sharey = True, gridspec_kw={'wspace': 0.03, 'hspace': 0.03})



	for i, file_i in enumerate(filenames):
		n_params = pd.unique(df[df['filename'] == file_i]['number of time parameters'])
		n_params = n_params[0]
		print(n_params)
		if abs(n_params/dim/dim - 1.) < 0.01:
			# axes[colormap[n_params], 1].loglog(x_vec[i], y_vec[i], marker='.', markersize = 0.5, color='gray', 
			# 	linestyle = '', linewidth=0, alpha = 1., label=n_params)
			# fit_fun, fit_data = fit_values(x_vec[i], y_vec[i])
			# axes[colormap[n_params], 1].loglog(x_vec[i][1:], fit_fun(x_vec[i][1:], *fit_data), marker='', markersize = 0.0, color='gray', 
			# 	linestyle = '-', linewidth=0.5, alpha = 1., label=n_params)
			pass
		elif n_params < dim*dim:
			plot_line_and_fit(x_vec[i], y_vec[i], axes[colormap[n_params], 0], n_params, dim)
		elif n_params > dim*dim:
			plot_line_and_fit(x_vec[i], y_vec[i], axes[colormap[n_params], 1], n_params, dim)



	for i in range( int( (len(filenames) - 1)/2 ) ):
		axes[i, 0].set_ylabel("Loss (log axis)")	
	
	axes[int( (len(filenames) - 1)/2 )-1, 0].set_xlabel(x_axis_label)
	axes[int( (len(filenames) - 1)/2 )-1, 1].set_xlabel(x_axis_label)

	axes[0,0].set_title('underparameterized', loc = 'center', pad = 3, fontdict = {'fontsize': 10})
	axes[0,1].set_title('overparameterized', loc = 'center', pad = 3, fontdict = {'fontsize': 10})
	

	fig.set_size_inches(5,6)
	plt.savefig("./figures/"+filename, bbox_inches='tight')	
	plt.close()


def build_and_save_plot_exact(df, filename, 
	x_axis_label = "GD Steps (log axis)"):

	filenames, x_vec, y_vec = return_steps_and_loss_vecs(df)
	dim = pd.unique(df['dimension of unitary matrix'])
	dim = dim[0]
	colormap = build_param_colormap(df, dim)

	# style
	plt.style.use('seaborn-darkgrid')
	 
	# create a color palette
	palette = plt.get_cmap('Set1')

	# fig, axes = plt.subplots( 1, 3 , sharex = True, sharey = True, gridspec_kw={'wspace': 0.05})
	fig, ax = plt.subplots( 1, 1 )



	for i, file_i in enumerate(filenames):
		n_params = pd.unique(df[df['filename'] == file_i]['number of time parameters'])
		n_params = n_params[0]
		print(n_params)
		if abs(n_params/dim/dim - 1.) < 0.01:
			plot_line_and_fit(x_vec[i], y_vec[i], ax, n_params, dim, alpha_exp = 0.)


	fig.set_size_inches(3.75,2.5)

	leg = ax.legend(['data', 'power law fit', 'exponential fit'],
		loc='center left', bbox_to_anchor=(1, 0.5))
	leg.legendHandles[0]._legmarker.set_markersize(5)
	for lh in leg.legendHandles: 
		lh.set_alpha(1)

	ax.set_ylabel("Loss (log axis)")	
	
	ax.set_xlabel(x_axis_label)

	ax.set_title('exactly $d^2$ parameterized', loc = 'center', pad = 3, fontdict = {'fontsize': 10})

	plt.gcf().subplots_adjust(bottom=0.2)
	plt.gcf().subplots_adjust(left=0.2)

	plt.savefig("./figures/"+filename, bbox_inches='tight')	
	plt.close()



if __name__ == '__main__':


	df = pd.read_csv('./csv_files/combined_csv/all_runs_paper_full_5_cut.csv')

	build_and_save_plot(df, 'figure_full_transition_unitary_loglog.pdf')
	# build_and_save_plot(df, 'figure_full_transition_unitary_loglog.png')


	df = pd.read_csv('./csv_files/combined_csv/all_runs_paper_full_5.csv')
	# build_and_save_plot_exact(df, 'figure_full_transition_unitary_loglog_justd2.png')
	build_and_save_plot_exact(df, 'figure_full_transition_unitary_loglog_justd2.pdf')
	build_and_save_power_law_exp(df, 'figure_power_law_fit.pdf')
