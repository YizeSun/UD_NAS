import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import OrderedDict


def return_steps_and_loss_vecs(df):
	filenames = pd.unique(df['filename'])
	x_vec = []
	y_vec = []
	for file_i in filenames:
		rows = df['filename'] == file_i
		x_vec.append(df[rows]['gradient descent step'].to_numpy())
		y_vec.append(df[rows]['loss'].to_numpy())
	return filenames, x_vec, y_vec

def build_param_colormap(df):
	params = pd.unique(df.sort_values(by=['number of time parameters'])['number of time parameters'])
	print(params)
	colormap = {}
	n = 0
	for i in params:
		colormap[i] = n
		n += 1
	return colormap

def format_param_str(params, dim):
	new_params = []
	dim = dim[0]
	for i in params:
		# new_params.append('{}: ${:0.2}d^2$'.format(i,float(i)/dim/dim))
		new_params.append('${:0.2f}d^2$'.format(float(i)/dim/dim))

	return new_params

def build_and_save_plot(df, filename, 
	legend_title = 'Number of parameters', 
	x_axis_label = "Gradient Descent Steps"):
	filenames, x_vec, y_vec = return_steps_and_loss_vecs(df)
	colormap = build_param_colormap(df)

	# style
	plt.style.use('seaborn-darkgrid')
	 
	# create a color palette
	palette = plt.get_cmap('Set1')

	for i, file_i in enumerate(filenames):
		n_params = pd.unique(df[df['filename'] == file_i]['number of time parameters'])
		n_params = n_params[0]
		plt.semilogy(x_vec[i], y_vec[i], marker='', color=palette(colormap[n_params]), linewidth=1.5, alpha=0.7, label=n_params)

	handles, labels = plt.gca().get_legend_handles_labels()
	by_label = OrderedDict(zip(labels, handles))
	labels, handles = zip(*sorted(zip(by_label.keys(), by_label.values()), key=lambda t: int(t[0])))
	print(labels)
	labels = format_param_str(labels, pd.unique(df[df['filename'] == file_i]['dimension of unitary matrix']))
	print(labels)
	plt.legend(handles, labels, bbox_to_anchor=(1.02,1), 
		title = legend_title)

	plt.subplots_adjust(right=0.75)

	plt.ylabel("Loss (log axis)")
	plt.xlabel(x_axis_label)

	fig = plt.gcf()
	fig.set_size_inches(5,4)

	plt.savefig("./figures/"+filename)	
	plt.close()



if __name__ == '__main__':

	# df = pd.read_csv('./csv_files/combined_csv/new_dataset_for_full_plots_full.csv')
	# print(df.columns)

	# build_and_save_plot(df, 'figure_full_unitary_expanded.pdf', '# params (2K)')

	df = pd.read_csv('./csv_files/combined_csv/all_runs_paper_full_2.csv')
	print(df.columns)

	build_and_save_plot(df, 'figure_d4_unitary.pdf', '# params (2K)')


	df = pd.read_csv('./csv_files/combined_csv/all_runs_paper_full_3_adam.csv')
	print(df.columns)

	build_and_save_plot(df[df['number of target parameters'] == 8], 
		'figure_d4_unitary_adam.pdf', 
		legend_title = '# params (2K)', 
		x_axis_label = 'Num. of Adam optimizer steps')
	build_and_save_plot(df[df['number of target parameters'] != 8], 
		'figure_full_unitary_adam.pdf',
		legend_title = '# params (2K)', 
		x_axis_label = 'Num. of Adam optimizer steps')


	# df = pd.read_csv('./csv_files/combined_csv/all_runs_paper_full_4.csv')
	# print(df.columns)

	# build_and_save_plot(df, 'figure_d2_unitary.pdf', '# params (2K)')


	# df = pd.read_csv('./csv_files/combined_csv/all_runs_paper_full_5.csv')
	# print(df.columns)

	# build_and_save_plot(df, 'figure_full_transition_unitary.eps', '# params (2K)')
