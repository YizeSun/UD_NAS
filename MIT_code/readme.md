# Learning Unitaries by Gradient Descent

Code for paper "Learning Unitaries by Gradient Descent", arXiv:2001.11897

## Getting Started

These instructions will allow you to run simulations as in our paper (https://arxiv.org/abs/2001.11897). Familiarity with Python and Pytorch is needed to run the code. Furthermore, access to a GPU will significantly speed up simulations.


### Prerequisites

The code runs on Python 3. We recommend installing Python using Anaconda to simplify the process. These packages must also be installed:

```
Pytorch (version 1.2.0 was used for simulations)
Numpy
Pandas
Matplotlib
Scipy

```

### Running the code

All simulations output csv files which contain the raw data for a simulation (e.g. the loss at each step). These csv files are later used to perform plotting.

Please follow the steps below to perform simulations:

1. Setup config.py
    1. If you have access to gpu set is_Cuda = True and choose gpuid (default is 0 for a computer with only one gpu)
    2. select dtype by setting dtype = torch.float64 or dtype = torch.float32 (other values not recommended)
2. Run desired simulation
	1. run_batch.py: performs a batch of simulations where each simulation learns a target unitary for given dimensions. If save_results set to True, then outputs will be saved as csv files to the csv_files directory.
	2. get_loss_grid.py: calculates 2D grid of loss values (see Fig. 4 in paper) and saves grid as csv file

To recreate plots, first csv files must be created using the run_batch.py file or get_loss_grid.py file. Then, for plotting simulations (not the loss grid), run the combine_csv.py file. This file combines all csv data in the csv_files folder into a single file. Two merged csv files are created: 

1. combined_csv_file_name controls the name of the csv file which contains summary data (e.g. number of steps, end loss)
2. full_csv_file_name conrols the name of the csv file which contains full data including every step (not just summary)

After running combine_csv.py it is recommended that the csv files in the csv_files folder be deleted or moved so that future csv files can be created and merged.

Various plotting files are included each beginning with "plot_". These take, as input, the combined csv files described above and output figures to the figures directory. The names of the combined csv files must be inputted into the plotting files.

Sample combined_csv file and figure are included in their respective folders.

## Authors

* **Bobak Kiani** - *MIT* - [github:bkiani](https://github.com/bkiani)
* **Reevu Maity** - *University of Oxford*
* **Seth Lloyd** - *MIT*

