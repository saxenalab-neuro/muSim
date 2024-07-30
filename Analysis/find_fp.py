'''
examples/torch/run_FlipFlop.py
Written for Python 3.8.17 and Pytorch 2.0.1
@ Matt Golub, June 2023
Please direct correspondence to mgolub@cs.washington.edu
'''

import pdb
import sys
import numpy as np
import torch
import pickle


PATH_TO_FIXED_POINT_FINDER = './fixed-point-finder/'
PATH_TO_HELPER = './fixed-point-finder/examples/helper/'
PATH_TO_TORCH = './fixed-point-finder/examples/torch/'
PATH_TO_SAC = '../SAC/'
sys.path.insert(0, PATH_TO_FIXED_POINT_FINDER)
sys.path.insert(0, PATH_TO_HELPER)
sys.path.insert(0, PATH_TO_TORCH)
sys.path.insert(0, PATH_TO_SAC)

from FlipFlop import FlipFlop
from FixedPointFinderTorch import FixedPointFinderTorch as FixedPointFinder
from FlipFlopData import FlipFlopData
from plot_utils import plot_fps


def find_fixed_points(model, rnn_trajectories, rnn_input):
	''' Find, analyze, and visualize the fixed points of the trained RNN.

	Args:
		model: 

			Trained RNN model, as returned by uSim training.

		valid_predictions: dict.

			Model trajectories for training and testing conditions.

	Returns:
		None.
	'''

	NOISE_SCALE = 0.5 # Standard deviation of noise added to initial states
	# N_INITS = 1024*10
	N_INITS = rnn_trajectories.shape[0] * rnn_trajectories.shape[1] # The number of initial states to provide

	n_hidden_units = rnn_trajectories.shape[2]

	'''Fixed point finder hyperparameters. See FixedPointFinder.py for detailed
	descriptions of available hyperparameters.'''
	fpf_hps = {
		'max_iters': 10000,
		'lr_init': 1.,
		'outlier_distance_scale': 10.0,
		'verbose': True, 
		'super_verbose': True}

	# Setup the fixed point finder
	fpf = FixedPointFinder(model, **fpf_hps)

	'''Draw random, noise corrupted samples of those state trajectories
	to use as initial states for the fixed point optimizations.'''
	initial_states = fpf.sample_states(rnn_trajectories,
		n_inits=N_INITS,
		noise_scale=NOISE_SCALE)

	# Study the system in the absence of input pulses (e.g., all inputs are 0)
	inputs = np.zeros([1, n_hidden_units])
	# inputs = rnn_input.reshape(-1, n_hidden_units)

	# Run the fixed point finder
	unique_fps, all_fps = fpf.find_fixed_points(initial_states, inputs)

	# Visualize identified fixed points with overlaid RNN state trajectories
	# All visualized in the 3D PCA space fit the the example RNN states.
	fig = plot_fps(unique_fps, rnn_trajectories,
		plot_batch_idx=list(range(6)),
		plot_start_time=0)

def main():

	# Step 1: Load the uSim RNN model, RNN trajectories and RNN input
	PATH_TO_MODEL = '../checkpoint/actor_rnn_best_fpf.pth' 
	actor_rnn = torch.load(PATH_TO_MODEL)
	model = actor_rnn

	#Load the test data
	with open('../test_data/test_data.pkl', 'rb') as file:
		test_data = pickle.load(file)

	#Get the uSim RNN hidden trajectories and inputs
	rnn_trajectories = []
	for cond in range(len(test_data['rnn_activity'])):
		rnn_trajectories.append(test_data['rnn_activity'][cond])

	rnn_trajectories = np.array(rnn_trajectories)  #[n_conds, n_timepoints, n_hidden_units]

	rnn_input = []
	for cond in range(len(test_data['rnn_input_fp'])):
		rnn_input.append(test_data['rnn_input_fp'][cond])

	rnn_input = np.array(rnn_input) #[n_conds, n_timepoints, n_hidden_units]

	# STEP 2: Find, analyze, and visualize the fixed points of the trained RNN
	find_fixed_points(model, rnn_trajectories, rnn_input)

	print('Entering debug mode to allow interaction with objects and figures.')
	print('You should see a figure with:')

	pdb.set_trace()

if __name__ == '__main__':
	main()