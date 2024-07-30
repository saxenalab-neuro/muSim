import numpy as np

#Neural perturbation when the mode in configs.py is "neural_pert"

#It should be a numpy array of the shape [timepoints, n_hidden_units (for uSim controller/policy RNN)]
neural_pert = np.ones((100, 256))

#Specify the weighting with various neural regularizations used in uSim/nuSim
#weighting with loss for simple dynamics
alpha = 0.1

#weighting with loss for minimizing the neural activations
beta = 0.01 

#weighting with loss for minimizing the synaptic weights
gamma = 0.001

#weighting with loss for nuSim constraining a sub-population of RNN units to experimentally recorded neurons
zeta = 0