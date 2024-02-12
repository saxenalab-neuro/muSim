from sklearn.cross_decomposition import CCA
import numpy as np
from scipy.io import loadmat
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score

#Load the experimental activities

exp_alt1 = loadmat('/Users/malmani/Downloads/mean_firing_rate_alt1.mat')['cell_mean_firing_rate'][0, 1]

exp_alt_slow = loadmat('/Users/malmani/Downloads/mean_firing_rate_alt_slow.mat')['cell_mean_firing_rate'][0, 1]

exp_alt_fast = loadmat('/Users/malmani/Downloads/mean_firing_rate_alt_fast.mat')['cell_mean_firing_rate'][0, 1]

exp_alt1= exp_alt1[:, 225:405].T
exp_alt_slow= exp_alt_slow[:, 255:475].T
exp_alt_fast= exp_alt_fast[:, 230:400].T

#Now load the lstm activities
#Make sure the dimensions are {#timepoints, #neurons]
agent_alt1= np.load()
agent_slow= np.load()
agent_fast= np.load()

#Now fit the linear regression models and save the prediction matrices.
A_exp = np.concatenate((exp_alt_slow, exp_alt_fast), axis=0)
A_agent = np.concatenate((agent_slow, agent_fast), axis= 0)

lr_drl = Ridge(alpha= 1.2).fit(A_agent, A_exp)

pred_alt1_drl = lr_drl.predict(agent_alt1)
print(r2_score(pred_alt1_drl, exp_alt1, multioutput= 'variance_weighted'))

#Now plot the predicted and actual neural activities
