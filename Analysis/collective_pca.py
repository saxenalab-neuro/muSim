from sklearn.cross_decomposition import CCA
import numpy as np
import pickle
from scipy.io import loadmat
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from scipy.spatial import procrustes
from sklearn.decomposition import PCA

import sys
sys.path.insert(0, '../SAC/')
import kinematics_preprocessing_specs

#Load the test data of nusim

with open('../test_data/test_data.pkl', 'rb') as file:
    test_data = pickle.load(file)
    
print(test_data.keys())

#Get the timepoints of each condition per cycle
with open('../kinematics_data/kinematics.pkl', 'rb') as file:
    kin_train_test = pickle.load(file)
    
kin_train = kin_train_test['train']
kin_test = kin_train_test['test']

#First update the keys of self.kin_test
for cond in range(len(kin_test)):
    kin_test[len(kin_train) + cond] = kin_test.pop(cond)
    
kin = kin_train
kin.update(kin_test)

conds = [kin[cond].shape[-1] for cond in range(len(kin))]
total_conds = len(conds)

#Select the cycle for each condition (training conditions followed by testing): 0 for 1st cycle and so on
#The number of elements should be equal to num_train_conditions + num_test_conditions
cycles = [2, 2, 2, 2, 2, 2]

#Number of fixedsteps in the start of each condition
#Fix this {todo}: Get the values automatically from ..SAC/kinematics_preprocessing_specs.py
n_fixedsteps= kinematics_preprocessing_specs.n_fixedsteps

#Load the network activities
A_agent = []

for idx, cond_activity in test_data['rnn_activity'].items():
    act_agent = cond_activity
    act_agent = act_agent[n_fixedsteps + cycles[idx] * conds[idx] : n_fixedsteps + (cycles[idx]+1) * conds[idx]]
    print(act_agent.shape)
    A_agent.append(act_agent[:, :])

#Do the collective PCA for all speeds
nusim_pca = PCA(n_components= 3)

A_agent_c = A_agent
#concatenate the musim activity for all conditions
for i_cond in range(len(A_agent_c)):
    
    if i_cond == 0:
        nusim_activity_pca = A_agent_c[i_cond]
    else:
        nusim_activity_pca = np.concatenate((nusim_activity_pca, A_agent_c[i_cond]), axis=0)

nusim_activity_pca = nusim_pca.fit_transform(nusim_activity_pca)

#Plot the PCA of the activities
colors = plt.cm.ocean(np.linspace(0,1,8))
ax = plt.figure(dpi=100).add_subplot(projection='3d')

prev_cond = 0
for i_cond in range(len(A_agent_c)):
    ax.plot(nusim_activity_pca[prev_cond:prev_cond+A_agent_c[i_cond].shape[0],0], 
            nusim_activity_pca[prev_cond:prev_cond+A_agent_c[i_cond].shape[0], 1], 
            nusim_activity_pca[prev_cond:prev_cond+A_agent_c[i_cond].shape[0], 2], color= colors[i_cond])
    
    prev_cond += A_agent_c[i_cond].shape[0]

    
# Hide grid lines
ax.grid(False)
plt.grid(b=None)

# Hide axes ticks
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
plt.axis('off')

plt.show()