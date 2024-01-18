from sklearn.cross_decomposition import CCA
import numpy as np
from scipy.io import loadmat
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import pickle


#Load the nusim and experimental activities

#Load nusim for training and testing

with open("./Training/nusim_train.pkl", 'rb') as file:
    # Deserialize and retrieve the variable from the file
    nusim_train = pickle.load(file)

with open("./Testing/nusim_test.pkl", 'rb') as file:
    # Deserialize and retrieve the variable from the file
    nusim_test = pickle.load(file)

#Load experimental for training and testing

with open("./Training/exp_train.pkl", 'rb') as file:
    # Deserialize and retrieve the variable from the file
    exp_train = pickle.load(file)

with open("./Testing/exp_test.pkl", 'rb') as file:
    # Deserialize and retrieve the variable from the file
    exp_test = pickle.load(file)

# Now do the CCA for the training conditions and plot those one-by-one
for i_cond in range(len(nusim_train)+len(nusim_test)):
	
	if i_cond < len(nusim_train):
		A_exp = exp_train[i_cond]
		A_agent = nusim_train[i_cond]
	else:
		i_cond_test = i_cond - len(nusim_train)
		A_exp = exp_test[i_cond_test]
		A_agent = nusim_test[i_cond_test]

	#First filter the agent's activity with 20ms gaussian as done with experimental activity during preprocessing
	A_agent = gaussian_filter1d(A_agent.T, 20).T

	#Reduce the activity using PCA to the first 10 components
	PC_agent = PCA(n_components= 10)
	PC_exp = PCA(n_components= 10)
	#

	# A_agent = PC_agent.fit_transform(A_agent)
	A_exp = PC_exp.fit_transform(A_exp)
	A_agent = PC_agent.fit_transform(A_agent)

	#Do the CCA
	cca = CCA(n_components=10)
	U_c, V_c = cca.fit_transform(A_exp, A_agent)

	result = np.corrcoef(U_c[:,9], V_c[:,9])
	U_prime = cca.inverse_transform(V_c)

	plt.figure(figsize= (6, 6))

	for k in range(10):
	    if k==0:
	        plt.plot(A_exp[:,9-k]/np.max(A_exp[:,9-k]) + k*4, linewidth=1.5, c = 'k')
	        plt.plot(U_prime[:,9-k]/np.max(A_exp[:,9-k]) + k*4, linewidth= 1.5, c=(50/255, 205/255, 50/255), label= 'Network Reconstruction')
	    else:
	        plt.plot(A_exp[:, 9 - k]/np.max(A_exp[:,9-k]) + k * 4, linewidth=1.5, c='k')
	        plt.plot(U_prime[:, 9 - k]/np.max(A_exp[:,9-k]) + k * 4, linewidth=1.5, c=(50 / 255, 205 / 255, 50 / 255))

	plt.ylabel('Reconstructed M1 Population Activity', size=14)
	plt.xticks([0, 500], ['0', '0.5'], size= 14)
	plt.yticks([])
	# plt.legend()
	# plt.savefig('C:/Users/malma/Dropbox/NatureFigs2/Fig2/CCA_619.svg', format='svg', dpi=300, transparent= True)
	if i_cond < len(nusim_train):
		plt.title(f"Inverse CCA Train Condition {i_cond+1}")
	else:
		plt.title(f"Inverse CCA for Test Condition {i_cond_test+1}")
	plt.show()

	#Now plot the PCs on the same plot here
	ax = plt.figure(figsize= (6,6), dpi=100).add_subplot(projection='3d')
	ax.plot(A_exp[:,0], A_exp[:, 1], A_exp[:, 2], c = 'k')
	ax.plot(U_prime[:,0], U_prime[:, 1], U_prime[:, 2], c=(50/255, 205/255, 50/255))

	# Hide grid lines
	ax.grid(False)
	plt.grid(b=None)

	# Hide axes ticks
	ax.set_xticks([])
	ax.set_yticks([])
	ax.set_zticks([])
	plt.axis('off')

	if i_cond < len(nusim_train):
		plt.title(f"PC plot Train Condition {i_cond+1}")
	else:
		plt.title(f"PC plot for Test Condition {i_cond_test+1}")

	plt.show()
	# plt.savefig('C:/Users/malma/Dropbox/NatureFigs2/Fig2/PCA_619.svg', format='svg', dpi=300, transparent= True)

	sum = 0
	for k in range(3):
	    sum = sum + np.corrcoef(A_exp[:, k], U_prime[:, k])[0, 1]
	average = sum / 3;

	if i_cond < len(nusim_train):
		print(f"Correlation for Train Condition {i_cond+1}", average)
	else:
		print(f"Correlation for Test Condition {i_cond_test+1}", average)