from sklearn.cross_decomposition import CCA
import numpy as np
from scipy.io import loadmat
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter1d
import argparse

#Load the experimental activities
def main():

    parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
    parser.add_argument('--speed', default="fast",
                        help='fast, slow, med')
    args = parser.parse_args()

    exp_alt1 = loadmat('data/mean_firing_rate_alt1.mat')['cell_mean_firing_rate'][0, 1]
    exp_alt_slow = loadmat('data/mean_firing_rate_alt_slow.mat')['cell_mean_firing_rate'][0, 1]
    exp_alt_fast = loadmat('data/mean_firing_rate_alt_fast.mat')['cell_mean_firing_rate'][0, 1]

    exp_alt1= exp_alt1[:, 230:406].T
    exp_alt_slow= exp_alt_slow[:, 256:476].T
    exp_alt_fast= exp_alt_fast[:, 239:401].T

    #Now load the lstm activities
    #dim= [#timepoints, neurons]
    agent_alt1 = np.load('mouse_experiments/data/mouse_1_activity.npy')
    agent_alt1 = agent_alt1[1:]

    agent_slow = np.load('mouse_experiments/data/mouse_slow_activity.npy')

    agent_fast = np.load('mouse_experiments/data/mouse_fast_activity.npy')
    agent_fast = agent_fast[1:]

    # Test the magnitudes between the one that works well and the one that doesnt

    #Now do the CCA
    if args.speed == 'med':
        A_exp = exp_alt1
        A_agent = agent_alt1
    elif args.speed == 'fast':
        A_exp = exp_alt_fast
        A_agent = agent_fast
    elif args.speed == 'slow':
        A_exp = exp_alt_slow
        A_agent = agent_slow


    PC_agent = PCA(n_components= 10)
    PC_exp = PCA(n_components= 10)

    A_agent = gaussian_filter1d(A_agent.T, 20)
    A_agent = A_agent.T
    A_agent = PC_agent.fit_transform(A_agent)

    A_exp = PC_exp.fit_transform(A_exp)

    cca = CCA(n_components=10)
    U_c, V_c = cca.fit_transform(A_exp, A_agent)

    # print(U_c.shape)
    # result = np.corrcoef(U_c[:,9], V_c[:,9])#.diagonal(offset=1)
    U_prime = cca.inverse_transform(V_c)
    plt.figure(figsize= (6, 6))

    for k in range(10):
        if k==0:
            plt.plot(U_prime[:,9-k]/np.max(A_exp[:,9-k]) + k*4, linewidth= 1.5, c=(50/255, 205/255, 50/255), label= 'Network Reconstruction')
            plt.plot(A_exp[:,9-k]/np.max(A_exp[:,9-k]) + k*4, linewidth=1.5, c = 'k')
        else:
            plt.plot(U_prime[:, 9 - k]/np.max(A_exp[:,9-k]) + k * 4, linewidth=1.5, c=(50 / 255, 205 / 255, 50 / 255))
            plt.plot(A_exp[:, 9 - k]/np.max(A_exp[:,9-k]) + k * 4, linewidth=1.5, c='k')

    if args.speed == 'med':

        plt.ylabel('Reconstructed M1 Population Activity', fontsize=16)
        plt.xticks([0, 163], ['0', '0.5'], size= 14)
        plt.yticks([])
        plt.savefig('mouse_experiments/data/cca_1.png')

    elif args.speed == 'fast':

        plt.ylabel('Reconstructed M1 Population Activity', fontsize=16)
        plt.xticks([0, 177], ['0', '0.5'], size= 14)
        plt.yticks([])
        plt.savefig('mouse_experiments/data/cca_fast.png')

    elif args.speed == 'slow':

        plt.ylabel('Reconstructed M1 Population Activity', fontsize=16)
        plt.xticks([0, 220], ['0', '0.5'], size= 14)
        plt.yticks([])
        plt.savefig('mouse_experiments/data/cca_slow.png')

    print('Now printing the correlations')

    print(np.corrcoef(A_exp[:,0], U_prime[:,0]))
    print(np.corrcoef(A_exp[:,1], U_prime[:,1]))
    print(np.corrcoef(A_exp[:,2], U_prime[:,2]))
    print(np.corrcoef(A_exp[:,3], U_prime[:,3]))
    print(np.corrcoef(A_exp[:,4], U_prime[:,4]))
    print(np.corrcoef(A_exp[:,5], U_prime[:,5]))
    print(np.corrcoef(A_exp[:,6], U_prime[:,6]))
    print(np.corrcoef(A_exp[:,7], U_prime[:,7]))
    print(np.corrcoef(A_exp[:,8], U_prime[:,8]))
    print(np.corrcoef(A_exp[:,9], U_prime[:,9]))

    sum = 0
    for k in range(10):
        sum = sum + np.corrcoef(A_exp[:, k], U_prime[:, k])[0, 1]
    average = sum / 10

    print(average)

if __name__ == '__main__':
    main()