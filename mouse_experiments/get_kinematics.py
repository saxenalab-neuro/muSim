import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from scipy import signal
import scipy.io
import argparse
from scipy.interpolate import Akima1DInterpolator

plt.rcParams.update({'font.size': 14})

def preprocess(cycles=1):

    ########################### Data_Fast ###############################
    mat = scipy.io.loadmat('data/kinematics_session_mean_alt_fast.mat')
    data = np.array(mat['kinematics_session_mean'][2])
    data_fast_orig = data[231:401:1] * -1
    data_fast_orig = [-13.45250312, *data_fast_orig[8:-1]]
    data_fast = [*data_fast_orig] * cycles
    if cycles > 1:
        # This needs to be done for smooth kinematics with cycles since they end at arbitrary points
        x = np.arange(0, len(data_fast))
        cs = Akima1DInterpolator(x, data_fast)
        # end point of kinematics and start point of next cycle
        x_interp = np.linspace(len(data_fast_orig)-1, len(data_fast_orig), 14)
        y_interp = cs(x_interp)
        # Get the new interpolated kinematics without repeating points
        data_fast = [*data_fast_orig, *y_interp[2:-2]] * cycles
        np.save('mouse_experiments/data/interp_fast', data_fast)

    # Data must start and end at same spot or there is jump
    ########################### Data_Slow ###############################
    mat = scipy.io.loadmat('data/kinematics_session_mean_alt_slow.mat')
    data = np.array(mat['kinematics_session_mean'][2])
    data_slow_orig = data[256:476:1] * -1
    data_slow_orig = [*data_slow_orig[:-6]]
    data_slow = [*data_slow_orig] * cycles
    if cycles > 1:
        x = np.arange(0, len(data_slow))
        cs = Akima1DInterpolator(x, data_slow)
        x_interp = np.linspace(len(data_slow_orig)-1, len(data_slow_orig), 5)
        y_interp = cs(x_interp)
        data_slow = [*data_slow_orig, *y_interp[1:-1]] * cycles
        np.save('mouse_experiments/data/interp_slow', data_slow)

    ############################ Data_1 ##############################
    mat = scipy.io.loadmat('data/kinematics_session_mean_alt1.mat')
    data = np.array(mat['kinematics_session_mean'][2])
    data_1_orig = data[226:406:1] * -1
    data_1_orig = [-13.45250312, *data_1_orig[4:-3]]
    data_1 = [*data_1_orig] * cycles
    if cycles > 1:
        x = np.arange(0, len(data_1))
        cs = Akima1DInterpolator(x, data_1)
        x_interp = np.linspace(len(data_1_orig)-1, len(data_1_orig), 3)
        y_interp = cs(x_interp)
        data_1 = [*data_1_orig, *y_interp[1:-1]] * cycles
        np.save('mouse_experiments/data/interp_1', data_1)

    return data_fast, data_slow, data_1

def main():

    parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
    parser.add_argument('--plot', type=str, default="kinematics",
                        help='kinematics, sim_x, sim_y, med')
    parser.add_argument('--cycles', type=int, default=1, metavar='N',
                        help='Number of times to cycle the kinematics (Default: 1)')

    args = parser.parse_args()

    data_1_kinematics = np.load('mouse_experiments/data/mouse_1.npy')
    data_fast_kinematics = np.load('mouse_experiments/data/mouse_fast.npy')
    data_slow_kinematics = np.load('mouse_experiments/data/mouse_slow.npy')

    data_fast, data_slow, data_1 = preprocess(args.cycles)

    scale = 21
    offset = -.713

    data_fast = np.array(data_fast) / scale - offset
    data_slow = np.array(data_slow) / scale - offset
    data_1 = np.array(data_1) / scale - offset

    fast_mse = (np.sum((data_fast - data_fast_kinematics)**2)) / 163
    slow_mse = (np.sum((data_slow - data_slow_kinematics)**2)) / 220
    med_mse = (np.sum((data_1 - data_1_kinematics)**2)) / 177

    print('Difference from target trajectory fast (MSE): {}'.format(fast_mse))
    print('Difference from target trajectory slow (MSE): {}'.format(slow_mse))
    print('Difference from target trajectory med (MSE): {}'.format(med_mse))

    # Plot the kinematics (currently for simulated)
    plt.plot(data_fast + .02, label='Experimental', linewidth=4, color='orange')
    plt.plot(data_fast_kinematics + .02, label='Model Output', linewidth=4, linestyle='dashed', color='black')

    plt.plot(data_1, linewidth=4, color='orange')
    plt.plot(data_1_kinematics, linewidth=4, color='black', linestyle='dashed')

    plt.plot(data_slow - .02, linewidth=4, color='orange')
    plt.plot(data_slow_kinematics - .02, linewidth=4, linestyle='dashed', color='black')

    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Position')
    plt.yticks([])
    plt.savefig('mouse_experiments/data/data_all_kinematics_plot.png')
    plt.show()

if __name__ == '__main__':
    main()