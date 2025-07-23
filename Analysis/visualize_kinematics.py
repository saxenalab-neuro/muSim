import numpy as np
import pickle
import matplotlib.pyplot as plt

from matplotlib.pyplot import figure
from sklearn.metrics import mean_squared_error as MSE

import sys
sys.path.insert(0, '../')
#sys.path.insert(0, '../SAC/')
import SAC.kinematics_preprocessing_specs

import config

from SAC.RL_Framework_Mujoco import *


def create_graphs(env, env_name, colors):
    #Load the test data of usim
    with open('../test_data/' + env_name + '/test_data.pkl', 'rb') as file:
        test_data = pickle.load(file)
        
    kin_train = env.kin_to_sim

    kin_agent = []

    for idx, cond_kin in test_data['kinematics_mbodies'].items():
        kin_agent.append(cond_kin[marker, :, :])

    kin_sim = []

    for idx, cond_kin in test_data['kinematics_mtargets'].items():
        kin_sim.append(cond_kin[marker, :, :])

    for cond in range(len(kin_agent)):
        figure(figsize=(20, 5), dpi=80)

        all_ranges = []
        for coord in range(kin_agent[cond].shape[-1]):
            values = np.concatenate([kin_sim[cond][:, coord], kin_agent[cond][:, coord]])
            all_ranges.append(np.max(values) - np.min(values))
        max_range = max(all_ranges)
        
        MSE_cond = 0
        for coord in range(kin_agent[cond].shape[-1]):
            
            plt.subplot(1, kin_agent[cond].shape[-1], coord+1)
            plt.plot(kin_sim[cond][:, coord], '--', label= 'Target')
            plt.plot(kin_agent[cond][:, coord], label= 'Achieved')

            sim_vals = kin_sim[cond][:, coord]
            agent_vals = kin_agent[cond][:, coord]
            
            combined = np.concatenate([sim_vals, agent_vals])
            mid = (np.max(combined) + np.min(combined)) / 2
            plt.ylim(mid - max_range / 2, mid + max_range / 2)
            
            MSE_coord = MSE(kin_sim[cond][:, coord], kin_agent[cond][:, coord])
            
            plt.title(env_name + f' Kinematics for COND{cond+1} MARKER{marker} \n MSE{MSE_coord}')
            
            #omit the z coord for MSE calculations
            if coord != 1:
                MSE_cond += MSE_coord
        
        print(f'MSE for COND {cond}: ',  MSE_cond/2)
        plt.legend()
        plt.savefig('./graphs/' + env_name + f'/MSE for COND {cond}')


    #Now plot the kinematics for all training conditions together

    plt.figure(figsize= (10, 10))

    for cond in range(len(kin_train)):
        for coord in range((kin_agent[cond].shape[-1])):

            if coord != 1:

                kin_agent_coord = kin_agent[cond][:, coord]
                kin_sim_coord = kin_sim[cond][:, coord]

                # Normalize for pretty display
                norm_factor = np.min(kin_sim_coord) + np.ptp(kin_sim_coord) / 2

                kin_agent_coord = (kin_agent_coord - norm_factor)
                kin_sim_coord = (kin_sim_coord - norm_factor)

                if coord == 0:
                    col = (0 / 255, 191 / 255, 255 / 255)
                else:
                    col = (205 / 255, 133 / 255, 63 / 255)

                plt.plot(kin_agent_coord - cond*0.1, '-', linewidth= 1.5, c=col)
                plt.plot(kin_sim_coord - cond*0.1, '--', linewidth= 1.5, c=col)

    plt.xticks([])
    plt.yticks([])

    plt.title(env_name + ' Kinematics Multiple Graphs')
    plt.legend(['Achieved', 'Target'])
    plt.savefig('./graphs/' + env_name + '/Kinematics Multiple Graphs')

    #Now plot the kinematics for all training conditions together

    plt.figure(figsize= (10, 10))

    for cond in range(len(kin_train)):
        kin_agent_coord = kin_agent[cond]
        plt.plot(kin_agent_coord[:, 0], kin_agent_coord[:, 2], '-', linewidth=3, c=colors[cond][0])
        kin_sim_coord = kin_sim[cond]
        plt.plot(kin_sim_coord[:, 0], kin_sim_coord[:, 2], '--', linewidth=3, c=colors[cond][1])

    plt.xticks([])
    plt.yticks([])

    plt.title(env_name + ' Kinematics Single Graph')
    plt.savefig('./graphs/' + env_name + '/Kinematics Single Graph')

    plt.close()


if __name__ == '__main__':
    parser = config.config_parser()
    args, unknown = parser.parse_known_args()

    #Select the target/marker for which to visualize the kinematics
    marker = 0

    env_names = ['Reach', 'CurvedReachClk', 'CurvedReachCClk', 'Sinusoid', 'SinusoidInv', \
          'FullReach', 'Figure8', 'Figure8Inv']

    envs = [DlyReach, DlyCurvedReachClk, DlyCurvedReachCClk, DlySinusoid, DlySinusoidInv, \
                DlyFullReach, DlyFigure8, DlyFigure8Inv]

    env_names = ['CircleClk', 'CircleCClk']

    envs = [DlyCircleClk, DlyCircleCClk]

    color_pairs = [
    ((0.121, 0.466, 0.705), (0.4, 0.65, 0.85)),   # blues
    ((1.0, 0.498, 0.054), (1.0, 0.65, 0.3)),      # oranges
    ((0.172, 0.627, 0.172), (0.4, 0.8, 0.4)),     # greens
    ((0.839, 0.152, 0.156), (0.9, 0.4, 0.4)),     # reds
    ((0.580, 0.404, 0.741), (0.7, 0.6, 0.9)),     # purples
    ((0.549, 0.337, 0.294), (0.7, 0.5, 0.4)),     # browns
    ((0.890, 0.467, 0.761), (0.95, 0.65, 0.85)),  # pinks
    ((0.498, 0.498, 0.498), (0.7, 0.7, 0.7)),     # grays
    ]

    assert len(env_names) == len(envs)

    args.initial_pose_path = '../' + args.initial_pose_path
    
    for i in range(len(env_names)):
        
        env = envs[i]('../' + args.musculoskeletal_model_path[:-len('musculoskeletal_model.xml')] + 'musculo_targets.xml', 1, args)
        create_graphs(env, env_names[i], color_pairs)