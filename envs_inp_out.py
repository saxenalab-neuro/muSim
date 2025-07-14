import sys
from pathlib import Path

#project_root = Path(__file__).resolve().parents[1]
#sys.path.insert(0, str(project_root))


import warnings
#warnings.filterwarnings("ignore")

import torch as th
import os
from SAC.RL_Framework_Mujoco import DlyReach, DlyCurvedReachClk, DlyCurvedReachCClk, DlySinusoid, DlySinusoidInv
from SAC.RL_Framework_Mujoco import DlyFullReach, DlyCircleClk, DlyCircleCClk, DlyFigure8, DlyFigure8Inv
import matplotlib.pyplot as plt
import numpy as np
import config
import pickle
#import dPCA
#from dPCA import dPCA
import tqdm as tqdm
import itertools
from sklearn.decomposition import PCA
#from losses import l1_dist
import scipy
import matplotlib.patches as mpatches
import matplotlib as mpl
from itertools import product

env_dict = {
    "DlyHalfReach": DlyReach,
    "DlyHalfCircleClk": DlyCurvedReachClk,
    "DlyHalfCircleCClk": DlyCurvedReachCClk,
    "DlySinusoid": DlySinusoid,
    "DlySinusoidInv": DlySinusoidInv,
    "DlyFullReach": DlyFullReach,
    "DlyFullCircleClk": DlyCircleClk,
    "DlyFullCircleCClk": DlyCircleCClk,
    "DlyFigure8": DlyFigure8,
    "DlyFigure8Inv": DlyFigure8Inv
}

plt.rcParams.update({'font.size': 18})  # Sets default font size for all text

def create_dir(save_path):
    # Check if the directory exists, and create it if it doesn't
    if not os.path.exists(save_path):
        os.makedirs(save_path)

def save_fig(save_path, eps=False):
    # Tell matplotlib to embed fonts as text, not outlines
    mpl.rcParams['pdf.fonttype'] = 42  # 42 = TrueType (editable in Illustrator)
    mpl.rcParams['ps.fonttype'] = 42
    # Simple function to save figure while creating dir and closing
    dir = os.path.dirname(save_path)
    create_dir(dir)
    plt.tight_layout()
    if eps:
        plt.savefig(save_path + ".pdf", format="pdf")
    else:
        plt.savefig(save_path + ".png")
    plt.close()


def plot_task_trajectories(args):
    """ This function will simply plot the target at each timestep for different orientations of the task
        This is not for kinematics

    Args:
        config_path (_type_): _description_
        model_path (_type_): _description_
        model_file (_type_): _description_
        exp_path (_type_): _description_
    """
    exp_path = f"Analysis/results/trajectories"

    create_dir(exp_path)

    for env in env_dict:
        for speed in range(8):

            options = {"batch_size": 8, "reach_conds": th.arange(0, 32, 4), "speed_cond": speed}

            #effector = mn.effector.RigidTendonArm26(mn.muscle.MujocoHillMuscle())
            cur_env = env_dict[env](args.musculoskeletal_model_path[:-len('musculoskeletal_model.xml')] + 'musculo_targets.xml', 1, args)
            obs = cur_env.reset(cond_to_select=speed % cur_env.n_exp_conds)#=options) # fix this at some point. I'm not sure what options or cond_to_select is

            # Get kinematics and activity in a center out setting
            # On random and delay
            colors = plt.cm.inferno(np.linspace(0, 1, cur_env.traj.shape[1]))

            for i, tg in enumerate(cur_env.traj):
                plt.scatter(tg[:, 0], tg[:, 1], s=10, color=colors)
                plt.scatter(tg[0, 0], tg[0, 1], s=150, marker='x', color="black")
                plt.scatter(tg[-1, 0], tg[-1, 1], s=150, marker='^', color="black")
            save_fig(os.path.join(exp_path, f"{env}_speed{speed}_tg_trajectory.png"))


def plot_task_input_output(model_name, args):
    """ This function will simply plot the target at each timestep for different orientations of the task
        This is not for kinematics

    Args:
        config_path (_type_): _description_
        model_path (_type_): _description_
        model_file (_type_): _description_
        exp_path (_type_): _description_
    """
    model_path = f"checkpoints/{model_name}"
    model_file = f"{model_name}.pth"
    exp_path = f"results/input"

    create_dir(exp_path)

    for env in env_dict:

        options = {"batch_size": 8, "reach_conds": th.arange(0, 32, 4), "speed_cond": 5, "delay_cond": 0}

        #effector = mn.effector.RigidTendonArm26(mn.muscle.MujocoHillMuscle())
        cur_env = env_dict[env](args.musculoskeletal_model_path[:-len('musculoskeletal_model.xml')] + 'musculo_targets.xml', 1, args)

        obs = cur_env.reset(0)

        for batch in range(options["batch_size"]):
            fig, ax = plt.subplots(5, 1)
            fig.set_size_inches(3, 6)
            plt.rc('font', size=6)

            delay = cur_env.delay_time
            movement = delay + cur_env.movement_time
            hold = movement + cur_env.hold_time

            ax[0].imshow(th.tensor(cur_env.rule_input).unsqueeze(0).repeat(cur_env._max_episode_steps, 1).T, vmin=-1, vmax=1,
                         cmap="seismic", aspect="auto")
            # Remove top and right only (common for minimalist style)
            ax[0].spines['top'].set_visible(False)
            ax[0].spines['right'].set_visible(False)
            ax[0].spines['bottom'].set_visible(False)
            ax[0].set_xticks([])
            ax[0].set_title("Rule Input")
            ax[0].axvline(delay, color="grey", linestyle="dashed")
            ax[0].axvline(movement, color="grey", linestyle="dashed")
            ax[0].axvline(hold, color="grey", linestyle="dashed")

            ax[1].plot([cur_env.speed_scalar] * cur_env._max_episode_steps, color="blue")
            ax[1].spines['top'].set_visible(False)
            ax[1].spines['right'].set_visible(False)
            ax[1].spines['bottom'].set_visible(False)
            ax[1].set_xticks([])
            ax[1].set_title("Speed Scalar")
            ax[1].axvline(delay, color="grey", linestyle="dashed")
            ax[1].axvline(movement, color="grey", linestyle="dashed")
            ax[1].axvline(hold, color="grey", linestyle="dashed")

            ax[2].plot(cur_env.go_cue, color="blue")
            ax[2].spines['top'].set_visible(False)
            ax[2].spines['right'].set_visible(False)
            ax[2].spines['bottom'].set_visible(False)
            ax[2].set_xticks([])
            ax[2].set_title("Go Cue")
            ax[2].axvline(delay, color="grey", linestyle="dashed")
            ax[2].axvline(movement, color="grey", linestyle="dashed")
            ax[2].axvline(hold, color="grey", linestyle="dashed")

            #ax[3].imshow(obs[78:84], vmin=-1, vmax=1, cmap="seismic", aspect="auto")
            ax[3].spines['top'].set_visible(False)
            ax[3].spines['right'].set_visible(False)
            ax[3].spines['bottom'].set_visible(False)
            ax[3].set_xticks([])
            ax[3].set_title("Visual Input")
            ax[3].axvline(delay, color="grey", linestyle="dashed")
            ax[3].axvline(movement, color="grey", linestyle="dashed")
            ax[3].axvline(hold, color="grey", linestyle="dashed")

            ax[4].plot(cur_env.traj[batch, :, 0]) # , vmin=-1, vmax=1, cmap="seismic", aspect="auto"
            ax[4].plot(cur_env.traj[batch, :, 1])
            ax[4].spines['top'].set_visible(False)
            ax[4].spines['right'].set_visible(False)
            ax[4].spines['bottom'].set_visible(False)
            ax[4].set_xticks([])
            ax[4].set_title("Tg Output (Only Movement Epoch)")
            ax[4].axvline(delay, color="grey", linestyle="dashed")
            ax[4].axvline(movement, color="grey", linestyle="dashed")
            ax[4].axvline(hold, color="grey", linestyle="dashed")

            save_fig(os.path.join(exp_path, f"{env}_input_orientation{batch}"), eps=True)


if __name__ == '__main__':

    ### PARAMETERS ###
    parser = config.config_parser()
    args = parser.parse_args()
    args.mode = 'test'


    #plot_task_trajectories(args)

    plot_task_input_output(args.checkpoint_file, args)