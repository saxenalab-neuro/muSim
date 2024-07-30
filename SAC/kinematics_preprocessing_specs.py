from itertools import compress
import numpy as np
import pickle
import os

import sys
sys.path.insert(0, '../')
import config

parser = config.config_parser()
args, unknown = parser.parse_known_args()

#Kinematics preprocessing for simulation
#Adjustment instructions:

#The timestep for the simulation: Keep 0 for default simulation timestep
sim_dt = 0   # in seconds

#The frames/timepoints for which the same action should be repeated during training of the agent
frame_repeat = 5

#Number of fixedsteps in the beginning of the simulation. The target will remain at kinematic[timestep0] for n_fixedsteps
#If a good initial position is found using CMA-ES / TR Optimization, n_fixedsteps = 25 is a good estimate. Otherwise increase
#if the starting reward does not increase with the training iterations. 
n_fixedsteps = 25

#Timestep limit is max number of timesteps after which the episode will terminate.
#Multiple cycles of the same condition will be simulated if the timestep_limit > number of timsteps for that condition.
timestep_limit = (1319 * 5) + n_fixedsteps

#Adjusts/scales the length of the trajectory
#Should be the same as num_targets
trajectory_scaling = [1/0.038]
radius = trajectory_scaling

#Adjusts the starting point of the kinematics trajectory
#Should be the same as num_targets, num_coords=3
center = [[0.06, 0.083, 0]]

###Please do not change this code###
### -----------------------------------------------###

#Load the train and test kinematics
#Get the timepoints of each condition per cycle

#Check if the file is being called from main.py or Jupyter notebooks
if os.path.isfile(args.kinematics_path + '/kinematics.pkl'):
	with open(args.kinematics_path + '/kinematics.pkl', 'rb') as f:
		kin_train_test = pickle.load(f)

else:
	with open('../' + args.kinematics_path + '/kinematics.pkl', 'rb') as f:
		kin_train_test = pickle.load(f)


kin_train = kin_train_test['train']
kin_test = kin_train_test['test']

#Load a random condition
cond2load = np.random.randint(0, len(kin_train))
kin2load = kin_train[cond2load] #[num_targets, num_coords, timepoints]

xyz_target = []
for target in range(kin2load.shape[0]):
	xyz_coord = []
	for coord in range(kin2load.shape[1]):
		xyz_coord.append(int(np.logical_not(np.isnan(kin2load[target, coord, :]).any())))

	xyz_target.append(xyz_coord)


#Targets are as specified in kin_train.pkl, kin_test.pkl
#Specify the musculo bodies in order that have to track the targets as specified in the given kinematics pkl file.
#For example, here the musculo body 'hand' has to track the first 'target' kinematics.
musculo_tracking = kin_train_test['marker_names']


#Specify the xyz coordinates of the target movement
# x: -->, y: ↑, z: out of page
#Set True/False for x, y and z coords respectively
#Defines the num_coords
#Should be num_targets, num_coords
#Provide a NaN option in the kinematics path file
xyz_target = xyz_target
num_targets = len(xyz_target)

musculo_target_joints = []
for target_append in range(num_targets):
	musculo_tracking[target_append] = (musculo_tracking[target_append], f'target{target_append}')

	#The joints associated with the target
	musculo_target_joints_t = [f'box:x{target_append}', f'box:y{target_append}', f'box:z{target_append}']
	musculo_target_joints_t= list(compress(musculo_target_joints_t, xyz_target[target_append]))
	musculo_target_joints = [*musculo_target_joints, *musculo_target_joints_t]

### -----------------------------------------------####