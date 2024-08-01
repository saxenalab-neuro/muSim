import config
import numpy as np
from SAC.IK_Framework_Mujoco import Muscle_Env
from SAC import kinematics_preprocessing_specs
from SAC.TR_Algorithm import TR_Algorithm
import cma

### PARAMETERS ###
parser = config.config_parser()
args = parser.parse_args()

#Setup the mujoco_env with the given args
env = Muscle_Env(args.musculoskeletal_model_path[:-len('musculoskeletal_model.xml')] + 'musculo_targets.xml', 0, 0, args)

#Set condition 0 and timpoint 0 for finding the initial position
env.set_cond_to_simulate(0, 0)
initial_state = env.get_musculo_state()

#Define the objective function to be minimized for the IK optimization algorithm
def obj_func(state):

	#Set the env qpos to the state
	env.set_state_musculo(state)

	#Return the l2 norm of the resuling difference between the musculo bodies and targets

	return np.linalg.norm(env.get_obs_musculo_bodies() - env.get_obs_targets())


s_final, s_final_musculo, loss_min, cum_loss, success = TR_Algorithm(obj_func, initial_state, env)

print(f'success: {success}', f'final/min_loss:{loss_min}')

if success:
	print('Initial Pose found and saved')
	np.save(args.initial_pose_path + '/initial_qpos_opt.npy', s_final)