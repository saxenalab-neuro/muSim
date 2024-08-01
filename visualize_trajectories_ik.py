import ipdb
import config
import numpy as np
from SAC.IK_Framework_Mujoco import Muscle_Env
from SAC import kinematics_preprocessing_specs
from SAC.TR_Algorithm import TR_Algorithm
import pickle

### PARAMETERS ###
parser = config.config_parser()
args = parser.parse_args()

#Setup the mujoco_env with the given args
env = Muscle_Env(args.musculoskeletal_model_path[:-len('musculoskeletal_model.xml')] + 'musculo_targets.xml', 0, 0, args)

#Set the environment in the inital pose found by IK
initial_state = np.load(args.initial_pose_path + '/initial_qpos_opt.npy')
env.set_state(initial_state)

#get the initial musculo state
initial_state = env.get_musculo_state()

#Set the environment in condition 0 and timepoint 0
env.set_cond_to_simulate(0, 0)

#Define the objective function to be minimized for the IK optimization algorithm
def obj_func(state):

	#Set the env qpos to the state
	env.set_state_musculo(state)

	#Return the l2 norm of the resuling difference between the musculo bodies and targets

	return np.linalg.norm(env.get_obs_musculo_bodies() - env.get_obs_targets())

#Create a dict to save the sensory_feedback from IK
sensory_feedback_ik = {}

for cond in range(len(env.kin_to_sim)):
	print("Simulating CONDITION:", cond+1)
	#Set the environment in the inital pose found by IK
	initial_state = np.load(args.initial_pose_path + '/initial_qpos_opt.npy')
	env.set_state(initial_state)

	#Get the initial musculo state
	initial_state = env.get_musculo_state()
	env.set_cond_to_simulate(cond, 0)

	sensory_feedback_cond_ik = []
	for tpoint in range(env.kin_to_sim[cond].shape[-1]-1):
		s_final, s_final_musculo, loss_min, _, _ = TR_Algorithm(obj_func, initial_state, env)

		sensory_feedback_cond_ik.append(env._get_obs())
		
		#Update the initial state and env(condition, timepoint)

		initial_state = s_final_musculo
		env.set_cond_to_simulate(cond, tpoint+1)

	s_cond= np.array(sensory_feedback_cond_ik)
	sensory_feedback_ik[cond] = s_cond

print('Saving inverse kinematics file to: ' + args.test_data_filename + '/sensory_feedback_ik.pkl' )
pickle.dump(sensory_feedback_ik, open(args.test_data_filename + '/sensory_feedback_ik.pkl', "wb"))

