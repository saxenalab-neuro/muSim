import config
import numpy as np
from SAC.IK_Framework_Mujoco import Muscle_Env
from SAC import kinematics_preprocessing_specs
from SAC.TR_Algorithm import TR_Algorithm
import cma

### PARAMETERS ###
parser = config.config_parser()
args = parser.parse_args()

#Parameters for the CMA-ES algorithm
sigma= 0.5

### PARAMETERS ###
parser = config.config_parser()
args = parser.parse_args()

#Setup the mujoco_env with the given args
env = Muscle_Env(args.musculoskeletal_model_path[:-len('musculoskeletal_model.xml')] + 'musculo_targets.xml', 0, 0, args)

#Set condition 0 and timpoint 0 for finding the initial position
env.set_cond_to_simulate(0, 0)
initial_state = env.get_musculo_state()

# Define the objective function to be minimized for the IK optimization algorithm
def obj_func(state):

	#Set the env qpos to the state
	env.set_state_musculo(state)

	#Return the l2 norm of the resuling difference between the musculo bodies and targets

	return np.linalg.norm(env.get_obs_musculo_bodies() - env.get_obs_targets())


#Use the CMA-ES algorithm to find the initial position
es = cma.CMAEvolutionStrategy(len(initial_state) * [0], sigma)


while not es.stop():

	initial_states = es.ask()
	es.tell(initial_states, [TR_Algorithm(obj_func, initial_state, env)[3] for initial_state in initial_states])

	es.logger.add()
	es.disp()

es.result_pretty()
cma.plot()

#Saving the result using the 
print('Initial Pose found and saved using CMA-ES and Inverse Kinematics')

qpos_to_save = env.sim.data.qpos.flat.copy()
qpos_to_save[env.qpos_idx_musculo] = es.result.xbest

np.save(args.initial_pose_path + '/initial_qpos_opt.npy', qpos_to_save)