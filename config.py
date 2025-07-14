import configargparse

def list_of_list_of_floats(x):
    temp = x.replace('[', '').replace(']', '').split(',')
    temp2 = [float(temp_e) for temp_e in temp]

    return temp2

def list_of_string_names(x):

    return x

def list_of_tuples_of_strings(x):

    temp = x.replace('[', '').replace(']', '').split(';')

    return temp

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def config_parser():
    parser = configargparse.ArgumentParser()

    parser.add_argument("--config", is_config_file=True, help="config file path")

    parser.add_argument('--model',
                        type=str, 
                        default="rnn",
                        help='rnn, gru')

    parser.add_argument('--gamma', 
                        type=float, 
                        default=0.99, 
                        help='discount factor for reward (default: 0.99)')

    parser.add_argument('--tau', 
                        type=float, 
                        default=0.005, 
                        help='target smoothing coefficient(τ) (default: 0.005)')

    parser.add_argument('--lr', 
                        type=float, 
                        default=0.0003, 
                        help='learning rate (default: 0.001)')

    parser.add_argument('--alpha', 
                        type=float, 
                        default=0.2, 
                        help='Temperature parameter α determines the relative importance of the entropy\
                                term against the reward (default: 0.2)')

    parser.add_argument('--automatic_entropy_tuning', 
                        type=boolean_string, 
                        default=True, 
                        help='Automaically adjust α (default: False)')

    parser.add_argument('--seed', 
                        type=int, 
                        default=123456, 
                        help='random seed (default: 123456)')

    parser.add_argument('--policy_batch_size', 
                        type=int, 
                        default=8, 
                        help='batch size (default: 8)')

    parser.add_argument('--hidden_size', 
                        type=int, 
                        default=256, 
                        help='hidden size (default: 1000)')

    parser.add_argument('--policy_replay_size', 
                        type=int, 
                        default=50000, 
                        help='size of replay buffer (default: 2800)')

    parser.add_argument('--multi_policy_loss', 
                        type=boolean_string, 
                        default=False, 
                        help='use additional policy losses')

    parser.add_argument('--batch_iters', 
                        type=int, 
                        default=1, 
                        help='iterations to apply update')

    parser.add_argument('--cuda', 
                        action="store_true",
                        help='run on CUDA (default: False)')

    parser.add_argument('--visualize', 
                        type=boolean_string, 
                        default=False,
                        help='visualize monkey/mouse')

    parser.add_argument('--root_dir', 
                        type=str, 
                        default='',
                        help='specify you root directory')

    parser.add_argument('--checkpoint_file', 
                        type=str, 
                        default='agent_networks',
                        help='specify the name of the file in which you would like to save model weights/training params (do not add extension). Also saves statistics file in root of project folder with same name')

    parser.add_argument('--checkpoint_folder', 
                        type=str, 
                        default= 'checkpoint',
                        help='specify the name of the folder in which you would like to save the checkpoint file')

    parser.add_argument('--statistics_folder', 
                        type=str, 
                        default= 'training_statistics',
                        help='specify the name of the folder in which you would like to save the training statistics')

    parser.add_argument('--total_episodes', 
                        type=int, 
                        default=5000000, 
                        help='total number of episodes')

    parser.add_argument('--save_iter', 
                        type=int, 
                        default=100, 
                        help='number of episodes until checkpoint is saved')

    parser.add_argument('--mode', 
                        type=str, 
                        default="train", 
                        help='select whether to train or test a model (train, test, SFE, sensory_pert, neural_pert, musculo_properties)')


    parser.add_argument('--verbose_training', 
                        type=boolean_string, 
                        default=False, 
                        help='Print statistics during training')

    parser.add_argument('--load_saved_nets_for_training', 
                        type=boolean_string, 
                        default=False,
                        help='select whether to train or test a model (train, test)')

    parser.add_argument('--musculoskeletal_model_path', 
                        type=str, 
                        default='musculoskeletal_model/musculoskeletal_model.xml',
                        help='path of musculoskeletal model')

    parser.add_argument('--initial_pose_path', 
                        type=str, 
                        default='initial_pose',
                        help='path of musculoskeletal model')

    parser.add_argument('--kinematics_path', 
                        type=str, 
                        default='kinematics_data',
                        help='path to kinematics data')

    parser.add_argument('--nusim_data_path', 
                        type=str, 
                        default='nusim_neural_data',
                        help='path to nusim neural data for training and testing')

    parser.add_argument('--stimulus_data_path', 
                        type=str, 
                        default='stimulus_data',
                        help='path to experimental stimulus data for training and testing')

    parser.add_argument('--test_data_filename', 
                        type=str, 
                        default='test_data',
                        help='filename for saving the testing data')
                        
    parser.add_argument('--condition_selection_strategy', 
                        type=str, 
                        default='reward',
                        help='whether to select the next condition based on the corresponding average reward accumulated so far')

    parser.add_argument('--sim_dt', 
                        type=int, 
                        default=0,
                        help='The timestep for the simulation: Keep 0 for default simulation timestep')

    parser.add_argument('--frame_repeat', 
                        type=int, 
                        default=5,
                        help='The frames/timepoints for which the same action should be repeated during training of the agent')

    parser.add_argument('--n_fixedsteps', 
                        type=int, 
                        default=25,
                        help='The target will remain at kinematic[timestep=0] for n_fixedsteps')

    parser.add_argument('--timestep_limit', 
                        type=int, 
                        default=1000,
                        help='Timestep limit is max number of timesteps after which the episode will terminate.')

    parser.add_argument('--trajectory_scaling', 
                        type= float, 
                        nargs= '+',
                        default= None,
                        help='Adjusts/scales the length of the trajectory')

    parser.add_argument('--center', 
                        type= list_of_list_of_floats,
                        nargs= '+',
                        default= None,
                        help='Adjusts the starting point of the kinematics trajectory')

    parser.add_argument('--stimulus_feedback', 
                        type= boolean_string,
                        default= False,
                        help='Experimental stimulus feedback to be included in the sensory feedback')

    parser.add_argument('--proprioceptive_feedback', 
                        type= boolean_string,
                        default= True,
                        help='Proprioceptive feedback consists of muscle lengths and velocities')

    parser.add_argument('--muscle_forces', 
                        type= boolean_string,
                        default= False,
                        help='Muscle forces consist of appled muscle forces')

    parser.add_argument('--joint_feedback', 
                        type= boolean_string,
                        default= False,
                        help='Joint feedback consists of joint positions and velocities')

    parser.add_argument('--visual_feedback', 
                        type= boolean_string,
                        default= False,
                        help='Visual feedback consists of x/y/z coordinates of the specified bodies in the model')

    parser.add_argument('--visual_feedback_bodies', 
                        type= list_of_string_names,
                        nargs= '*',
                        default= None,
                        help='Append the names musculo bodies from which visual feedback should be included')

    parser.add_argument('--visual_distance_bodies', 
                        type= list_of_tuples_of_strings,
                        nargs= '*',
                        default= None,
                        help='Specify the names of the bodies as tuples for which the visual distance should be included in the feedback')

    parser.add_argument('--visual_velocity', 
                        type= list_of_string_names,
                        nargs= '*',
                        default= None,
                        help='Specify the names of the bodies for which the visual velocity should be included in the feedback')

    parser.add_argument('--sensory_delay_timepoints', 
                        type= int,
                        default= 0,
                        help='Specify the delay in the sensory feedback in terms of the timepoints')

    parser.add_argument('--alpha_usim', 
                        type= float,
                        default= 0.1,
                        help='weighting with loss for enforcing simple neural dynamics for uSim/nuSim')

    parser.add_argument('--beta_usim', 
                        type= float,
                        default= 0.01,
                        help='weighting with loss for minimizing the neural activations for uSim/nuSim')

    parser.add_argument('--gamma_usim', 
                        type= float,
                        default= 0.001,
                        help='weighting with loss for minimizing the synaptic weights for uSim/nuSim')

    parser.add_argument('--zeta_nusim', 
                        type= float,
                        default= 0,
                        help='weighting with loss for nuSim constraining a sub-population of RNN units to experimentally recorded neurons for nuSim')


    return parser
