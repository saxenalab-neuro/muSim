import configargparse

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
                        type=bool, 
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
                        type=bool, 
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
                        type=bool, 
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

    parser.add_argument('--load_saved_nets_for_training', 
                        type=bool, 
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

    parser.add_argument('--test_data_filename', 
                        type=str, 
                        default='test_data',
                        help='filename for saving the testing data')
                        
    parser.add_argument('--condition_selection_strategy', 
                        type=str, 
                        default='reward',
                        help='whether to select the next condition based on the corresponding average reward accumulated so far')

    return parser
