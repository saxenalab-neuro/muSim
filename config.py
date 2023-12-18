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
                        default=False, 
                        help='Automaically adjust α (default: False)')

    parser.add_argument('--seed', 
                        type=int, 
                        default=123456, 
                        help='random seed (default: 123456)')

    parser.add_argument('--policy_batch_size', 
                        type=int, 
                        default=8, 
                        help='batch size (default: 6)')

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
                        help='visualize mouse')

    parser.add_argument('--root_dir', 
                        type=str, 
                        default='',
                        help='specify you root directory')

    parser.add_argument('--checkpoint_file', 
                        type=str, 
                        default='',
                        help='specify the name of the file in which you would like to save model weights/training params (do not add extension). Also saves statistics file in root of project folder with same name')

    parser.add_argument('--checkpoint_folder', 
                        type=str, 
                        help='specify the name of the folder in which you would like to save the checkpoint file')

    parser.add_argument('--total_episodes', 
                        type=int, 
                        default=5000000, 
                        help='total number of episodes')

    parser.add_argument('--save_iter', 
                        type=int, 
                        default=1000, 
                        help='number of episodes until checkpoint is saved')

    parser.add_argument('--test', 
                        type=bool, 
                        default=False, 
                        help='test a model')

    parser.add_argument('--muscle_path', 
                        type=str, 
                        default='monkey/monkeyArm_current_scaled.xml',
                        help='path of musculoskeletal model')

    parser.add_argument('--muscle_params_path', 
                        type=str, 
                        default='monkey/params_monkey.pckl',
                        help='path of musculoskeletal model parameters')

    parser.add_argument('--kinematics_path', 
                        type=str, 
                        default='monkey/monkey_data_xycoord',
                        help='path to kinematics data')

    parser.add_argument('--test_data_filename', 
                        type=str, 
                        default='',
                        help='filename for saving the testing data')

    return parser