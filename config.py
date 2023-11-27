import configargparse

def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument("--config", is_config_file=True, help="config file path")
    parser.add_argument('--env_name', type=str, default="monkey",
                        help='humanreacher-v0, muscle_arm-v0, torque_arm-v0')
    parser.add_argument('--model', type=str, default="rnn",
                        help='rnn, gru')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy\
                                term against the reward (default: 0.2)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                        help='Automaically adjust α (default: False)')
    parser.add_argument('--seed', type=int, default=123456, metavar='N',
                        help='random seed (default: 123456)')
    parser.add_argument('--policy_batch_size', type=int, default=8, metavar='N',
                        help='batch size (default: 6)')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 1000)')
    parser.add_argument('--policy_replay_size', type=int, default=50000, metavar='N',
                        help='size of replay buffer (default: 2800)')
    parser.add_argument('--multi_policy_loss', type=bool, default=False, metavar='N',
                        help='use additional policy losses')
    parser.add_argument('--batch_iters', type=int, default=1, metavar='N',
                        help='iterations to apply update')
    parser.add_argument('--cuda', action="store_true",
                        help='run on CUDA (default: False)')
    parser.add_argument('--visualize', type=bool, default=False,
                        help='visualize mouse')
    parser.add_argument('--model_save_name', type=str, default='',
                        help='name used to save the model with')
    parser.add_argument('--total_episodes', type=int, default=5000000, metavar='N',
                        help='total number of episodes')
    parser.add_argument('--save_iter', type=int, default=1000, metavar='N',
                        help='number of episodes until checkpoint is saved')
    parser.add_argument('--test', type=bool, default=False, metavar='N',
                        help='test a model')
    parser.add_argument('--checkpoint_path', type=str, default='',
                        help='path of checkpoint training parameters')
    parser.add_argument('--muscle_path', type=str, default='monkey/monkeyArm_current_scaled.xml',
                        help='path of musculoskeletal model')
    parser.add_argument('--muscle_params_path', type=str, default='monkey/params_monkey.pckl',
                        help='path of musculoskeletal model parameters')
    return parser