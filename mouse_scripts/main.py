import pybullet as p
import numpy as np
import time
import argparse
import itertools
import scipy.io
import torch
import matplotlib.pyplot as plt
from scipy.interpolate import Akima1DInterpolator

import farms_pylog as pylog
import model_utils as model_utils
from model_utils import get_speed
from Mouse_RL_Environment import Mouse_Env, Mouse_Env_Simulated
from SAC.replay_memory import PolicyReplayMemoryRNN, PolicyReplayMemoryLSTM
from SAC.sac import SAC, SACRNN, SACLSTM

file_path = "model_utilities/mouse_fixed.sdf" # mouse model, body fixed except for right arm
pose_file = "model_utilities/right_forelimb_pose.yaml" # pose file for original pose
muscle_config_file = "model_utilities/right_forelimb.yaml" # muscle file for right arm

model_offset = (0.0, 0.0, .0475) # z position modified with global scaling

#ARM CONTROL
ctrl = [107, 108, 109, 110, 111, 113, 114]

###JOINT TO INDEX###
#RShoulder_rotation - 107
#RShoulder_adduction - 108
#RShoulder_flexion - 109
#RElbow_flexion - 110
#RElbow_supination - 111
#RWrist_adduction - 113
#RWrist_flexion - 114
#RMetacarpus1_flexion - 115, use link (carpus) for pos

def get_avg_speed(data):
    speed_list = []
    for i in range(1, len(data)):
        speed_list.append(get_speed(data[i], data[i-1]))
    return sum(speed_list)/len(speed_list)
 
def preprocess(cycles):

    ########################### Data_Fast ###############################
    mat = scipy.io.loadmat('data/kinematics_session_mean_alt_fast.mat')
    data = np.array(mat['kinematics_session_mean'][2])
    data_fast_orig = data[231:401:1] * -1
    data_fast_orig = [-13.452503122486936, *data_fast_orig[8:-1]]
    data_fast = [*data_fast_orig] * cycles

    # This needs to be done for smooth kinematics with cycles since they end at arbitrary points
    x = np.arange(0, len(data_fast))
    cs = Akima1DInterpolator(x, data_fast)
    # end point of kinematics and start point of next cycle
    x_interp = np.linspace(len(data_fast_orig)-1, len(data_fast_orig), 16)
    y_interp = cs(x_interp)
    # Get the new interpolated kinematics without repeating points
    fast_once_cycle_len = len([*data_fast_orig, *y_interp[1:-1]])
    data_fast = [*data_fast_orig, *y_interp[1:-1]] * cycles
    np.save('mouse_experiments/data/interp_fast', data_fast)

    # Data must start and end at same spot or there is jump
    ########################### Data_Slow ###############################
    mat = scipy.io.loadmat('data/kinematics_session_mean_alt_slow.mat')
    data = np.array(mat['kinematics_session_mean'][2])
    data_slow_orig = data[256:476:1] * -1
    data_slow_orig = [*data_slow_orig[:-6]]
    data_slow = [*data_slow_orig] * cycles

    x = np.arange(0, len(data_slow))
    cs = Akima1DInterpolator(x, data_slow)
    x_interp = np.linspace(len(data_slow_orig)-1, len(data_slow_orig), 5)
    y_interp = cs(x_interp)
    slow_once_cycle_len = len([*data_slow_orig, *y_interp[1:-1]])
    data_slow = [*data_slow_orig, *y_interp[1:-1]] * cycles
    np.save('mouse_experiments/data/interp_slow', data_slow)

    ############################ Data_1 ##############################
    mat = scipy.io.loadmat('data/kinematics_session_mean_alt1.mat')
    data = np.array(mat['kinematics_session_mean'][2])
    data_1_orig = data[226:406:1] * -1
    data_1_orig = [-13.452503122486936, *data_1_orig[4:-3]]
    data_1 = [*data_1_orig] * cycles

    x = np.arange(0, len(data_1))
    cs = Akima1DInterpolator(x, data_1)
    x_interp = np.linspace(len(data_1_orig)-1, len(data_1_orig), 3)
    y_interp = cs(x_interp)
    med_once_cycle_len = len([*data_1_orig, *y_interp[1:-1]])
    data_1 = [*data_1_orig, *y_interp[1:-1]] * cycles
    np.save('mouse_experiments/data/interp_1', data_1)

    return data_fast, data_slow, data_1, fast_once_cycle_len, slow_once_cycle_len, med_once_cycle_len

def train_episode(mouseEnv, agent, policy_memory, episode_reward, episode_steps, one_cycle_len, args):

    done = False
    ### GET INITAL STATE + RESET MODEL BY POSE
    state = mouseEnv.get_start_state()
    ep_trajectory = []

    policy_loss_tracker = []
    policy_loss_2_tracker = []
    policy_loss_3_tracker = []
    policy_loss_4_tracker = []

    #num_layers specified in the policy model 
    h_prev = torch.zeros(size=(1, 1, args.hidden_size))
    c_prev = torch.zeros(size=(1, 1, args.hidden_size))

    ### STEPS PER EPISODE ###
    for i in range(mouseEnv._max_episode_steps):
        
        with torch.no_grad():
            action, h_current, c_current, _ = agent.select_action(state, h_prev, c_prev, evaluate=False)  # Sample action from policy
        
        if i < one_cycle_len:
            # larger for first cycle
            mouseEnv.threshold = 0.0035
        else:
            # tighter for other cycles
            mouseEnv.threshold = 0.003

        ### SIMULATION ###
        if len(policy_memory.buffer) > args.policy_batch_size:
            # Number of updates per step in environment
            for j in range(args.updates_per_step):
                # Update parameters of all the networks
                if args.type == 'rnn':
                    critic_1_loss, critic_2_loss, policy_loss, policy_loss_2, policy_loss_3, policy_loss_4, ent_loss, alpha = agent.update_parameters(policy_memory, args.policy_batch_size)

                    policy_loss_2_tracker.append(policy_loss_2)
                    policy_loss_3_tracker.append(policy_loss_3)
                    policy_loss_4_tracker.append(policy_loss_4)

                elif args.type == 'lstm':
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(policy_memory, args.policy_batch_size)

                policy_loss_tracker.append(policy_loss)

        ### TRACKING REWARD + EXPERIENCE TUPLE###
        next_state, reward, done = mouseEnv.step(action, i)
        episode_reward += reward
        episode_steps += 1

        mask = 1 if episode_steps == mouseEnv._max_episode_steps else float(not done)

        if args.type == 'rnn':
            ep_trajectory.append((state, action, reward, next_state, mask, h_current.squeeze(0).cpu().numpy(),  c_current.squeeze(0).cpu().numpy()))
        elif args.type == 'lstm':
            ep_trajectory.append((state, action, np.array([reward]), next_state, np.array([mask]), h_prev.detach().cpu(), c_prev.detach().cpu(), h_current.detach().cpu(),  c_current.detach().cpu()))

        state = next_state
        h_prev = h_current
        c_prev = c_current
        
        ### EARLY TERMINATION OF EPISODE
        if done:
            break

    return ep_trajectory, episode_reward, episode_steps, policy_loss_tracker, policy_loss_2_tracker, policy_loss_3_tracker, policy_loss_4_tracker

def test(mouseEnv, agent, episode_reward, episode_steps, args):

    episode_reward = 0
    done = False

    x_kinematics = []
    lstm_activity = []

    ### GET INITAL STATE + RESET MODEL BY POSE
    state = mouseEnv.get_cur_state()

    #num_layers specified in the policy model 
    h_prev = torch.zeros(size=(1, 1, args.hidden_size))
    c_prev = torch.zeros(size=(1, 1, args.hidden_size))

    ### STEPS PER EPISODE ###
    for i in range(mouseEnv._max_episode_steps):

        hand_pos = p.getLinkState(mouseEnv.model, 115)[0][0]
        x_kinematics.append(hand_pos)

        with torch.no_grad():
            action, h_current, c_current, lstm_out = agent.select_action(state, h_prev, c_prev, evaluate=True)  # Sample action from policy
            lstm_out = np.squeeze(lstm_out)
            lstm_activity.append(lstm_out)

        ### TRACKING REWARD + EXPERIENCE TUPLE###
        next_state, reward, done = mouseEnv.step(action, i)
        episode_reward += reward

        state = next_state
        h_prev = h_current
        c_prev = c_current

        episode_steps += 1
        
        ### EARLY TERMINATION OF EPISODE
        if done:
            break
    
    return episode_reward, x_kinematics, lstm_activity

def main():

    ### PARAMETERS ###
    parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
    parser.add_argument('--env-name', default="HalfCheetah-v2",
                        help='Mujoco Gym environment (default: HalfCheetah-v2)')
    parser.add_argument('--eval', type=bool, default=False,
                        help='Evaluates a policy a policy every 10 episode (default: True)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='G',
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
    parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                        help='maximum number of steps (default: 1000000)')
    parser.add_argument('--hidden_size', type=int, default=512, metavar='N',
                        help='hidden size (default: 1000)')
    parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                        help='model updates per simulator step (default: 1)')
    parser.add_argument('--policy_replay_size', type=int, default=5000, metavar='N',
                        help='size of replay buffer (default: 2800)')
    parser.add_argument('--cuda', action="store_true",
                        help='run on CUDA (default: False)')
    parser.add_argument('--threshold', type=float, default=0.0035, metavar='G',
                        help='threshold (default: 0.0035)')
    parser.add_argument('--visualize', type=bool, default=False,
                        help='visualize mouse')
    parser.add_argument('--env_type', type=str, default='kin',
                        help='type of environment (kin, sim)')
    parser.add_argument('--test_model', type=bool, default=False,
                        help='test kinematics and get activities')
    parser.add_argument('--save_model', type=bool, default=False,
                        help='save models and optimizer during training')
    parser.add_argument('--model_save_name', type=str, default='',
                        help='name used to save the model with')
    parser.add_argument('--type', type=str, default='rnn',
                        help='There are two types: rnn or lstm. RNN uses multiple losses, LSTM is original implementation')
    parser.add_argument('--two_speeds', type=bool, default=False,
                        help='Only train on slow an medium speed, leave fast for testing')
    parser.add_argument('--cost_scale', type=float, default=0.0, metavar='G',
                        help='scaling of the cost, default: 0.0')
    parser.add_argument('--cycles', type=int, default=2, metavar='N',
                        help='Number of times to cycle the kinematics (Default: 1)')
    parser.add_argument('--training_desc', type=str, default='None', metavar='N',
                        help='A description of the training procedure for a saved model')
    args = parser.parse_args()

    ###SIMULATION PARAMETERS###
    frame_skip = 1
    timestep = 170

    ### DATA SET LOADING/PROCESSING ###
    data_fast, data_slow, data_1, fast_cycle_len, slow_cycle_len, med_cycle_len = preprocess(args.cycles)
    all_datasets = [data_fast, data_slow, data_1]
    cycle_lens = [fast_cycle_len, slow_cycle_len, med_cycle_len]
    dataset_names = ['data_fast', 'data_slow', 'data_1']
    sim_timesteps = [150, 200, 250]
    max_cycle_len = len(data_slow)

    highest_reward_1 = -50
    highest_reward_fast = -50
    highest_reward_slow = -50

    ### CREATE ENVIRONMENT, AGENT, MEMORY ###
    if args.env_type == 'kin':
        mouseEnv = Mouse_Env(file_path, muscle_config_file, pose_file, frame_skip, ctrl, timestep, model_offset, args.visualize, args.threshold, args.cost_scale, max_cycle_len)
    elif args.env_type == 'sim':
        mouseEnv = Mouse_Env_Simulated(file_path, muscle_config_file, pose_file, frame_skip, ctrl, timestep, model_offset, args.visualize, args.threshold, args.cost_scale)
    else:
        raise NotImplementedError

    if args.type == 'rnn':
        policy_memory = PolicyReplayMemoryRNN(args.policy_replay_size, args.seed)
        agent = SACRNN(45, mouseEnv.action_space, args)
    elif args.type == 'lstm':
        policy_memory = PolicyReplayMemoryLSTM(args.policy_replay_size, args.seed)
        agent = SACLSTM(45, mouseEnv.action_space, args)
    else:
        raise NotImplementedError

    if args.test_model:
        agent.critic.load_state_dict(torch.load(f'models/value_net_{args.model_save_name}.pth'))
        agent.policy.load_state_dict(torch.load(f'models/policy_net_{args.model_save_name}.pth'))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ### DISABLES CURRENT MOVEMENT ###
    model_utils.disable_control(mouseEnv.model)
    ### 1SEC REAL TIME = 1 ms SIMULATION ###
    p.setTimeStep(.001)

    ### INITIALIZE ALL VALUES TO TRACK ###
    reward_tracker_slow = []
    reward_tracker_fast = []
    reward_tracker_1 = []

    highest_reward = 0

    policy_loss_tracker = []
    policy_loss_2_tracker = []
    policy_loss_3_tracker = []
    policy_loss_4_tracker = []

    ### BEGIN TRAINING LOOP
    for i_episode in itertools.count(1):

        episode_reward = 0
        episode_steps = 0

        # Select the speed based on environment type
        if args.env_type == 'kin':
            mouseEnv._max_episode_steps = len(all_datasets[i_episode % 3])
            mouseEnv.x_pos = all_datasets[i_episode % 3]
            #mouseEnv.avg_vel = get_avg_speed(mouseEnv.x_pos)
            data_curr = dataset_names[i_episode % 3]
            one_cycle_len = cycle_lens[i_episode % 3]
        elif args.env_type == 'sim':
            mouseEnv.timestep = sim_timesteps[i_episode % 3]
        
        # reset after changing the speed
        mouseEnv.reset(pose_file)

        # Training
        if not args.test_model:

            # Skip the fast speed during training if only using two speeds
            if i_episode % 3 == 0 and args.two_speeds:
                continue
            
            # Run the episode
            ep_trajectory, episode_reward, episode_steps, policy_loss, policy_loss_2, policy_loss_3, policy_loss_4 = train_episode(mouseEnv, agent, policy_memory, episode_reward, episode_steps, one_cycle_len, args)

            if len(policy_memory.buffer) > args.policy_batch_size:

                policy_loss_tracker.append(policy_loss)
                policy_loss_2_tracker.append(policy_loss_2)
                policy_loss_3_tracker.append(policy_loss_3)
                policy_loss_4_tracker.append(policy_loss_4)

                np.save(f'mouse_experiments/data/policy_loss_{args.model_save_name}', policy_loss_tracker)
                np.save(f'mouse_experiments/data/policy_loss_2_{args.model_save_name}', policy_loss_2_tracker)
                np.save(f'mouse_experiments/data/policy_loss_3_{args.model_save_name}', policy_loss_3_tracker)
                np.save(f'mouse_experiments/data/policy_loss_4_{args.model_save_name}', policy_loss_4_tracker)

            ### SAVING MODELS + TRACKING VARIABLES ###
            if episode_reward > highest_reward:
                highest_reward = episode_reward 
            
            # Save the model if necessary
            if args.save_model:
                torch.save(agent.policy.state_dict(), f'models/policy_net_{args.model_save_name}.pth')
                torch.save(agent.critic.state_dict(), f'models/value_net_{args.model_save_name}.pth')

            # Printing rewards
            pylog.debug('Iteration: {} | reward with total timestep {} ({} speed): {}, timesteps completed: {}'.format(i_episode, mouseEnv._max_episode_steps, data_curr, episode_reward, episode_steps))
            pylog.debug('highest reward so far: {}'.format(highest_reward))

            # Push the episode to replay
            policy_memory.push(ep_trajectory)

        # Testing, i.e. getting kinematics and activities
        else:

            # Run the episode for testing
            episode_reward, x_kinematics, lstm_activity = test(mouseEnv, agent, episode_reward, episode_steps, args)

            # Check to see the highest reward for each speed, then save
            if episode_reward > highest_reward_1 and data_curr == 'data_1':
                    x_kinematics = np.array(x_kinematics)
                    lstm_activity = np.array(lstm_activity)
                    print(f'New highest reward for data_1: {episode_reward}')
                    np.save('mouse_experiments/data/mouse_1', x_kinematics)
                    np.save('mouse_experiments/data/mouse_1_activity', lstm_activity)
                    highest_reward_1 = episode_reward

            elif episode_reward > highest_reward_slow and data_curr == 'data_slow':
                    x_kinematics = np.array(x_kinematics)
                    lstm_activity = np.array(lstm_activity)
                    print(f'New highest reward for data_slow: {episode_reward}')
                    np.save('mouse_experiments/data/mouse_slow', x_kinematics)
                    np.save('mouse_experiments/data/mouse_slow_activity', lstm_activity)
                    highest_reward_slow = episode_reward

            elif episode_reward > highest_reward_fast and data_curr == 'data_fast':
                    x_kinematics = np.array(x_kinematics)
                    lstm_activity = np.array(lstm_activity)
                    print(f'New highest reward for data_fast: {episode_reward}')
                    np.save('mouse_experiments/data/mouse_fast', x_kinematics)
                    np.save('mouse_experiments/data/mouse_fast_activity', lstm_activity)
                    highest_reward_fast = episode_reward

    mouseEnv.close() #disconnects server

if __name__ == '__main__':
    main()
