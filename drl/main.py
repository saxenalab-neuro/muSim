import numpy as np
import time
import argparse
import itertools
import scipy.io
import torch
import matplotlib.pyplot as plt
import gym
import model_utils_snn as model_utils
from model_utils_snn import get_speed
from SAC.replay_memory_snn import PolicyReplayMemorySNN, PolicyReplayMemoryANN
from SAC.sac import SAC, SACSNN, SACANN
import warmup  # noqa

def train_episode(env, agent, policy_memory, episode_reward, episode_steps, args):

    done = False
    state = env.reset()
    policy_loss_tracker = []
    ep_trajectory = []

    mem2_rec_policy = {}
    spk2_rec_policy = {}

    if args.model == 'snn':
        for name in agent.policy.named_children():
            if "lif" in name[0]:
                    spk2_rec_policy[name[0]], mem2_rec_policy[name[0]] = name[1].init_rleaky()

    ### STEPS PER EPISODE ###
    for i in range(env._max_episode_steps):

        with torch.no_grad():
            if args.model == 'snn':
                action, mem2_rec_policy, spk2_rec_policy = agent.select_action(state, spk2_rec_policy, mem2_rec_policy, evaluate=False)  # Sample action from policy
            elif args.model == 'ann':
                action = agent.select_action(state, evaluate=False)  # Sample action from policy

        ### TRACKING REWARD + EXPERIENCE TUPLE###
        next_state, reward, done, info = env.step(action)
        episode_reward += reward

        if args.visualize == True:
             env.render()

        mask = 0 if done else 1

        ep_trajectory.append([list(state), list(action), reward, list(next_state), mask])

        state = next_state
        episode_steps += 1

        ### EARLY TERMINATION OF EPISODE
        if done:
            break

    policy_memory.push(ep_trajectory)

    ### SIMULATION ###
    if len(policy_memory.buffer) > args.policy_batch_size:
        critic_1_loss, critic_2_loss, policy_loss = agent.update_parameters(policy_memory, args.policy_batch_size)

    return episode_reward, episode_steps, policy_loss_tracker

def test(mouseEnv, agent, episode_reward, episode_steps, args):

    episode_reward = 0
    done = False

    x_kinematics = []
    lstm_activity = []

    ### GET INITAL STATE + RESET MODEL BY POSE
    state = mouseEnv.get_cur_state()

    ### STEPS PER EPISODE ###
    for i in range(mouseEnv._max_episode_steps):

        with torch.no_grad():
            action = agent.select_action(state, evaluate=True)  # Sample action from policy

        ### TRACKING REWARD + EXPERIENCE TUPLE###
        next_state, reward, done = mouseEnv.step(action, i)
        episode_reward += reward

        state = next_state
        episode_steps += 1
        
        ### EARLY TERMINATION OF EPISODE
        if done:
            break
    
    return episode_reward, x_kinematics, lstm_activity

def main():

    ### PARAMETERS ###
    parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
    parser.add_argument('--env_name', type=str, default="humanreacher-v0",
                        help='humanreacher-v0, muscle_arm-v0, torque_arm-v0')
    parser.add_argument('--model', type=str, default="snn",
                        help='snn, ann')
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
    parser.add_argument('--policy_batch_size', type=int, default=6, metavar='N',
                        help='batch size (default: 6)')
    parser.add_argument('--hidden_size', type=int, default=512, metavar='N',
                        help='hidden size (default: 1000)')
    parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                        help='model updates per simulator step (default: 1)')
    parser.add_argument('--policy_replay_size', type=int, default=1000000, metavar='N',
                        help='size of replay buffer (default: 2800)')
    parser.add_argument('--cuda', action="store_true",
                        help='run on CUDA (default: False)')
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
    parser.add_argument('--cost_scale', type=float, default=0.0, metavar='G',
                        help='scaling of the cost, default: 0.0')
    args = parser.parse_args()

    env = gym.make(args.env_name)

    if args.model == 'snn':
        policy_memory = PolicyReplayMemorySNN(args.policy_replay_size, args.seed)
        agent = SACSNN(env.observation_space.shape[0], env.action_space.shape[0], args)
    elif args.model == 'ann':
        policy_memory = PolicyReplayMemoryANN(args.policy_replay_size, args.seed)
        agent = SACANN(env.observation_space.shape[0], env.action_space.shape[0], args)

    if args.test_model:
        agent.critic.load_state_dict(torch.load(f'models/value_net_{args.model_save_name}.pth'))
        agent.policy.load_state_dict(torch.load(f'models/policy_net_{args.model_save_name}.pth'))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ### INITIALIZE ALL VALUES TO TRACK ###
    highest_reward = -1000
    reward_tracker = []
    policy_loss_tracker = []

    ### BEGIN TRAINING LOOP
    for i_episode in itertools.count(1):

        episode_reward = 0
        episode_steps = 0

        # Training
        if not args.test_model:

            # Run the episode
            episode_reward, episode_steps, policy_loss = train_episode(env, agent, policy_memory, episode_reward, episode_steps, args)

            ### SAVING MODELS + TRACKING VARIABLES ###
            if episode_reward > highest_reward:
                highest_reward = episode_reward 
            
            # Save the model if necessary
            if args.save_model:
                torch.save(agent.policy.state_dict(), f'models/policy_net_{args.model_save_name}.pth')
                torch.save(agent.critic.state_dict(), f'models/value_net_{args.model_save_name}.pth')

            # Printing rewards
            print('Iteration: {} | reward {} | timesteps completed: {}'.format(i_episode, episode_reward, episode_steps))
            print('highest reward so far: {}'.format(highest_reward))

        # Testing, i.e. getting kinematics and activities
        else:

            # Run the episode for testing
            episode_reward, x_kinematics, lstm_activity = test(env, agent, episode_reward, episode_steps, args)

    env.close() #disconnects server

if __name__ == '__main__':
    main()
