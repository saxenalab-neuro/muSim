import numpy as np
import time
import argparse
import itertools
import scipy.io
import torch
import matplotlib.pyplot as plt
import gym
from SAC.replay_memory import PolicyReplayMemoryLSNN, PolicyReplayMemoryANN, PolicyReplayMemoryLSTM, PolicyReplayMemorySNN
from SAC.sac import SAC, SACLSNN, SACANN, SACLSTM, SACSNN
from simulation import Simulate_ANN, Simulate_LSTM, Simulate_LSNN, Simulate_SNN
import warmup  # noqa
from tqdm import tqdm
from statistics import mean

def main():

    ### PARAMETERS ###
    parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
    parser.add_argument('--env_name', type=str, default="muscle_arm-v0",
                        help='humanreacher-v0, muscle_arm-v0, torque_arm-v0')
    parser.add_argument('--model', type=str, default="snn",
                        help='snn, ann')
    parser.add_argument('--gamma', type=float, default=0.9, metavar='G',
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
    parser.add_argument('--hidden_size', type=int, default=512, metavar='N',
                        help='hidden size (default: 1000)')
    parser.add_argument('--policy_replay_size', type=int, default=1000000, metavar='N',
                        help='size of replay buffer (default: 2800)')
    parser.add_argument('--batch_iters', type=int, default=30, metavar='N',
                        help='iterations to apply update')
    parser.add_argument('--experience_sampling', type=int, default=100, metavar='N',
                        help='how many episodes to run before training')
    parser.add_argument('--cuda', action="store_true",
                        help='run on CUDA (default: False)')
    parser.add_argument('--visualize', type=bool, default=False,
                        help='visualize mouse')
    parser.add_argument('--test_model', type=bool, default=False,
                        help='test kinematics and get activities')
    parser.add_argument('--save_model', type=bool, default=False,
                        help='save models and optimizer during training')
    parser.add_argument('--model_save_name', type=str, default='',
                        help='name used to save the model with')
    parser.add_argument('--total_episodes', type=int, default=5000000, metavar='N',
                        help='total number of episodes')
    parser.add_argument('--tracking', type=bool, default=False, metavar='N',
                        help='track experience')
    args = parser.parse_args()

    env = gym.make(args.env_name)

    if args.model == 'lsnn':
        policy_memory = PolicyReplayMemoryLSNN(args.policy_replay_size, args.seed)
        agent = SACLSNN(env.observation_space.shape[0], env.action_space.shape[0], args)
        simulator = Simulate_LSNN(env, agent, policy_memory, args.policy_batch_size, args.hidden_size, args.visualize, args.batch_iters, args.experience_sampling)
    if args.model == 'snn':
        policy_memory = PolicyReplayMemorySNN(args.policy_replay_size, args.seed)
        agent = SACSNN(env.observation_space.shape[0], env.action_space.shape[0], args)
        simulator = Simulate_SNN(env, agent, policy_memory, args.policy_batch_size, args.hidden_size, args.visualize, args.batch_iters, args.experience_sampling)
    elif args.model == 'ann':
        policy_memory = PolicyReplayMemoryANN(args.policy_replay_size, args.seed)
        agent = SACANN(env.observation_space.shape[0], env.action_space.shape[0], args)
        simulator = Simulate_ANN(env, agent, policy_memory, args.policy_batch_size, args.hidden_size, args.visualize, args.batch_iters, args.experience_sampling)
    elif args.model == 'lstm':
        policy_memory = PolicyReplayMemoryLSTM(args.policy_replay_size, args.seed)
        agent = SACLSTM(env.observation_space.shape[0], env.action_space.shape[0], args)
        simulator = Simulate_LSTM(env, agent, policy_memory, args.policy_batch_size, args.hidden_size, args.visualize, args.batch_iters, args.experience_sampling)

    # TODO checkpoints
    if args.test_model:
        agent.critic.load_state_dict(torch.load(f'models/value_net_{args.model_save_name}.pth'))
        agent.policy.load_state_dict(torch.load(f'models/policy_net_{args.model_save_name}.pth'))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ### INITIALIZE ALL VALUES TO TRACK ###
    highest_reward = -1000
    reward_tracker = []
    steps_tracker = []
    success_tracker = []

    ### BEGIN TRAINING LOOP
    for i_episode in tqdm(range(1, args.total_episodes)):

        episode_reward = 0
        episode_steps = 0

        # Training
        if not args.test_model:

            # Run the episode
            episode_reward, episode_steps, success = simulator.train(i_episode)

            reward_tracker.append(episode_reward)
            steps_tracker.append(episode_steps)
            success_tracker.append(success)

            ### SAVING MODELS + TRACKING VARIABLES ###
            if episode_reward > highest_reward:
                highest_reward = episode_reward 
            
            # Save the model if necessary
            if args.save_model:
                torch.save(agent.policy.state_dict(), f'models/policy_net_{args.model_save_name}.pth')
                torch.save(agent.critic.state_dict(), f'models/value_net_{args.model_save_name}.pth')

            if i_episode % args.experience_sampling == 0:
                # Printing rewards
                print('highest reward: {} | mean_reward: {} | mean timesteps completed: {}'.format(max(reward_tracker), mean(reward_tracker), mean(steps_tracker)))
                print('highest reward so far: {}'.format(highest_reward))

            if args.tracking == True:
                np.savetxt(f'tracking/success/episode_success_{args.model_save_name}', success_tracker)
                np.savetxt(f'tracking/rewards/episode_rewards_{args.model_save_name}', reward_tracker)
                np.savetxt(f'tracking/policy_losses/policy_loss_{args.model_save_name}', simulator.policy_loss_tracker)
                np.savetxt(f'tracking/critic_1_losses/critic_1_loss_{args.model_save_name}', simulator.critic1_loss_tracker)
                np.savetxt(f'tracking/critic_1_losses/critic_2_loss_{args.model_save_name}', simulator.critic2_loss_tracker)

        # Testing, i.e. getting kinematics and activities
        else:

            # Run the episode for testing
            episode_reward, success = simulator.test(i_episode)
            print(f'Episode Reward: {episode_reward} | Success: {success}')

    env.close() #disconnects server

if __name__ == '__main__':
    main()
