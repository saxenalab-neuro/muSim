import numpy as np
import time
import argparse
import itertools
import scipy.io
import torch
import matplotlib.pyplot as plt

class Simulate(object):
    def __init__(self, env, agent, policy_memory, policy_batch_size, hidden_size, visualize):
        
        self.env = env
        self.agent = agent
        self.policy_memory = policy_memory
        self.policy_batch_size = policy_batch_size
        self.hidden_size = hidden_size
        self.visualize = visualize

        self.policy_loss_tracker = []
        self.critic1_loss_tracker = []
        self.critic2_loss_tracker = []

    def _step(self, action, iteration, episode_reward, episode_steps):
        ### TRACKING REWARD + EXPERIENCE TUPLE###
        next_state, reward, done, info = self.env.step(action)
        episode_reward += reward
        episode_steps += 1

        return next_state, reward, done, episode_reward, episode_steps
    
    def _check_update(self, iteration, batch_iters):
        if iteration % 100 == 0:
            for _ in range(batch_iters):
                critic_1_loss, critic_2_loss, policy_loss = self.agent.update_parameters(self.policy_memory, self.policy_batch_size)
                self.policy_loss_tracker.append(policy_loss)
                self.critic1_loss_tracker.append(critic_1_loss)
                self.critic2_loss_tracker.append(critic_2_loss)
    
    def train(self, iteration, batch_iters):
        pass

    def test(self):
        pass

    
class Simulate_RNN(Simulate):
    def __init__(self, env, agent, policy_memory, policy_batch_size, hidden_size, visualize):
        super(Simulate_RNN, self).__init__(env, agent, policy_memory, policy_batch_size, hidden_size, visualize)

    def train(self, iteration, batch_iters):

        done = False
        episode_reward = 0
        episode_steps = 0

        ### GET INITAL STATE + RESET MODEL BY POSE
        state = self.env.reset()

        ep_trajectory = []

        #num_layers specified in the policy model 
        h_prev = torch.zeros(size=(1, 1, self.hidden_size))

        ### STEPS PER EPISODE ###
        for i in range(self.env._max_episode_steps):
            with torch.no_grad():
                action, h_current, c_current, _ = self.agent.select_action(state, h_prev, evaluate=False)  # Sample action from policy
            
            ### TRACKING REWARD + EXPERIENCE TUPLE###
            next_state, reward, done, episode_reward, episode_steps = self._step(action, i, episode_reward, episode_steps)

            if self.visualize == True:
                self.env.render()

            mask = 0 if done else 1

            ep_trajectory.append((state, action, reward, next_state, mask, h_current.squeeze(0).cpu().numpy(),  c_current.squeeze(0).cpu().numpy()))

            state = next_state
            h_prev = h_current
            
            ### EARLY TERMINATION OF EPISODE
            if done:
                break

        ### SIMULATION ###
        self._check_update(iteration, batch_iters)
        
        # Push the episode to replay
        self.policy_memory.push(ep_trajectory)

        return episode_reward, episode_steps
    
    def test(self):

        episode_reward = 0
        done = False

        x_kinematics = []
        rnn_activity = []

        ### GET INITAL STATE + RESET MODEL BY POSE
        state = self.env.reset()

        #num_layers specified in the policy model 
        h_prev = torch.zeros(size=(1, 1, self.hidden_size))

        ### STEPS PER EPISODE ###
        for i in range(self.env._max_episode_steps):

            with torch.no_grad():
                action, h_current, c_current, rnn_out = self.agent.select_action(state, h_prev, evaluate=True)  # Sample action from policy
                rnn_out = np.squeeze(rnn_out)
                rnn_activity.append(rnn_out)

            ### TRACKING REWARD + EXPERIENCE TUPLE###
            next_state, reward, done = self.env.step(action, i)
            episode_reward += reward

            if self.visualize == True:
                self.env.render()

            state = next_state
            h_prev = h_current

            ### EARLY TERMINATION OF EPISODE
            if done:
                break
        
        return episode_reward, x_kinematics, rnn_activity
        

class Simulate_ANN(Simulate):
    def __init__(self, env, agent, policy_memory, policy_batch_size, hidden_size, visualize):
        super(Simulate_ANN, self).__init__(env, agent, policy_memory, policy_batch_size, hidden_size, visualize)

    def train(self, iteration, batch_iters):

        done = False
        episode_reward = 0
        episode_steps = 0

        ### GET INITAL STATE + RESET MODEL BY POSE
        state = self.env.reset()

        ### STEPS PER EPISODE ###
        for i in range(self.env._max_episode_steps):

            with torch.no_grad():
                action = self.agent.select_action(state, evaluate=False)  # Sample action from policy
            
            ### TRACKING REWARD + EXPERIENCE TUPLE###
            next_state, reward, done, episode_reward, episode_steps = self._step(action, i, episode_reward, episode_steps)

            if self.visualize == True:
                self.env.render()

            mask = 0 if done else 1

            self.policy_memory.push([list(state), list(action), reward, list(next_state), mask])

            state = next_state
            
            ### EARLY TERMINATION OF EPISODE
            if done:
                break

        ### SIMULATION ###
        self._check_update(iteration, batch_iters)

        return episode_reward, episode_steps
    
    def test(self):

        episode_reward = 0
        done = False

        x_kinematics = []
        lstm_activity = []

        ### GET INITAL STATE + RESET MODEL BY POSE
        state = self.env.reset()

        ### STEPS PER EPISODE ###
        for i in range(self.env._max_episode_steps):

            with torch.no_grad():
                action = self.agent.select_action(state, evaluate=True)  # Sample action from policy

            ### TRACKING REWARD + EXPERIENCE TUPLE###
            next_state, reward, done = self.env.step(action, i)
            episode_reward += reward

            if self.visualize == True:
                self.env.render()

            state = next_state

            ### EARLY TERMINATION OF EPISODE
            if done:
                break
        
        return episode_reward, x_kinematics, lstm_activity


class Simulate_RSNN(Simulate):
    def __init__(self, env, agent, policy_memory, policy_batch_size, hidden_size, visualize):
        super().__init__(env, agent, policy_memory, policy_batch_size, hidden_size, visualize)
    
    def _init_rleaky(self):
        mem2_rec = {}
        spk2_rec = {}
        for name in self.agent.policy.named_children():
            if "lif" in name[0]:
                    spk2_rec[name[0]], mem2_rec[name[0]] = name[1].init_rleaky()
        return spk2_rec, mem2_rec

    def train(self, iteration, batch_iters):

        done = False
        episode_reward = 0
        episode_steps = 0

        ### GET INITAL STATE + RESET MODEL BY POSE
        state = self.env.reset()

        ep_trajectory = []

        #num_layers specified in the policy model 
        spk2_rec_policy, mem2_rec_policy = self._init_rleaky()

        ### STEPS PER EPISODE ###
        for i in range(self.env._max_episode_steps):

            with torch.no_grad():
                action, mem2_rec_policy, spk2_rec_policy = self.agent.select_action(state, spk2_rec_policy, mem2_rec_policy, evaluate=False)  # Sample action from policy
            
            ### TRACKING REWARD + EXPERIENCE TUPLE###
            next_state, reward, done, episode_reward, episode_steps = self._step(action, i, episode_reward, episode_steps)
            mask = 0 if done else 1
            state = next_state

            if self.visualize == True:
                self.env.render()
            
            ### EARLY TERMINATION OF EPISODE
            if done:
                ep_trajectory.append([state, action, reward, next_state, mask, episode_steps])
                break
            else:
                ep_trajectory.append([state, action, reward, next_state, mask])

        ### SIMULATION ###
        self._check_update(iteration, batch_iters)

        # Push the episode to replay
        self.policy_memory.push(ep_trajectory)

        return episode_reward, episode_steps
    
    def test(self):

        episode_reward = 0
        done = False

        x_kinematics = []
        activity = []

        ### GET INITAL STATE + RESET MODEL BY POSE
        state = self.env.reset()

        #num_layers specified in the policy model 
        spk2_rec_policy, mem2_rec_policy = self._init_rleaky()

        ### STEPS PER EPISODE ###
        for i in range(self.env._max_episode_steps):

            with torch.no_grad():
                action, mem2_rec_policy, spk2_rec_policy = self.agent.select_action(state, spk2_rec_policy, mem2_rec_policy, evaluate=True)  # Sample action from policy

            ### TRACKING REWARD + EXPERIENCE TUPLE###
            next_state, reward, done = self.env.step(action, i)
            episode_reward += reward

            if self.visualize == True:
                self.env.render()

            state = next_state

            ### EARLY TERMINATION OF EPISODE
            if done:
                break
        
        return episode_reward, x_kinematics, activity


class Simulate_SNN(Simulate):
    def __init__(self, env, agent, policy_memory, policy_batch_size, hidden_size, visualize):
        super().__init__(env, agent, policy_memory, policy_batch_size, hidden_size, visualize)
    
    def _init_leaky(self):
        mem2_rec = {}
        for name in self.agent.policy.named_children():
            if "lif" in name[0]:
                mem2_rec[name[0]] = name[1].init_leaky()
        return mem2_rec

    def train(self, iteration, batch_iters):

        done = False
        episode_reward = 0
        episode_steps = 0

        ### GET INITAL STATE + RESET MODEL BY POSE
        state = self.env.reset()

        ep_trajectory = []

        #num_layers specified in the policy model 
        mem2_rec_policy = self._init_leaky()

        ### STEPS PER EPISODE ###
        for i in range(self.env._max_episode_steps):

            with torch.no_grad():
                action, mem2_rec_policy = self.agent.select_action(state, mem2_rec_policy, evaluate=False)  # Sample action from policy
            
            ### TRACKING REWARD + EXPERIENCE TUPLE###
            next_state, reward, done, episode_reward, episode_steps = self._step(action, i, episode_reward, episode_steps)
            mask = 0 if done else 1
            state = next_state

            if self.visualize == True:
                self.env.render()
            
            ### EARLY TERMINATION OF EPISODE
            if done:
                ep_trajectory.append([state, action, reward, next_state, mask, episode_steps])
                break
            else:
                ep_trajectory.append([state, action, reward, next_state, mask])

        ### SIMULATION ###
        self._check_update(iteration, batch_iters)

        # Push the episode to replay
        self.policy_memory.push(ep_trajectory)

        return episode_reward, episode_steps
    
    def test(self):

        episode_reward = 0
        done = False

        x_kinematics = []
        activity = []

        ### GET INITAL STATE + RESET MODEL BY POSE
        state = self.env.reset()

        #num_layers specified in the policy model 
        spk2_rec_policy, mem2_rec_policy = self._init_rleaky()

        ### STEPS PER EPISODE ###
        for i in range(self.env._max_episode_steps):

            with torch.no_grad():
                action, mem2_rec_policy, spk2_rec_policy = self.agent.select_action(state, spk2_rec_policy, mem2_rec_policy, evaluate=True)  # Sample action from policy

            ### TRACKING REWARD + EXPERIENCE TUPLE###
            next_state, reward, done = self.env.step(action, i)
            episode_reward += reward

            if self.visualize == True:
                self.env.render()

            state = next_state

            ### EARLY TERMINATION OF EPISODE
            if done:
                break
        
        return episode_reward, x_kinematics, activity