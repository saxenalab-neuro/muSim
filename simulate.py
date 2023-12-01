import numpy as np
import torch
from SAC.sac import SAC_Agent
from SAC.replay_memory import PolicyReplayMemory
import pickle
import os

class Simulate():
    def __init__(self, 
                 env: str,
                 model: str,
                 gamma: float,
                 tau: float,
                 lr: float,
                 alpha: float,
                 automatic_entropy_tuning: bool,
                 seed: int,
                 policy_batch_size: int,
                 hidden_size: int,
                 policy_replay_size: int,
                 multi_policy_loss: bool,
                 batch_iters: int,
                 cuda: bool,
                 visualize: bool,
                 root_dir: str,
                 checkpoint_file: str,
                 checkpoint_folder: str,
                 episodes: int,
                 save_iter: int,
                 muscle_path: str,
                 muscle_params_path: str):

        """Train a soft actor critic agent to control a musculoskeletal model to follow a kinematic trajectory.

        Parameters
        ----------
        env: str
            specify which environment (model) to train on
        model: str
            specify whether to use an rnn or a gated rnn (gru)
        optimizer_spec: float
            gamma discount parameter in sac algorithm
        tau: float
            tau parameter for soft updates of the critic and critic target networks
        lr: float
            learning rate of the critic and actor networks
        alpha: float
            entropy term used in the sac policy update
        automatic entropy tuning: bool
            whether to use automatic entropy tuning during training
        seed: int
            seed for the environment and networks
        policy_batch_size: int
            the number of episodes used in a batch during training
        hidden_size: int
            number of hidden neurons in the actor and critic
        policy_replay_size: int
            number of episodes to store in the replay
        multi_policy_loss: bool
            use additional policy losses during updates (reduce norm of weights)
            ONLY USE WITH RNN, NOT IMPLEMENTED WITH GATING
        batch_iters: int
            How many experience replay rounds (not steps!) to perform between
            each update
        cuda: bool
            use cuda gpu
        visualize: bool
            visualize model
        model_save_name: str
            specify the name of the model to save
        episodes: int
            total number of episodes to run
        save_iter: int
            number of iterations before saving the model
        checkpoint_path: str
            specify path to save and load model
        muscle_path: str
            path for the musculoskeletal model
        muscle_params_path: str
            path for musculoskeletal model parameters
        """

        ### Load Custom Gym Environment ###
        self.env = env(muscle_path, muscle_params_path, 5)
        self.observation_shape = self.env.observation_space.shape[0]+3+3

        ### Load SAC Agent ###
        self.agent = SAC_Agent(self.observation_shape, 
                               self.env.action_space, 
                               hidden_size, 
                               lr, 
                               gamma, 
                               tau, 
                               alpha, 
                               automatic_entropy_tuning, 
                               model,
                               multi_policy_loss,
                               cuda)

        ### Load Replay Memory ###
        self.policy_memory = PolicyReplayMemory(policy_replay_size, seed)

        ### Specify Training Variables ###
        self.episodes = episodes
        self.hidden_size = hidden_size
        self.policy_batch_size = policy_batch_size
        self.visualize = visualize
        self.root_dir = root_dir
        self.checkpoint_file = checkpoint_file
        self.checkpoint_folder = checkpoint_folder
        self.batch_iters = batch_iters
        self.save_iter = save_iter

        ### Set Seed for Training ###
        torch.manual_seed(seed)
        np.random.seed(seed)

    def test(self, save_name):

        """ Use a saved model to generate kinematic trajectory

            saves these values: 
            --------
                episode_reward: int
                    - Total reward of the trained model
                kinematics: list
                    - Kinematics of the model during testing
                hidden_activity: list
                    - list containing the rnn activity during testing
        """

        Test_Values = {
            "hidden_act": [],
            "kinematics": [],
            "episode_reward": 0,
        }

        episode_reward = 0
        episode_steps = 0
        hidden_activity = []
        done = False

        ### GET INITAL STATE + RESET MODEL BY POSE
        state = self.env.reset()

        #num_layers specified in the policy model 
        h_prev = torch.zeros(size=(1, 1, self.hidden_size))

        ### STEPS PER EPISODE ###
        for timestep in range(self.env._max_episode_steps):

            ### Select Action ###
            with torch.no_grad():
                action, h_current, rnn_act = self.agent.select_action(state, h_prev, evaluate=True)
                hidden_activity.append(rnn_act)

            ### TRACKING REWARD + EXPERIENCE TUPLE###
            next_state, reward, done, info, episode_reward, episode_steps = self._step(action, timestep, episode_reward, episode_steps)
            episode_reward += reward

            ### VISUALIZE MODEL ###
            if self.visualize == True:
                self.env.render()

            state = next_state
            h_prev = h_current

            ### EARLY TERMINATION OF EPISODE
            if done:
                break
        
        # TODO get kinematics
        Test_Values["hidden_act"] = hidden_activity
        Test_Values["episode_reward"] = episode_reward
        
        with open(f'{save_name}.pkl', 'wb') as f:
            pickle.dump(Test_Values, f)
            print("Saved to %s" % f'{save_name}.pkl')
            print('--------------------------\n')

    def train(self):

        """ Train an RNN based SAC agent to follow kinematic trajectory
        """

        ### Dictionary for Storing Training Data ###
        Statistics = {
            "rewards": [],
            "steps": [],
            "policy_loss": [],
            "critic_loss": []
        }

        highest_reward = -float("inf") # used for storing highest reward throughout training

        ### Start Training ###
        for episode in range(self.episodes):

            ### Gather Episode Data Variables ###
            episode_reward = 0          # reward for single episode
            episode_steps = 0           # steps completed for single episode
            policy_loss_tracker = []    # stores policy loss throughout episode
            critic1_loss_tracker = []   # stores critic loss throughout episode
            done = False                # determines if episode is terminated

            ### GET INITAL STATE + RESET MODEL BY POSE
            state = self.env.reset()

            ep_trajectory = []  # used to store (s_t, a_t, r_t, s_t+1) tuple for replay storage

            h_prev = torch.zeros(size=(1, 1, self.hidden_size)) # num_layers specified in the policy model

            ### Loop Through Episode Timesteps ###
            for t in range(self.env._max_episode_steps):

                ### Select Action ###
                with torch.no_grad():
                    action, h_current, _ = self.agent.select_action(state, h_prev, evaluate=False)

                ### Update Critic and Actor Parameters ###
                if len(self.policy_memory.buffer) > self.policy_batch_size:
                    for _ in range(self.batch_iters):
                        critic_1_loss, critic_2_loss, policy_loss = self.agent.update_parameters(self.policy_memory, self.policy_batch_size)
                        ### Store Losses ###
                        policy_loss_tracker.append(policy_loss)
                        critic1_loss_tracker.append(critic_1_loss)

                ### SIMULATION ###
                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                episode_steps += 1

                ### VISUALIZE MODEL ###
                if self.visualize == True:
                    self.env.render()

                mask = 1 if episode_steps == self.env._max_episode_steps else float(not done) # ensure mask is not 0 if episode ends

                ### Store Tuple at Current Timestep ###
                ep_trajectory.append((state, 
                                        action, 
                                        reward, 
                                        next_state, 
                                        mask))  

                ### Move to Next State ###
                state = next_state
                h_prev = h_current

                ### EARLY TERMINATION OF EPISODE ###
                if done:
                    break
            
            # Push the episode to replay
            self.policy_memory.push(ep_trajectory)

            ### TRACKING ###
            Statistics["rewards"].append(episode_reward)
            Statistics["steps"].append(episode_steps)
            Statistics["policy_loss"].append(np.mean(np.array(policy_loss_tracker)))
            Statistics["critic_loss"].append(np.mean(np.array(critic1_loss_tracker)))

            ### Save to File ###
            with open(f'statistics_{self.checkpoint_file}.pkl', 'wb') as f:
                pickle.dump(Statistics, f)
                print("Saved to %s" % 'statistics.pkl')
                print('--------------------------\n')

            ### Determine Highest Reward ###
            if episode_reward > highest_reward:
                highest_reward = episode_reward 
            
            ### SAVING STATE DICT OF TRAINING ###
            f = os.path.join(self.root_dir, self.checkpoint_folder, self.checkpoint_file)
            if episode % self.save_iter == 0 and len(self.policy_memory.buffer) > self.policy_batch_size:
                torch.save({
                    'iteration': episode,
                    'agent_state_dict': self.agent.actor.state_dict(),
                    'critic_state_dict': self.agent.critic.state_dict(),
                    'critic_target_state_dict': self.agent.critic_target.state_dict(),
                    'agent_optimizer_state_dict': self.agent.actor_optim.state_dict(),
                    'critic_optimizer_state_dict': self.agent.critic_optim.state_dict(),
                }, f + '.pth')

            ### PRINT TRAINING OUTPUT ###
            print('-----------------------------------')
            print('highest reward: {} | reward: {} | timesteps completed: {}'.format(highest_reward, episode_reward, episode_steps))
            print('-----------------------------------\n')