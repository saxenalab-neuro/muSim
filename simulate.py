import numpy as np
import torch
from SAC.sac import SAC_Agent
from SAC.replay_memory import PolicyReplayMemory
from SAC.RL_Framework_Mujoco import Muscle_Env
import pickle
import os

#Set the current working directory
os.chdir(os.getcwd())

class Simulate():
    def __init__(self, 
                 env: Muscle_Env,
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
                 muscle_params_path: str,
                 kinematics_path: str,
                 condition_selection_strategy: str):

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

        ### LOAD CUSTOM GYM ENVIRONMENT ###
        self.env = env(muscle_path, 1)
        self.observation_shape = self.env.observation_space.shape[0]+3+3+1

        ### SAC AGENT ###
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

        ### REPLAY MEMORY ###
        self.policy_memory = PolicyReplayMemory(policy_replay_size, seed)

        ### TRAINING VARIABLES ###
        self.episodes = episodes
        self.hidden_size = hidden_size
        self.policy_batch_size = policy_batch_size
        self.visualize = visualize
        self.root_dir = root_dir
        self.checkpoint_file = checkpoint_file
        self.checkpoint_folder = checkpoint_folder
        self.batch_iters = batch_iters
        self.save_iter = save_iter
        self.condition_selection_strategy = condition_selection_strategy

        ### ENSURE SAVING FILES ARE ACCURATE ###
        assert isinstance(self.root_dir, str)
        assert isinstance(self.checkpoint_folder, str)
        assert isinstance(self.checkpoint_file, str)

        ### SEED ###
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

        #Update the environment kinematics to both the training and testing conditions
        self.env.update_kinematics_for_test()

        ### LOAD SAVED MODEL ###
        f = os.path.join(self.root_dir, self.checkpoint_folder, self.checkpoint_file + '.pth')
        # self.agent.actor.load_state_dict(torch.load(f['agent_state_dict']))
        self.agent.actor.load_state_dict(torch.load(f))

        ### TESTING PERFORMANCE ###
        Test_Values = {
            "hidden_act": [],
            "kinematics_hand": [],
            "kinematics_target": [],
            "episode_reward": 0,
        }

        hidden_activity_cum = []
        kinematics_hand_cum = []
        kinematics_target_cum = []
        episode_reward_cum = []

        for episode in range(self.env.n_exp_conds):

            ### TRACKING VARIABLES ###
            episode_reward = 0
            episode_steps = 0
            hidden_activity = []
            kinematics_hand = []
            kinematics_target = []
            done = False

            ### GET INITAL STATE + RESET MODEL BY POSE
            cond_to_select = episode % self.env.n_exp_conds
            state = self.env.reset(cond_to_select)
            state = [*state, self.env.condition_scalar]

            # Num_layers specified in the policy model 
            h_prev = torch.zeros(size=(1, 1, self.hidden_size))

            ### STEPS PER EPISODE ###
            for timestep in range(self.env.timestep_limit):

                ### SELECT ACTION ###
                with torch.no_grad():
                    action, h_current, rnn_act = self.agent.select_action(state, h_prev, evaluate=True)
                    hidden_activity.append(rnn_act[0, :])  # [1, n_hidden_units] --> [n_hidden_units,]

                ### TRACKING REWARD + EXPERIENCE TUPLE###
                next_state, reward, done, _ = self.env.step(action)
                next_state = [*next_state, self.env.condition_scalar]
                episode_reward += reward

                #now append the kinematics of the hand and the target
                kinematics_hand.append(self.env.sim.data.get_body_xpos("hand").copy())          #[3, ]
                kinematics_target.append(self.env.sim.data.get_body_xpos("target").copy())      #[3, ]

                ### VISUALIZE MODEL ###
                if self.visualize == True:
                    self.env.render()

                state = next_state
                h_prev = h_current
                
            hidden_activity = np.array(hidden_activity)   #shape: [ep_timepoints, n_hidden_units]
            kinematics_hand = np.array(kinematics_hand)    #shape: [ep_timepoints, 3]
            kinematics_target = np.array(kinematics_target) #shape: [ep_timepoints, 3]

            hidden_activity_cum.append(hidden_activity)
            kinematics_hand_cum.append(kinematics_hand)
            kinematics_target_cum.append(kinematics_target)

        ### SAVE TESTING STATS ###
        Test_Values["hidden_act"] = hidden_activity_cum
        Test_Values["kinematics_hand"] = kinematics_hand_cum
        Test_Values["kinematics_target"] = kinematics_target_cum
        Test_Values["episode_reward"] = episode_reward
        
        np.save(f'test_hidden_act_{save_name}.npy', Test_Values['hidden_act'])
        np.save(f'test_kinematics_hand_{save_name}.npy', Test_Values['kinematics_hand'])
        np.save(f'test_kinematics_target_{save_name}.npy', Test_Values['kinematics_target'])
        np.save(f'test_episode_reward_{save_name}.npy', Test_Values['episode_reward'])

    def train(self):

        """ Train an RNN based SAC agent to follow kinematic trajectory
        """

        ### TRAINING DATA DICTIONARY ###
        Statistics = {
            "rewards": [],
            "steps": [],
            "policy_loss": [],
            "critic_loss": []
        }

        highest_reward = -float("inf") # used for storing highest reward throughout training

        #Average reward across conditions initialization
        cond_train_count= np.ones((self.env.n_exp_conds,))
        cond_avg_reward = np.zeros((self.env.n_exp_conds,))
        cond_cum_reward = np.zeros((self.env.n_exp_conds,))
        cond_cum_count = np.zeros((self.env.n_exp_conds,))

        ### BEGIN TRAINING ###
        for episode in range(self.episodes):

            ### Gather Episode Data Variables ###
            episode_reward = 0          # reward for single episode
            episode_steps = 0           # steps completed for single episode
            policy_loss_tracker = []    # stores policy loss throughout episode
            critic1_loss_tracker = []   # stores critic loss throughout episode
            done = False                # determines if episode is terminated

            ### GET INITAL STATE + RESET MODEL BY POSE
            if self.condition_selection_strategy != "reward":
                cond_to_select = episode % self.env.n_exp_conds
                state = self.env.reset(cond_to_select)

            else:

                cond_indx = np.nonzero(cond_train_count>0)[0][0]
                cond_train_count[cond_indx] = cond_train_count[cond_indx] - 1
                state = self.env.reset(cond_indx)

            #Append the high-level task scalar signal
            state = [*state, self.env.condition_scalar]

            ep_trajectory = []  # used to store (s_t, a_t, r_t, s_t+1) tuple for replay storage

            h_prev = torch.zeros(size=(1, 1, self.hidden_size)) # num_layers specified in the policy model

            ### LOOP THROUGH EPISODE TIMESTEPS ###
            for t in range(self.env._max_episode_steps):

                ### SELECT ACTION ###
                with torch.no_grad():
                    action, h_current, _ = self.agent.select_action(state, h_prev, evaluate=False)
                    
                    #Now query the neural activity idx from the simulator
                    na_idx= self.env.coord_idx

                ### UPDATE MODEL PARAMETERS ###
                if len(self.policy_memory.buffer) > self.policy_batch_size:
                    for _ in range(self.batch_iters):
                        critic_1_loss, critic_2_loss, policy_loss = self.agent.update_parameters(self.policy_memory, self.policy_batch_size)
                        ### STORE LOSSES ###
                        policy_loss_tracker.append(policy_loss)
                        critic1_loss_tracker.append(critic_1_loss)

                ### SIMULATION ###
                reward = 0
                for _ in range(self.env.frame_repeat):
                    next_state, inter_reward, done, _ = self.env.step(action)
                    next_state = [*next_state, self.env.condition_scalar]

                    reward += inter_reward
                    episode_steps += 1

                episode_reward += reward

                ### VISUALIZE MODEL ###
                if self.visualize == True:
                    self.env.render()

                mask = 1 if episode_steps == self.env._max_episode_steps else float(not done) # ensure mask is not 0 if episode ends

                ### STORE CURRENT TIMESTEP TUPLE ###
                ep_trajectory.append((state, 
                                        action, 
                                        reward, 
                                        next_state, 
                                        mask,
                                        h_current.squeeze(0).cpu().numpy(),
                                        self.env.neural_activity[na_idx],
                                        np.array([na_idx])))

                ### MOVE TO NEXT STATE ###
                state = next_state
                h_prev = h_current

                ### EARLY TERMINATION OF EPISODE ###
                if done:
                    break
            
            ### PUSH TO REPLAY ###
            self.policy_memory.push(ep_trajectory)

            if self.condition_selection_strategy == "reward":

                cond_cum_reward[cond_indx] = cond_cum_reward[cond_indx] + episode_reward
                cond_cum_count[cond_indx] = cond_cum_count[cond_indx] + 1

                #Check if there are all zeros in the cond_train_count array
                if np.all((cond_train_count == 0)):
                    cond_avg_reward = cond_cum_reward / cond_cum_count
                    cond_train_count = np.ceil((np.max(cond_avg_reward)*np.ones((self.env.n_exp_conds,)))/cond_avg_reward)

            ### TRACKING ###
            Statistics["rewards"].append(episode_reward)
            Statistics["steps"].append(episode_steps)
            Statistics["policy_loss"].append(np.mean(np.array(policy_loss_tracker)))
            Statistics["critic_loss"].append(np.mean(np.array(critic1_loss_tracker)))

            ### SAVE DATA TO FILE (in root project folder) ###
            if len(self.root_dir) != 0 and len(self.checkpoint_folder) != 0 and len(self.checkpoint_file) != 0:
                np.save(f'rewards_{self.checkpoint_file}.npy', Statistics['rewards'])
                np.save(f'steps_{self.checkpoint_file}.npy', Statistics['steps'])
                np.save(f'policy_loss_{self.checkpoint_file}.npy', Statistics['policy_loss'])
                np.save(f'critic_loss_{self.checkpoint_file}.npy', Statistics['critic_loss'])

            if episode_reward > highest_reward:
                highest_reward = episode_reward

            ### SAVING STATE DICT OF TRAINING ###
            if len(self.root_dir) != 0 and len(self.checkpoint_folder) != 0 and len(self.checkpoint_file) != 0:
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
                if episode_reward > highest_reward:
                    torch.save({
                            'iteration': episode,
                            'agent_state_dict': self.agent.actor.state_dict(),
                            'critic_state_dict': self.agent.critic.state_dict(),
                            'critic_target_state_dict': self.agent.critic_target.state_dict(),
                            'agent_optimizer_state_dict': self.agent.actor_optim.state_dict(),
                            'critic_optimizer_state_dict': self.agent.critic_optim.state_dict(),
                        }, f + '_best.pth')

            ### PRINT TRAINING OUTPUT ###
            print('-----------------------------------')
            print('highest reward: {} | reward: {} | timesteps completed: {}'.format(highest_reward, episode_reward, episode_steps))
            print('-----------------------------------\n')
