import numpy as np
import torch
from SAC.sac import SAC_Agent, DDPG_Agent, TD3_Agent
from SAC.replay_memory import PolicyReplayMemory
from SAC.RL_Framework_Mujoco import Muscle_Env
from SAC import sensory_feedback_specs, kinematics_preprocessing_specs, perturbation_specs
import pickle
import os
from numpy.core.records import fromarrays
from scipy.io import savemat


#Set the current working directory
os.chdir(os.getcwd())

class Simulate():

    def __init__(self, env:Muscle_Env, args):

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

        ### TRAINING VARIABLES ###
        self.episodes = args.total_episodes
        self.hidden_size = args.hidden_size
        self.policy_batch_size = args.policy_batch_size
        self.visualize = args.visualize
        self.root_dir = args.root_dir
        self.checkpoint_file = args.checkpoint_file
        self.checkpoint_folder = args.checkpoint_folder
        self.statistics_folder = args.statistics_folder
        self.batch_iters = args.batch_iters
        self.save_iter = args.save_iter
        self.mode_to_sim = args.mode
        self.condition_selection_strategy = args.condition_selection_strategy
        self.load_saved_nets_for_training = args.load_saved_nets_for_training
        self.verbose_training = args.verbose_training

        ### ENSURE SAVING FILES ARE ACCURATE ###
        assert isinstance(self.root_dir, str)
        assert isinstance(self.checkpoint_folder, str)
        assert isinstance(self.checkpoint_file, str)

        ### SEED ###
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)


        ### LOAD CUSTOM GYM ENVIRONMENT ###
        if self.mode_to_sim in ["musculo_properties"]:
            self.env = env(args.musculoskeletal_model_path[:-len('musculoskeletal_model.xml')] + 'musculo_targets_pert.xml', 1, args)

        else:
            self.env = env(args.musculoskeletal_model_path[:-len('musculoskeletal_model.xml')] + 'musculo_targets.xml', 1, args)

        self.observation_shape = self.env.observation_space.shape[0]+len(self.env.sfs_visual_velocity)*3+1

        ### INITIATING AGENT ###
        if args.RL_algorithm == "SAC":
            self.agent = SAC_Agent(self.observation_shape,
                                   self.env.action_space,
                                   args.hidden_size,
                                   args.lr,
                                   args.gamma,
                                   args.tau,
                                   args.model,
                                   args.multi_policy_loss,
                                   args.alpha_usim,
                                   args.beta_usim,
                                   args.gamma_usim,
                                   args.zeta_nusim,
                                   args.cuda,
                                   args.alpha,
                                   args.automatic_entropy_tuning)
        elif args.RL_algorithm == "DDPG":
            self.agent = DDPG_Agent(self.observation_shape,
                                   self.env.action_space,
                                   args.hidden_size,
                                   args.lr,
                                   args.gamma,
                                   args.tau,
                                   args.model,
                                   args.multi_policy_loss,
                                   args.alpha_usim,
                                   args.beta_usim,
                                   args.gamma_usim,
                                   args.zeta_nusim,
                                   args.cuda)
        elif args.RL_algorithm == "TD3":
            self.agent = TD3_Agent(self.observation_shape,
                                   self.env.action_space,
                                   args.hidden_size,
                                   args.lr,
                                   args.gamma,
                                   args.tau,
                                   args.model,
                                   args.multi_policy_loss,
                                   args.alpha_usim,
                                   args.beta_usim,
                                   args.gamma_usim,
                                   args.zeta_nusim,
                                   args.cuda,
                                   args.target_noise,
                                   args.target_noise_clip,
                                   args.policy_delay)
        else:
            raise NotImplementedError



        ### REPLAY MEMORY ###
        self.policy_memory = PolicyReplayMemory(args.policy_replay_size, args.seed)


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
        self.load_saved_nets_from_checkpoint(load_best = True)

        #Set the recurrent connections to zero if the mode is SFE
        if self.mode_to_sim in ["SFE"] and "recurrent_connections" in perturbation_specs.sf_elim:
            self.agent.actor.rnn.weight_hh_l0 = torch.nn.Parameter(self.agent.actor.rnn.weight_hh_l0 * 0)

        ### TESTING DATA ###
        Test_Data = {
            "emg": {},
            "rnn_activity": {},
            "rnn_input": {},
            "rnn_input_fp": {},
            "kinematics_mbodies": {},
            "kinematics_mtargets": {}
        }

        ###Data jPCA
        activity_jpca = []
        times_jpca = []
        n_fixedsteps_jpca = []
        condition_tpoints_jpca = []

        #Save the following after testing: EMG, RNN_activity, RNN_input, kin_musculo_bodies, kin_musculo_targets
        emg = {}
        rnn_activity = {}
        rnn_input = {}
        kin_mb = {}
        kin_mt = {}
        rnn_input_fp = {}

        for i_cond_sim in range(self.env.n_exp_conds):

            ### TRACKING VARIABLES ###
            episode_reward = 0
            episode_steps = 0 

            emg_cond = []
            kin_mb_cond = []
            kin_mt_cond = []
            rnn_activity_cond = []
            rnn_input_cond = []
            rnn_input_fp_cond = []

            done = False

            ### GET INITAL STATE + RESET MODEL BY POSE
            cond_to_select = i_cond_sim % self.env.n_exp_conds
            state = self.env.reset(cond_to_select)

            if self.mode_to_sim in ["SFE"] and "task_scalar" in perturbation_specs.sf_elim:
                state = [*state, 0]
            else:
                state = [*state, self.env.condition_scalar]

            # Num_layers specified in the policy model 
            h_prev = torch.zeros(size=(1, 1, self.hidden_size))

            ### STEPS PER EPISODE ###
            for timestep in range(self.env.timestep_limit):

                ### SELECT ACTION ###
                with torch.no_grad():

                    if self.mode_to_sim in ["neural_pert"]:
                        state = torch.FloatTensor(state).to(self.agent.device).unsqueeze(0).unsqueeze(0)
                        h_prev = h_prev.to(self.agent.device)
                        neural_pert = perturbation_specs.neural_pert[timestep % perturbation_specs.neural_pert.shape[0], :]
                        neural_pert = torch.FloatTensor(neural_pert).to(self.agent.device).unsqueeze(0).unsqueeze(0) 
                        action, h_current, rnn_act, rnn_in = self.agent.actor.forward_for_neural_pert(state, h_prev, neural_pert)

                    elif self.mode_to_sim in ["SFE"] and "recurrent_connections" in perturbation_specs.sf_elim:
                        h_prev = h_prev*0
                        action, h_current, rnn_act, rnn_in = self.agent.select_action(state, h_prev, evaluate=True)
                    else:
                        action, h_current, rnn_act, rnn_in = self.agent.select_action(state, h_prev, evaluate=True)

                    
                    emg_cond.append(action) #[n_muscles, ]
                    rnn_activity_cond.append(rnn_act[0, :])  # [1, n_hidden_units] --> [n_hidden_units,]
                    rnn_input_cond.append(state) #[n_inputs, ]
                    rnn_input_fp_cond.append(rnn_in[0, 0, :])  #[1, 1, n_hidden_units] --> [n_hidden_units, ]


                ### TRACKING REWARD + EXPERIENCE TUPLE###
                next_state, reward, done, _ = self.env.step(action)
                
                if self.mode_to_sim in ["SFE"] and "task_scalar" in perturbation_specs.sf_elim:
                    next_state = [*next_state, 0]
                else:
                    next_state = [*next_state, self.env.condition_scalar]

                episode_reward += reward

                #now append the kinematics of the musculo body and the corresponding target
                kin_mb_t = []
                kin_mt_t = []
                for musculo_body in kinematics_preprocessing_specs.musculo_tracking:
                    kin_mb_t.append(self.env.data.xpos[self.env.model.body(musculo_body[0]).id].copy())   #[3, ]
                    kin_mt_t.append(self.env.data.xpos[self.env.model.body(musculo_body[1]).id].copy()) #[3, ]

                kin_mb_cond.append(kin_mb_t)   # kin_mb_t : [n_targets, 3]
                kin_mt_cond.append(kin_mt_t)   # kin_mt_t : [n_targets, 3]

                ### VISUALIZE MODEL ###
                if self.visualize == True:
                    self.env.render()

                state = next_state
                h_prev = h_current
            
            #Append the testing data
            emg[i_cond_sim] = np.array(emg_cond)  # [timepoints, muscles]
            rnn_activity[i_cond_sim] = np.array(rnn_activity_cond) # [timepoints, n_hidden_units]
            rnn_input[i_cond_sim] = np.array(rnn_input_cond)  #[timepoints, n_inputs]
            rnn_input_fp[i_cond_sim] = np.array(rnn_input_fp_cond)  #[timepoints, n_hidden_units]
            kin_mb[i_cond_sim] = np.array(kin_mb_cond).transpose(1, 0, 2)   # kin_mb_cond: [timepoints, n_targets, 3] --> [n_targets, timepoints, 3]
            kin_mt[i_cond_sim] = np.array(kin_mt_cond).transpose(1, 0, 2)   # kin_mt_cond: [timepoints, n_targets, 3] --> [n_targets, timepoints, 3]

            ##Append the jpca data
            activity_jpca.append(dict(A = rnn_activity_cond))
            times_jpca.append(dict(times = np.arange(timestep)))    #the timestep is assumed to be 1ms
            condition_tpoints_jpca.append(self.env.kin_to_sim[self.env.current_cond_to_sim].shape[-1])
            n_fixedsteps_jpca.append(self.env.n_fixedsteps)

        ### SAVE TESTING STATS ###
        Test_Data["emg"] = emg
        Test_Data["rnn_activity"] = rnn_activity
        Test_Data["rnn_input"] = rnn_input
        Test_Data["rnn_input_fp"] = rnn_input_fp
        Test_Data["kinematics_mbodies"] = kin_mb
        Test_Data["kinematics_mtargets"] = kin_mt

        ### Save the jPCA data
        Data_jpca = fromarrays([activity_jpca, times_jpca], names=['A', 'times'])

        #save test data
        with open(save_name + '/test_data.pkl', 'wb') as f:
            pickle.dump(Test_Data, f)

        #save jpca data
        savemat(save_name + '/Data_jpca.mat', {'Data' : Data_jpca})
        savemat(save_name + '/n_fixedsteps_jpca.mat', {'n_fsteps' : n_fixedsteps_jpca})
        savemat(save_name + '/condition_tpoints_jpca.mat', {'cond_tpoints': condition_tpoints_jpca})
        

    def train(self):

        """ Train an RNN based SAC agent to follow kinematic trajectory
        """

        #Load the saved networks from the last training
        if self.load_saved_nets_for_training:
            self.load_saved_nets_from_checkpoint(load_best= False)

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
            while not(done):

                ### SELECT ACTION ###
                with torch.no_grad():
                    action, h_current, _, _ = self.agent.select_action(state, h_prev, evaluate=False)
                    
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

                    ### EARLY TERMINATION OF EPISODE ###
                    if done:
                        break

                episode_reward += reward

                ### VISUALIZE MODEL ###
                if self.visualize == True:
                    self.env.render()

                mask = 1 if episode_steps == self.env._max_episode_steps else float(not done) # ensure mask is not 0 if episode ends

                ### STORE CURRENT TIMESTEP TUPLE ###
                if not self.env.nusim_data_exists:
                    self.env.neural_activity[na_idx] = 0

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
            if policy_loss_tracker == [] and critic1_loss_tracker == []:
                Statistics["policy_loss"].append(np.nan)
                Statistics["critic_loss"].append(np.nan)
            else:
                Statistics["policy_loss"].append(np.mean(np.array(policy_loss_tracker)))
                Statistics["critic_loss"].append(np.mean(np.array(critic1_loss_tracker)))


            ### SAVE DATA TO FILE (in root project folder) ###
            if len(self.statistics_folder) != 0:
                np.save(self.statistics_folder + f'/stats_rewards.npy', Statistics['rewards'])
                np.save(self.statistics_folder + f'/stats_steps.npy', Statistics['steps'])
                np.save(self.statistics_folder + f'/stats_policy_loss.npy', Statistics['policy_loss'])
                np.save(self.statistics_folder + f'/stats_critic_loss.npy', Statistics['critic_loss'])


            ### SAVING STATE DICT OF TRAINING ###
            if len(self.checkpoint_folder) != 0 and len(self.checkpoint_file) != 0:
                if episode % self.save_iter == 0 and len(self.policy_memory.buffer) > self.policy_batch_size:
                    
                    #Save the state dicts
                    torch.save({
                         'iteration': episode,
                         'agent_state_dict': self.agent.actor.state_dict(),
                         'critic_state_dict': self.agent.critic.state_dict(),
                         'critic_target_state_dict': self.agent.critic_target.state_dict(),
                         'agent_optimizer_state_dict': self.agent.actor_optim.state_dict(),
                         'critic_optimizer_state_dict': self.agent.critic_optim.state_dict(),
                     }, self.checkpoint_folder + f'/{self.checkpoint_file}.pth')

                    #Save the pickled model for fixedpoint finder analysis
                    torch.save(self.agent.actor.rnn, self.checkpoint_folder + f'/actor_rnn_fpf.pth')

                
                if episode_reward > highest_reward:
                    torch.save({
                            'iteration': episode,
                            'agent_state_dict': self.agent.actor.state_dict(),
                            'critic_state_dict': self.agent.critic.state_dict(),
                            'critic_target_state_dict': self.agent.critic_target.state_dict(),
                            'agent_optimizer_state_dict': self.agent.actor_optim.state_dict(),
                            'critic_optimizer_state_dict': self.agent.critic_optim.state_dict(),
                        }, self.checkpoint_folder + f'/{self.checkpoint_file}_best.pth')

                    #Save the pickled model for fixedpoint finder analysis
                    torch.save(self.agent.actor.rnn, self.checkpoint_folder + f'/actor_rnn_best_fpf.pth')

            if episode_reward > highest_reward:
                highest_reward = episode_reward

            ### PRINT TRAINING OUTPUT ###
            if self.verbose_training:
                print('-----------------------------------')
                print('highest reward: {} | reward: {} | timesteps completed: {}'.format(highest_reward, episode_reward, episode_steps))
                print('-----------------------------------\n')

    def load_saved_nets_from_checkpoint(self, load_best: bool):

        #Load the saved networks from the checkpoint file
        #Saved networks include policy, critic, critic_target, policy_optimizer and critic_optimizer

        if not load_best:

            #Load the policy network
            self.agent.actor.load_state_dict(torch.load(self.checkpoint_folder + f'/{self.checkpoint_file}.pth')['agent_state_dict'])

            #Load the critic network
            self.agent.critic.load_state_dict(torch.load(self.checkpoint_folder + f'/{self.checkpoint_file}.pth')['critic_state_dict'])

            #Load the critic target network
            self.agent.critic_target.load_state_dict(torch.load(self.checkpoint_folder + f'/{self.checkpoint_file}.pth')['critic_target_state_dict'])

            #Load the policy optimizer 
            self.agent.actor_optim.load_state_dict(torch.load(self.checkpoint_folder + f'/{self.checkpoint_file}.pth')['agent_optimizer_state_dict'])

            #Load the critic optimizer
            self.agent.critic_optim.load_state_dict(torch.load(self.checkpoint_folder + f'/{self.checkpoint_file}.pth')['critic_optimizer_state_dict'])

        else:

            #Load the policy network
            self.agent.actor.load_state_dict(torch.load(self.checkpoint_folder + f'/{self.checkpoint_file}_best.pth')['agent_state_dict'])

            #Load the critic network
            self.agent.critic.load_state_dict(torch.load(self.checkpoint_folder + f'/{self.checkpoint_file}_best.pth')['critic_state_dict'])

            #Load the critic target network
            self.agent.critic_target.load_state_dict(torch.load(self.checkpoint_folder + f'/{self.checkpoint_file}_best.pth')['critic_target_state_dict'])

            #Load the policy optimizer 
            self.agent.actor_optim.load_state_dict(torch.load(self.checkpoint_folder + f'/{self.checkpoint_file}_best.pth')['agent_optimizer_state_dict'])

            #Load the critic optimizer
            self.agent.critic_optim.load_state_dict(torch.load(self.checkpoint_folder + f'/{self.checkpoint_file}_best.pth')['critic_optimizer_state_dict'])
