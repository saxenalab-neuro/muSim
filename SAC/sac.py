import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from .utils import soft_update, hard_update
from .model import Actor, Critic, DoubleCritic
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from .replay_memory import PolicyReplayMemory
import ipdb


class Agent():
    def __init__(self,
                 num_inputs: int,
                 action_space: int,
                 hidden_size: int,
                 lr: float,
                 gamma: float,
                 tau: float,
                 model: str,
                 multi_policy_loss: bool,
                 alpha_usim: float,
                 beta_usim: float,
                 gamma_usim: float,
                 zeta_nusim: float,
                 cuda: bool):

        if cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Save the regularization parameters
        self.alpha_usim = alpha_usim
        self.beta_usim = beta_usim
        self.gamma_usim = gamma_usim
        self.zeta_nusim = zeta_nusim


        ### SET ACTOR NETWORK ###
        self.actor = Actor(num_inputs, action_space.shape[0], hidden_size, model, action_space=None).to(self.device)
        self.actor_optim = Adam(self.actor.parameters(), lr=lr)

        ### SET TRAINING VARIABLES ###
        self.model = model
        self.multi_policy_loss = multi_policy_loss
        self.gamma = gamma
        self.tau = tau
        self.hidden_size = hidden_size


    # This loss encourages the simple low-dimensional dynamics in the RNN activity
    def _policy_loss_2(self, policy_state_batch, h0, len_seq, mask_seq):

        # Sample the hidden weights of the RNN
        J_rnn_w = self.actor.rnn.weight_hh_l0  # These weights would be of the size (hidden_dim, hidden_dim)

        # Sample the output of the RNN for the policy_state_batch
        rnn_out_r, _ = self.actor.forward_for_simple_dynamics(policy_state_batch, h0, sampling=False, len_seq=len_seq)
        rnn_out_r = rnn_out_r.reshape(-1, rnn_out_r.size()[-1])[mask_seq]

        # Reshape the policy hidden weights vector
        J_rnn_w = J_rnn_w.unsqueeze(0).repeat(rnn_out_r.size()[0], 1, 1)
        rnn_out_r = 1 - torch.pow(rnn_out_r, 2)

        R_j = torch.mul(J_rnn_w, rnn_out_r.unsqueeze(-1))

        policy_loss_2 = torch.norm(R_j) ** 2

        return policy_loss_2

    # This loss encourages the minimization of the firing rates for the linear and the RNN layer.
    def _policy_loss_3(self, policy_state_batch, h0, len_seq, mask_seq):

        # Find the loss encouraging the minimization of the firing rates for the linear and the RNN layer
        # Sample the output of the RNN for the policy_state_batch
        rnn_out_r, linear_out = self.actor.forward_for_simple_dynamics(policy_state_batch, h0, sampling=False,
                                                                       len_seq=len_seq)
        rnn_out_r = rnn_out_r.reshape(-1, rnn_out_r.size()[-1])[mask_seq]
        linear_out = linear_out.reshape(-1, linear_out.size()[-1])[mask_seq]

        policy_loss_3 = torch.norm(rnn_out_r) ** 2 + torch.norm(linear_out) ** 2

        return policy_loss_3

    # This loss encourages the minimization of the input and output weights of the RNN and the layers downstream/
    # upstream of the RNN.
    def _policy_loss_4(self):

        # Find the loss encouraging the minimization of the input and output weights of the RNN and the layers downstream
        # and upstream of the RNN
        # Sample the input weights of the RNN
        J_rnn_i = self.actor.rnn.weight_ih_l0
        J_in1 = self.actor.linear1.weight

        # Sample the output weights
        J_out1 = self.actor.mean_linear.weight
        J_out2 = self.actor.log_std_linear.weight

        policy_loss_4 = torch.norm(J_in1) ** 2 + torch.norm(J_rnn_i) ** 2 + torch.norm(J_out1) ** 2 + torch.norm(
            J_out2) ** 2

        return policy_loss_4

    # Define a loss function that constraints a subset of the RNN nodes to the experimental neural data
    def _policy_loss_exp_neural_constrain(self, policy_state_batch, h0, len_seq, neural_activity_batch, na_idx_batch,
                                          mask_seq):
        # Find the loss for neural activity constrainting
        lstm_out = self.actor.forward_lstm(policy_state_batch, h0, sampling=False, len_seq=len_seq)
        lstm_out = lstm_out.reshape(-1, lstm_out.size()[-1])[mask_seq]
        lstm_activity = lstm_out[:, 0:neural_activity_batch.shape[-1]]

        # Now filter the neural activity batch and lstm activity batch using the na_idx batch
        with torch.no_grad():
            na_idx_batch = na_idx_batch.squeeze(-1) > 0

        lstm_activity = lstm_activity[na_idx_batch]
        neural_activity_batch = neural_activity_batch[na_idx_batch]

        policy_loss_exp_c = F.mse_loss(lstm_activity, neural_activity_batch)

        return policy_loss_exp_c

    def select_action(self, state: np.ndarray, h_prev: torch.Tensor, evaluate=False) -> (
    np.ndarray, torch.Tensor, np.ndarray):

        state = torch.FloatTensor(state).to(self.device).unsqueeze(0).unsqueeze(0)
        h_prev = h_prev.to(self.device)

        ### IF TRAINING ###
        if evaluate == False:
            # get action sampled from gaussian
            action, _, _, h_current, _, rnn_out, rnn_in = self.actor.sample(state, h_prev, sampling=True, len_seq=None)
        ### IF TESTING ###
        else:
            # get the action without noise
            _, _, action, h_current, _, rnn_out, rnn_in = self.actor.sample(state, h_prev, sampling=True, len_seq=None)

        return action.detach().cpu().numpy()[
            0], h_current.detach(), rnn_out.detach().cpu().numpy(), rnn_in.detach().cpu().numpy()

class SAC_Agent(Agent):
    def __init__(self, 
                 num_inputs: int, 
                 action_space: int, 
                 hidden_size: int, 
                 lr: float, 
                 gamma: float, 
                 tau: float,
                 model: str, 
                 multi_policy_loss: bool,
                 alpha_usim:float,
                 beta_usim:float,
                 gamma_usim:float,
                 zeta_nusim:float,
                 cuda: bool,
                 alpha: float,
                 automatic_entropy_tuning: bool):

        super().__init__(num_inputs,
                 action_space,
                 hidden_size,
                 lr,
                 gamma,
                 tau,
                 model,
                 multi_policy_loss,
                 alpha_usim,
                 beta_usim,
                 gamma_usim,
                 zeta_nusim,
                 cuda)

        ### SET CRITIC NETWORKS ###
        self.critic = DoubleCritic(num_inputs, action_space.shape[0], hidden_size).to(self.device)
        self.critic_target = DoubleCritic(num_inputs, action_space.shape[0], hidden_size).to(self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=lr)
        hard_update(self.critic_target, self.critic)

        self.alpha = alpha
        self.automatic_entropy_tuning = automatic_entropy_tuning

        # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
        if automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = Adam([self.log_alpha], lr=lr)

    def update_parameters(self, policy_memory: PolicyReplayMemory, policy_batch_size: int) -> (int, int, int):
        ### SAMPLE FROM REPLAY ###
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch, h_batch, policy_state_batch, neural_activity_batch, na_idx_batch = policy_memory.sample(batch_size=policy_batch_size)

        ### CONVERT DATA TO TENSOR ###
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
        h_batch = torch.FloatTensor(h_batch).to(self.device).permute(1, 0, 2)
        neural_activity_batch = torch.FloatTensor(neural_activity_batch).to(self.device)
        na_idx_batch = torch.FloatTensor(na_idx_batch).to(self.device)

        h0 = torch.zeros(size=(1, next_state_batch.shape[0], self.hidden_size)).to(self.device)
        ### SAMPLE NEXT Q VALUE FOR CRITIC LOSS ###
        with torch.no_grad():
            next_state_action, next_state_log_pi, _, _, _, _, _ = self.actor.sample(next_state_batch.unsqueeze(1), h_batch, sampling=True)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)

        ### CALCULATE CRITIC LOSS ###
        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        ### TAKE GRAIDENT STEP ###
        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        ### SAMPLE FROM ACTOR NETWORK ###
        h0 = torch.zeros(size=(1, len(policy_state_batch), self.hidden_size)).to(self.device)
        len_seq = list(map(len, policy_state_batch))
        policy_state_batch = torch.FloatTensor(pad_sequence(policy_state_batch, batch_first= True)).to(self.device)
        pi_action_bat, log_prob_bat, _, _, mask_seq, _, _  = self.actor.sample(policy_state_batch, h0, sampling=False, len_seq=len_seq)

        ### MASK POLICY STATE BATCH ###
        policy_state_batch_pi = policy_state_batch.reshape(-1, policy_state_batch.size()[-1])[mask_seq]

        ### GET VALUE OF CURRENT STATE AND ACTION PAIRS ###
        qf1_pi, qf2_pi = self.critic(policy_state_batch_pi, pi_action_bat)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        ### CALCULATE POLICY LOSS ###
        task_loss = ((self.alpha * log_prob_bat) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        policy_loss = task_loss

        ############################
        # ADDITIONAL POLICY LOSSES #
        ############################

        if self.multi_policy_loss:

            loss_simple_dynamics = self._policy_loss_2(policy_state_batch, h0, len_seq, mask_seq)
            loss_activations_min = self._policy_loss_3(policy_state_batch, h0, len_seq, mask_seq)
            loss_weights_min = self._policy_loss_4()
            loss_exp_constrain = self._policy_loss_exp_neural_constrain(policy_state_batch, h0, len_seq, neural_activity_batch, na_idx_batch, mask_seq)

            ### CALCULATE FINAL POLICY LOSS ###
            #To implement nuSim training use a weighting of 1e+04 with loss_exp_constrain
            policy_loss += (self.alpha_usim*(loss_simple_dynamics))  \
                            + (self.beta_usim*(loss_activations_min))  \
                            + (self.gamma_usim*(loss_weights_min))  \
                            + (self.zeta_nusim*(loss_exp_constrain))

        ### TAKE GRADIENT STEP ###
        self.actor_optim.zero_grad()
        policy_loss.backward()
        self.actor_optim.step()

        ### AUTOMATIC ENTROPY TUNING ###
        log_pi = log_prob_bat
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()

        ### SOFT UPDATE OF CRITIC TARGET ###
        soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item()


class DDPG_Agent(Agent):
    def __init__(self,
                 num_inputs: int,
                 action_space: int,
                 hidden_size: int,
                 lr: float,
                 gamma: float,
                 tau: float,
                 model: str,
                 multi_policy_loss: bool,
                 alpha_usim: float,
                 beta_usim: float,
                 gamma_usim: float,
                 zeta_nusim: float,
                 cuda: bool):

        super().__init__(num_inputs,
                         action_space,
                         hidden_size,
                         lr,
                         gamma,
                         tau,
                         model,
                         multi_policy_loss,
                         alpha_usim,
                         beta_usim,
                         gamma_usim,
                         zeta_nusim,
                         cuda)

        ### SET CRITIC NETWORKS ###
        self.critic = Critic(num_inputs, action_space.shape[0], hidden_size).to(self.device)
        self.critic_target = Critic(num_inputs, action_space.shape[0], hidden_size).to(self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=lr)
        hard_update(self.critic_target, self.critic)

    def update_parameters(self, policy_memory: PolicyReplayMemory, policy_batch_size: int) -> (int, int, int):
        ### SAMPLE FROM REPLAY ###
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch, h_batch, policy_state_batch, neural_activity_batch, na_idx_batch = policy_memory.sample(
            batch_size=policy_batch_size)

        ### CONVERT DATA TO TENSOR ###
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
        h_batch = torch.FloatTensor(h_batch).to(self.device).permute(1, 0, 2)
        neural_activity_batch = torch.FloatTensor(neural_activity_batch).to(self.device)
        na_idx_batch = torch.FloatTensor(na_idx_batch).to(self.device)

        h0 = torch.zeros(size=(1, next_state_batch.shape[0], self.hidden_size)).to(self.device)
        ### SAMPLE NEXT Q VALUE FOR CRITIC LOSS ###
        with torch.no_grad():
            next_state_action, next_state_log_pi, _, _, _, _, _ = self.actor.sample(next_state_batch.unsqueeze(1),
                                                                                    h_batch, sampling=True)
            qf_next_target = self.critic_target(next_state_batch, next_state_action)
            next_q_value = reward_batch + mask_batch * self.gamma * qf_next_target

        ### CALCULATE CRITIC LOSS ###
        qf = self.critic(state_batch,
                               action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf_loss = F.mse_loss(qf, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]

        ### TAKE GRADIENT STEP ###
        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        ### SAMPLE FROM ACTOR NETWORK ###
        h0 = torch.zeros(size=(1, len(policy_state_batch), self.hidden_size)).to(self.device)
        len_seq = list(map(len, policy_state_batch))
        policy_state_batch = torch.FloatTensor(pad_sequence(policy_state_batch, batch_first=True)).to(self.device)
        pi_action_bat, log_prob_bat, _, _, mask_seq, _, _ = self.actor.sample(policy_state_batch, h0, sampling=False,
                                                                              len_seq=len_seq)

        ### MASK POLICY STATE BATCH ###
        policy_state_batch_pi = policy_state_batch.reshape(-1, policy_state_batch.size()[-1])[mask_seq]

        ### GET VALUE OF CURRENT STATE AND ACTION PAIRS ###
        qf_pi = self.critic(policy_state_batch_pi, pi_action_bat)

        ### CALCULATE POLICY LOSS ###
        policy_loss = (-qf_pi).mean()  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        ############################
        # ADDITIONAL POLICY LOSSES #
        ############################

        if self.multi_policy_loss:
            loss_simple_dynamics = self._policy_loss_2(policy_state_batch, h0, len_seq, mask_seq)
            loss_activations_min = self._policy_loss_3(policy_state_batch, h0, len_seq, mask_seq)
            loss_weights_min = self._policy_loss_4()
            loss_exp_constrain = self._policy_loss_exp_neural_constrain(policy_state_batch, h0, len_seq,
                                                                        neural_activity_batch, na_idx_batch, mask_seq)

            ### CALCULATE FINAL POLICY LOSS ###
            # To implement nuSim training use a weighting of 1e+04 with loss_exp_constrain
            policy_loss += (self.alpha_usim * (loss_simple_dynamics)) \
                           + (self.beta_usim * (loss_activations_min)) \
                           + (self.gamma_usim * (loss_weights_min)) \
                           + (self.zeta_nusim * (loss_exp_constrain))

        ### TAKE GRADIENT STEP ###
        self.actor_optim.zero_grad()
        policy_loss.backward()
        self.actor_optim.step()

        ### SOFT UPDATE OF CRITIC TARGET ###
        soft_update(self.critic_target, self.critic, self.tau)

        return qf_loss.item(), 0, policy_loss.item()


class TD3_Agent(Agent):
    def __init__(self,
                 num_inputs: int,
                 action_space: int,
                 hidden_size: int,
                 lr: float,
                 gamma: float,
                 tau: float,
                 model: str,
                 multi_policy_loss: bool,
                 alpha_usim: float,
                 beta_usim: float,
                 gamma_usim: float,
                 zeta_nusim: float,
                 cuda: bool,
                 target_noise: float,
                 noise_clip: float,
                 policy_delay: int):

        super().__init__(num_inputs,
                         action_space,
                         hidden_size,
                         lr,
                         gamma,
                         tau,
                         model,
                         multi_policy_loss,
                         alpha_usim,
                         beta_usim,
                         gamma_usim,
                         zeta_nusim,
                         cuda)

        ### SET CRITIC NETWORKS ###
        self.critic = DoubleCritic(num_inputs, action_space.shape[0], hidden_size).to(self.device)
        self.critic_target = DoubleCritic(num_inputs, action_space.shape[0], hidden_size).to(self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=lr)
        hard_update(self.critic_target, self.critic)

        self.target_noise = target_noise
        self.noise_clip = noise_clip
        self.lowest_action = torch.from_numpy(action_space.low)
        self.highest_action = torch.from_numpy(action_space.high)
        self.policy_delay = policy_delay

        self.normal = torch.distributions.Normal(0, target_noise)
        self.iteration = 0

    def update_parameters(self, policy_memory: PolicyReplayMemory, policy_batch_size: int) -> (int, int, int):
        ### SAMPLE FROM REPLAY ###
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch, h_batch, policy_state_batch, neural_activity_batch, na_idx_batch = policy_memory.sample(
            batch_size=policy_batch_size)

        ### CONVERT DATA TO TENSOR ###
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
        h_batch = torch.FloatTensor(h_batch).to(self.device).permute(1, 0, 2)
        neural_activity_batch = torch.FloatTensor(neural_activity_batch).to(self.device)
        na_idx_batch = torch.FloatTensor(na_idx_batch).to(self.device)

        h0 = torch.zeros(size=(1, next_state_batch.shape[0], self.hidden_size)).to(self.device)
        ### SAMPLE NEXT Q VALUE FOR CRITIC LOSS ###
        with torch.no_grad():
            next_state_action, next_state_log_pi, _, _, _, _, _ = self.actor.sample(next_state_batch.unsqueeze(1),
                                                                                    h_batch, sampling=True)
            next_state_action = torch.clip(next_state_action + torch.clip(self.normal.sample(next_state_action.shape), -self.noise_clip, self.noise_clip), self.lowest_action, self.highest_action)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)

        ### CALCULATE CRITIC LOSS ###
        qf1, qf2 = self.critic(state_batch,
                               action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        ### TAKE GRAIDENT STEP ###
        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        if self.iteration % self.policy_delay == 0:
            ### SAMPLE FROM ACTOR NETWORK ###
            h0 = torch.zeros(size=(1, len(policy_state_batch), self.hidden_size)).to(self.device)
            len_seq = list(map(len, policy_state_batch))
            policy_state_batch = torch.FloatTensor(pad_sequence(policy_state_batch, batch_first=True)).to(self.device)
            pi_action_bat, log_prob_bat, _, _, mask_seq, _, _ = self.actor.sample(policy_state_batch, h0, sampling=False,
                                                                                  len_seq=len_seq)

            ### MASK POLICY STATE BATCH ###
            policy_state_batch_pi = policy_state_batch.reshape(-1, policy_state_batch.size()[-1])[mask_seq]

            ### GET VALUE OF CURRENT STATE AND ACTION PAIRS ###
            qf1_pi, qf2_pi = self.critic(policy_state_batch_pi, pi_action_bat)
            min_qf_pi = torch.min(qf1_pi, qf2_pi)

            ### CALCULATE POLICY LOSS ###
            policy_loss = (-min_qf_pi).mean()

            ############################
            # ADDITIONAL POLICY LOSSES #
            ############################

            if self.multi_policy_loss:
                loss_simple_dynamics = self._policy_loss_2(policy_state_batch, h0, len_seq, mask_seq)
                loss_activations_min = self._policy_loss_3(policy_state_batch, h0, len_seq, mask_seq)
                loss_weights_min = self._policy_loss_4()
                loss_exp_constrain = self._policy_loss_exp_neural_constrain(policy_state_batch, h0, len_seq,
                                                                            neural_activity_batch, na_idx_batch, mask_seq)

                ### CALCULATE FINAL POLICY LOSS ###
                # To implement nuSim training use a weighting of 1e+04 with loss_exp_constrain
                policy_loss += (self.alpha_usim * (loss_simple_dynamics)) \
                               + (self.beta_usim * (loss_activations_min)) \
                               + (self.gamma_usim * (loss_weights_min)) \
                               + (self.zeta_nusim * (loss_exp_constrain))

            ### TAKE GRADIENT STEP ###
            self.actor_optim.zero_grad()
            policy_loss.backward()
            self.actor_optim.step()

            self.policy_loss = policy_loss


        ### SOFT UPDATE OF CRITIC TARGET ###
        soft_update(self.critic_target, self.critic, self.tau)

        self.iteration += 1

        return qf1_loss.item(), qf2_loss.item(), self.policy_loss.item()