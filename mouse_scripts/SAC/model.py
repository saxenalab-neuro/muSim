import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import numpy as np
import colorednoise as cn
import math

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class QNetworkFF(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetworkFF, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):

        xu = torch.cat([state, action], 1)
        
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2

class QNetworkLSTM(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetworkLSTM, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        nn.init.xavier_normal_(self.linear1.weight)
        self.linear2 = nn.Linear(num_inputs + num_actions, hidden_dim)
        nn.init.xavier_normal_(self.linear2.weight)
        self.lstm1 = nn.LSTM(hidden_dim, hidden_dim, num_layers= 1, batch_first= True)
        self.linear3 = nn.Linear(2 * hidden_dim, hidden_dim)
        nn.init.xavier_normal_(self.linear3.weight)
        self.linear4 = nn.Linear(hidden_dim, 1)
        nn.init.xavier_normal_(self.linear4.weight)

        # Q2 architecture
        self.linear5 = nn.Linear(num_inputs + num_actions, hidden_dim)
        nn.init.xavier_normal_(self.linear5.weight)
        self.linear6 = nn.Linear(num_inputs + num_actions, hidden_dim)
        nn.init.xavier_normal_(self.linear6.weight)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, num_layers= 1, batch_first= True)
        self.linear7 = nn.Linear(2 * hidden_dim, hidden_dim)
        nn.init.xavier_normal_(self.linear7.weight)
        self.linear8 = nn.Linear(hidden_dim, 1)
        nn.init.xavier_normal_(self.linear8.weight)

        self.apply(weights_init_)
        # notes: weights_init for the LSTM layer

    def forward(self, state_action_packed, hidden):

        xu = state_action_packed
        xu_p, seq_lens = pad_packed_sequence(xu, batch_first= True)

        fc_branch_1 = F.relu(self.linear1(xu_p))

        lstm_branch_1 = F.relu(self.linear2(xu_p))
        lstm_branch_1 = pack_padded_sequence(lstm_branch_1, seq_lens, batch_first= True, enforce_sorted= False)
        lstm_branch_1, hidden_out_1 = self.lstm1(lstm_branch_1, hidden)
        lstm_branch_1, _ = pad_packed_sequence(lstm_branch_1, batch_first= True)

        x1 = torch.cat([fc_branch_1, lstm_branch_1], dim=-1)
        x1 = F.relu(self.linear3(x1))
        x1 = F.relu(self.linear4(x1))

        fc_branch_2 = F.relu(self.linear5(xu_p))

        lstm_branch_2 = F.relu(self.linear6(xu_p))
        lstm_branch_2 = pack_padded_sequence(lstm_branch_2, seq_lens, batch_first= True, enforce_sorted= False)
        lstm_branch_2, hidden_out_2 = self.lstm2(lstm_branch_2, hidden)
        lstm_branch_2, _ = pad_packed_sequence(lstm_branch_2, batch_first= True)

        x2 = torch.cat([fc_branch_2, lstm_branch_2], dim=-1)
        x2 = F.relu(self.linear7(x2))
        x2 = F.relu(self.linear8(x2))

        return x1, x2

class GaussianPolicyRNN(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicyRNN, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.lstm = nn.RNN(hidden_dim, hidden_dim, batch_first=True)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        # Pass none action space and adjust the action scale and bias manually
        if action_space is None:
            self.action_scale = torch.tensor(0.5)
            self.action_bias = torch.tensor(0.5)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state, h_prev, c_prev, sampling, len_seq= None):

        #x = F.relu(F.tanh(self.linear1(state)))
        #x = F.tanh(self.linear1(state))
        x = F.relu(self.linear1(state))

        if sampling == False:
            assert len_seq!=None, "Proved the len_seq"
            x = pack_padded_sequence(x, len_seq, batch_first= True, enforce_sorted= False)

        x, (h_current) = self.lstm(x, (h_prev))

        if sampling == False:
           x, len_x_seq = pad_packed_sequence(x, batch_first= True)

        if sampling == True:
            x = x.squeeze(1)

        x = F.relu(x)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)

        c_current= torch.tensor(0., requires_grad= True)
        return mean, log_std, h_current, c_current, x

    def sample(self, state, h_prev, c_prev, sampling, len_seq=None):

        mean, log_std, h_current, c_current, x = self.forward(state, h_prev, c_prev, sampling, len_seq)
        #if sampling == False; then mask the mean and log_std using len_seq
        if sampling == False:
            assert mean.size()[1] == log_std.size()[1], "There is a mismatch between and mean and sigma Sl_max"
            sl_max = mean.size()[1]
            with torch.no_grad():
                for seq_idx, k in enumerate(len_seq):
                    for j in range(1, sl_max + 1):
                        if j <= k:
                            if seq_idx == 0 and j == 1:
                                mask_seq = torch.tensor([True], dtype=bool)
                            else:
                                mask_seq = torch.cat((mask_seq, torch.tensor([True])), dim=0)
                        else:
                            mask_seq = torch.cat((mask_seq, torch.tensor([False])), dim=0)
        #The mask has been created, Now filter the mean and sigma using this mask
            print(mask_seq)
            mean = mean.reshape(-1, mean.size()[-1])[mask_seq]
            log_std = log_std.reshape(-1, log_std.size()[-1])[mask_seq]
        if sampling == True:
            mask_seq = [] #If sampling is True return a dummy mask seq

        std = log_std.exp()

        # white noise
        normal = Normal(mean, std)
        noise = normal.rsample()

        # pink noise
        #samples = math.prod(mean.squeeze().shape)
        #noise = cn.powerlaw_psd_gaussian(1, samples)
        #noise = torch.Tensor(noise).view(mean.shape).to(mean.device)

        y_t = torch.tanh(noise) # reparameterization trick
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(noise)
        # Enforce the action_bounds
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias

        return action, log_prob, mean, h_current, c_current, mask_seq, x

    def forward_for_simple_dynamics(self, state, h_prev, c_prev, sampling, len_seq= None):

        #x = F.relu(F.tanh(self.linear1(state)))
        #x = F.tanh(self.linear1(state))
        x = F.relu(self.linear1(state))

        #Tap the output of the first linear layer
        x_l1 = x
        # x = state

        if sampling == False:
            assert len_seq!=None, "Proved the len_seq"

            x = pack_padded_sequence(x, len_seq, batch_first= True, enforce_sorted= False)

        x, (h_current) = self.lstm(x, (h_prev))

        if sampling == False:
           x, len_x_seq = pad_packed_sequence(x, batch_first= True)

        x = F.relu(x)
        return x, x_l1

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicyRNN, self).to(device)

class GaussianPolicyLSTM(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicyLSTM, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        nn.init.xavier_normal_(self.linear1.weight)
        self.lstm = nn.LSTM(num_inputs, hidden_dim, num_layers=1, batch_first=True)
        self.linear2 = nn.Linear(2*hidden_dim, hidden_dim)
        nn.init.xavier_normal_(self.linear2.weight)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)
        # Adjust the initial weights of the recurrent LSTM layer

        # action rescaling
        # Pass none action space and adjust the action scale and bias manually
        if action_space is None:
            # Try different scales to see what works best
            self.action_scale = torch.tensor(0.5)
            self.action_bias = torch.tensor(0.5)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state, h_prev, c_prev, sampling):

        if sampling == True:
            fc_branch = F.relu(self.linear1(state))
            lstm_branch, (h_current, c_current) = self.lstm(state, (h_prev, c_prev))
        else:
            state_pad, _ = pad_packed_sequence(state, batch_first= True)
            fc_branch = F.relu(self.linear1(state_pad))
            lstm_branch, (h_current, c_current) = self.lstm(state, (h_prev, c_prev))
            lstm_branch, seq_lens = pad_packed_sequence(lstm_branch, batch_first= True)

        x = torch.cat([fc_branch, lstm_branch], dim=-1)
        x = F.relu(self.linear2(x))
        
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)

        return mean, log_std, h_current, c_current, lstm_branch

    def sample(self, state, h_prev, c_prev, sampling):

        mean, log_std, h_current, c_current, lstm_branch = self.forward(state, h_prev, c_prev, sampling)
        #if sampling == False; then reshape mean and log_std from (B, L_max, A) to (B*Lmax, A)

        mean_size = mean.size()
        log_std_size = log_std.size()

        mean = mean.reshape(-1, mean.size()[-1])
        log_std = log_std.reshape(-1, log_std.size()[-1])

        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()

        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias

        log_prob = normal.log_prob(x_t)

        # Enforce the action_bounds
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)

        mean = torch.tanh(mean) * self.action_scale + self.action_bias

        if sampling == False:
            action = action.reshape(mean_size[0], mean_size[1], mean_size[2])
            mean = mean.reshape(mean_size[0], mean_size[1], mean_size[2])
            log_prob = log_prob.reshape(log_std_size[0], log_std_size[1], 1) 

        return action, log_prob, mean, h_current, c_current, lstm_branch
    
    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicyLSTM, self).to(device)
