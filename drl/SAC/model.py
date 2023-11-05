import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import numpy as np
import snntorch as snn
from LSNN.lleaky import LLeaky
from LSNN import surrogate

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6


class PolicyLSTM(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(PolicyLSTM, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        nn.init.xavier_normal_(self.linear1.weight)
        self.lstm = nn.LSTM(num_inputs, hidden_dim, num_layers=1, batch_first=True)
        self.linear2 = nn.Linear(2*hidden_dim, hidden_dim)
        nn.init.xavier_normal_(self.linear2.weight)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.action_scale = torch.tensor(0.5)
        self.action_bias = torch.tensor(0.5)

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
        return super(PolicyLSTM, self).to(device)
    

class CriticLSTM(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(CriticLSTM, self).__init__()

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
        x1 = self.linear4(x1)

        fc_branch_2 = F.relu(self.linear5(xu_p))

        lstm_branch_2 = F.relu(self.linear6(xu_p))
        lstm_branch_2 = pack_padded_sequence(lstm_branch_2, seq_lens, batch_first= True, enforce_sorted= False)
        lstm_branch_2, hidden_out_2 = self.lstm2(lstm_branch_2, hidden)
        lstm_branch_2, _ = pad_packed_sequence(lstm_branch_2, batch_first= True)

        x2 = torch.cat([fc_branch_2, lstm_branch_2], dim=-1)
        x2 = F.relu(self.linear7(x2))
        x2 = self.linear8(x2)

        return x1, x2
    

# Define Policy SNN Network
class PolicyANN(nn.Module):
    def __init__(self, num_inputs=45, num_outputs=18, num_hidden=512, deterministic=False):
        super(PolicyANN, self).__init__()
        self.deterministic = deterministic
        self.noise = torch.Tensor(num_outputs)
        self.action_scale = .5
        self.action_bias = .5

        # initialize layers
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.fc3 = nn.Linear(num_hidden, num_hidden)
        self.fc4 = nn.Linear(num_hidden, num_hidden)

        self.mean_linear1 = nn.Linear(num_hidden, num_hidden)
        self.mean_linear2 = nn.Linear(num_hidden, num_outputs)

        self.log_std_linear1 = nn.Linear(num_hidden, num_hidden)
        self.log_std_linear2 = nn.Linear(num_hidden, num_outputs)

    def forward(self, x):

        # time-loop
        cur1 = F.relu(self.fc1(x))
        cur2 = F.relu(self.fc2(cur1))
        cur3 = F.relu(self.fc3(cur2))
        cur4 = F.relu(self.fc4(cur3))

        cur_mean = F.relu(self.mean_linear1(cur4))
        cur_mean = self.mean_linear2(cur_mean)

        cur_std = F.relu(self.log_std_linear1(cur4))
        cur_std = self.log_std_linear2(cur_std)

        cur_std = torch.clamp(cur_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)

        return cur_mean, cur_std
    
    def sample(self, state, sampling, len_seq=None):

        if sampling == True:
            state = state.unsqueeze(0)

        mean, std = self.forward(state) 

        if self.deterministic:
            mean = torch.tanh(mean) * self.action_scale + self.action_bias
            noise = self.noise.normal_(0., std=0.1).to('cuda')
            noise = noise.clamp(-0.25, 0.25)
            action = mean + noise
            return action, torch.tensor(0.), mean

        std = std.exp()
        # white noise
        normal = Normal(mean, std)
        noise = normal.rsample()
        y_t = torch.tanh(noise) # reparameterization trick
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(noise)
        # Enforce the action_bounds
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias

        return action, log_prob, mean


# Define critic SNN Network
class CriticANN(nn.Module):
    def __init__(self, num_inputs=63, action_space=18, num_hidden=512):
        super(CriticANN, self).__init__()

        # QNet 1
        self.fc1 = nn.Linear(num_inputs + action_space, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.fc3 = nn.Linear(num_hidden, num_hidden)
        self.fc4 = nn.Linear(num_hidden, 1)

        # QNet 2
        self.fc1_2 = nn.Linear(num_inputs + action_space, num_hidden)
        self.fc2_2 = nn.Linear(num_hidden, num_hidden)
        self.fc3_2 = nn.Linear(num_hidden, num_hidden)
        self.fc4_2 = nn.Linear(num_hidden, 1)

    def forward(self, state, action):

        x = torch.cat([state, action], dim=-1)

        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        q1 = self.fc4(out)

        out = F.relu(self.fc1_2(x))
        out = F.relu(self.fc2_2(out))
        out = F.relu(self.fc3_2(out))
        q2 = self.fc4_2(out)

        return q1, q2