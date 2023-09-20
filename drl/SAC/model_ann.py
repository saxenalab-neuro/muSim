import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import numpy as np

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Define Policy SNN Network
class PolicyANN(nn.Module):
    def __init__(self, num_inputs=45, num_outputs=18, num_hidden=512, num_steps=25, beta=.75):
        super(PolicyANN, self).__init__()
        self.num_steps = num_steps
        self.action_scale = .5
        self.action_bias = .5

        # initialize layers
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden)

        self.mean_linear = nn.Linear(num_hidden, num_outputs)
        self.mean_decoder = nn.Linear(num_outputs, num_outputs)

        self.log_std_linear = nn.Linear(num_hidden, num_outputs)
        self.log_std_decoder = nn.Linear(num_outputs, num_outputs)

    def forward(self, x):

        # time-loop
        cur1 = self.fc1(x)
        cur2 = self.fc2(cur1)

        cur_mean = self.mean_linear(cur2)
        cur_std = self.log_std_linear(cur2)

        return cur_mean, cur_std
    
    def sample(self, state, sampling, len_seq=None):

        if sampling == True:
            state = state.unsqueeze(0)

        mean, std = self.forward(state) 

        std = std.exp()

        # white noise
        normal = Normal(mean, std)
        noise = normal.rsample()

        y_t = torch.tanh(noise) # reparameterization trick
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(noise)
        # Enforce the action_bounds
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias

        return action, log_prob, mean

# Define critic SNN Network
class CriticANN(nn.Module):
    def __init__(self, num_inputs=63, num_outputs=18, num_hidden=512, num_steps=25, beta=.9):
        super(CriticANN, self).__init__()
        self.num_steps = num_steps

        # QNet 1
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_outputs)

        # QNet 2
        self.fc1_2 = nn.Linear(num_inputs, num_hidden)
        self.fc2_2 = nn.Linear(num_hidden, num_outputs)

    def forward(self, state, action):

        x = torch.cat([state, action], dim=-1)

        out = self.fc1(x)
        q1 = self.fc2(out)

        out = self.fc1_2(x)
        q2 = self.fc2_2(out)

        return q1, q2