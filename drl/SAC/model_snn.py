import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import numpy as np
import snntorch as snn
from snntorch import spikegen

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Define Policy SNN Network
class PolicySNN(nn.Module):
    def __init__(self, num_inputs=45, num_outputs=18, num_hidden=512, beta=.95):
        super(PolicySNN, self).__init__()
        self.action_scale = .5
        self.action_bias = .5

        # initialize layers
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.RLeaky(beta=beta, learn_threshold=True, linear_features=num_hidden, threshold=.5)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.lif2 = snn.RLeaky(beta=beta, learn_threshold=True, linear_features=num_hidden, threshold=.5)

        self.mean_linear = nn.Linear(num_hidden, num_hidden)
        self.mean_linear_lif = snn.RLeaky(beta=beta, learn_threshold=True, linear_features=num_hidden, threshold=.5)
        self.mean_decoder = nn.Linear(num_hidden, num_outputs)

        self.log_std_linear = nn.Linear(num_hidden, num_hidden)
        self.log_std_linear_lif = snn.RLeaky(beta=beta, learn_threshold=True, linear_features=num_hidden, threshold=.5)
        self.log_std_decoder = nn.Linear(num_hidden, num_outputs)


    def forward(self, x, spks=None, mems=None):

        next_mem2_rec = {}
        next_spk2_rec = {}

        # time-loop
        cur1 = self.fc1(x)
        spk1, mem1 = self.lif1(cur1, spks['lif1'], mems['lif1'])
        next_mem2_rec['lif1'], next_spk2_rec['lif1'] = mem1, spk1
        cur2 = self.fc2(spk1)
        spk2, mem2 = self.lif2(cur2, spks['lif2'], mems['lif2'])
        next_mem2_rec['lif2'], next_spk2_rec['lif2'] = mem2, spk2

        cur_mean = self.mean_linear(spk2)
        spk_mean, mem_mean = self.mean_linear_lif(cur_mean, spks['mean_linear_lif'], mems['mean_linear_lif'])
        next_mem2_rec['mean_linear_lif'], next_spk2_rec['mean_linear_lif'] = mem_mean, spk_mean

        cur_std = self.log_std_linear(spk2)
        spk_std, mem_std = self.log_std_linear_lif(cur_std, spks['log_std_linear_lif'], mems['log_std_linear_lif'])
        next_mem2_rec['log_std_linear_lif'], next_spk2_rec['log_std_linear_lif'] = mem_std, spk_std

        spk_mean_decoded = self.mean_decoder(spk_mean)
        spk_std_decoded = self.log_std_decoder(spk_std)

        return spk_mean_decoded, spk_std_decoded, next_mem2_rec, next_spk2_rec
    
    def sample(self, state, sampling, spks=None, mem=None, training=False):

        if sampling == True:
            state = state.unsqueeze(0)

        #state = spikegen.rate(state, num_steps=self.num_steps)

        spk_mean_decoded, spk_std_decoded, next_mem2_rec, next_spk2_rec = self.forward(state, spks=spks, mems=mem) 

        spk_std_decoded = spk_std_decoded.exp()

        # white noise
        normal = Normal(spk_mean_decoded, spk_std_decoded)
        noise = normal.rsample()

        y_t = torch.tanh(noise) # reparameterization trick
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(noise)
        # Enforce the action_bounds
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)
        mean = torch.tanh(spk_mean_decoded) * self.action_scale + self.action_bias
        
        if training:
            action = action.squeeze()
            log_prob = log_prob.squeeze(-1)

        return action, log_prob, mean, next_mem2_rec, next_spk2_rec

# Define critic SNN Network
class CriticSNN(nn.Module):
    def __init__(self, num_inputs=63, num_outputs=18, num_hidden=512, num_steps=10, beta=.95):
        super(CriticSNN, self).__init__()
        self.num_steps = num_steps

        # QNet 1
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.RLeaky(beta=beta, learn_threshold=True, linear_features=num_hidden, threshold=.5)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.lif2 = snn.RLeaky(beta=beta, learn_threshold=True, linear_features=num_hidden, threshold=.5)
        self.output_decoder_1 = nn.Linear(num_hidden, 1)

        # QNet 2
        self.fc1_2 = nn.Linear(num_inputs, num_hidden)
        self.lif1_2 = snn.RLeaky(beta=beta, learn_threshold=True, linear_features=num_hidden, threshold=.5)
        self.fc2_2 = nn.Linear(num_hidden, num_hidden)
        self.lif2_2 = snn.RLeaky(beta=beta, learn_threshold=True, linear_features=num_hidden, threshold=.5)
        self.output_decoder_2 = nn.Linear(num_hidden, 1)

    def forward(self, state, action, spk, mem, training=False):

        x = torch.cat([state, action], dim=-1)
    
        #-------------------------------------

        # Q1
        cur1 = self.fc1(x)
        spk['lif1'], mem['lif1'] = self.lif1(cur1, spk['lif1'], mem['lif1'])
        cur2 = self.fc2(spk['lif1'])
        spk['lif2'], mem['lif2'] = self.lif2(cur2, spk['lif2'], mem['lif2'])

        # Q Network 1 firing rate 
        q1_decoded = self.output_decoder_1(spk['lif2'])

        #---------------------------------------------------------

        # Q2
        cur1 = self.fc1_2(x)
        spk['lif1_2'], mem['lif1_2'] = self.lif1_2(cur1, spk['lif1_2'], mem['lif1_2'])
        cur2 = self.fc2_2(spk['lif1_2'])
        spk['lif2_2'], mem['lif2_2'] = self.lif2_2(cur2, spk['lif2_2'], mem['lif2_2'])

        q2_decoded = self.output_decoder_2(spk['lif2_2'])

        return q1_decoded, q2_decoded, mem, spk