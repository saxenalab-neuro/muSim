import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import numpy as np
import snntorch as snn
from snntorch import spikegen, surrogate

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Define Policy SNN Network
class PolicySNN(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hidden, beta=.95):
        super(PolicySNN, self).__init__()
        self.action_scale = .5
        self.action_bias = .5

        # initialize layers
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.atan(alpha=2))
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=surrogate.atan(alpha=2))

        self.mean_linear = nn.Linear(num_hidden, num_hidden)
        self.mean_linear_lif = snn.Leaky(beta=beta, spike_grad=surrogate.atan(alpha=2))
        self.mean_decoder = nn.Linear(num_hidden, num_outputs)

        self.log_std_linear = nn.Linear(num_hidden, num_hidden)
        self.log_std_linear_lif = snn.Leaky(beta=beta, spike_grad=surrogate.atan(alpha=2))
        self.log_std_decoder = nn.Linear(num_hidden, num_outputs)


    def forward(self, x, mems=None):

        # time-loop
        cur1 = self.fc1(x)
        spk1, mems['lif1'] = self.lif1(cur1, mems['lif1'])
        cur2 = self.fc2(spk1)
        spk2, mems['lif2'] = self.lif2(cur2, mems['lif2'])

        cur_mean = self.mean_linear(spk2)
        spk_mean, mems['mean_linear_lif'] = self.mean_linear_lif(cur_mean, mems['mean_linear_lif'])

        cur_std = self.log_std_linear(spk2)
        spk_std, mems['log_std_linear_lif'] = self.log_std_linear_lif(cur_std, mems['log_std_linear_lif'])

        spk_mean_decoded = self.mean_decoder(spk_mean)
        spk_std_decoded = self.log_std_decoder(spk_std)
        spk_std_decoded = torch.clamp(spk_std_decoded, min=LOG_SIG_MIN, max=LOG_SIG_MAX)

        return spk_mean_decoded, spk_std_decoded, mems
    
    def sample(self, state, sampling, mem=None, training=False):

        if sampling == True:
            state = state.unsqueeze(0)

        spk_mean_decoded, spk_std_decoded, next_mem2_rec = self.forward(state, mems=mem) 

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

        return action, log_prob, mean, next_mem2_rec


# Define critic SNN Network
class CriticSNN(nn.Module):
    def __init__(self, num_inputs=63, num_outputs=18, num_hidden=512, num_steps=10, beta=.95):
        super(CriticSNN, self).__init__()
        self.num_steps = num_steps

        # QNet 1
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.lif2 = snn.Leaky(beta=beta)
        self.output_decoder_1 = nn.Linear(num_hidden, 1)

        # QNet 2
        self.fc1_2 = nn.Linear(num_inputs, num_hidden)
        self.lif1_2 = snn.Leaky(beta=beta)
        self.fc2_2 = nn.Linear(num_hidden, num_hidden)
        self.lif2_2 = snn.Leaky(beta=beta)
        self.output_decoder_2 = nn.Linear(num_hidden, 1)

    def forward(self, state, action, mem, training=False):

        x = torch.cat([state, action], dim=-1)
    
        #-------------------------------------

        # Q1
        cur1 = self.fc1(x)
        spk1, mem['lif1'] = self.lif1(cur1, mem['lif1'])
        cur2 = self.fc2(spk1)
        spk2, mem['lif2'] = self.lif2(cur2, mem['lif2'])

        # Q Network 1 firing rate 
        q1_decoded = self.output_decoder_1(spk2)

        #---------------------------------------------------------

        # Q2
        cur1 = self.fc1_2(x)
        spk1, mem['lif1_2'] = self.lif1_2(cur1, mem['lif1_2'])
        cur2 = self.fc2_2(spk1)
        spk2, mem['lif2_2'] = self.lif2_2(cur2, mem['lif2_2'])

        q2_decoded = self.output_decoder_2(spk2)

        return q1_decoded, q2_decoded, mem


# Define Policy SNN Network
class PolicyRSNN(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hidden, beta=.95):
        super(PolicyRSNN, self).__init__()
        self.action_scale = .5
        self.action_bias = .5
        spike_grad = surrogate.custom_surrogate(self.custom_grad)
        #spike_grad = surrogate.atan()

        # initialize layers
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out')
        self.lif1 = snn.RLeaky(beta=beta, linear_features=num_hidden, spike_grad=spike_grad)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_out')
        self.lif2 = snn.RLeaky(beta=beta, linear_features=num_hidden, spike_grad=spike_grad)

        self.mean_linear = nn.Linear(num_hidden, num_hidden)
        nn.init.kaiming_normal_(self.mean_linear.weight, mode='fan_out')
        self.mean_linear_lif = snn.RLeaky(beta=beta, linear_features=num_hidden, spike_grad=spike_grad)
        self.mean_decoder = nn.Linear(num_hidden, num_outputs)
        nn.init.kaiming_normal_(self.mean_decoder.weight, mode='fan_out')

        self.log_std_linear = nn.Linear(num_hidden, num_hidden)
        nn.init.kaiming_normal_(self.log_std_linear.weight, mode='fan_out')
        self.log_std_linear_lif = snn.RLeaky(beta=beta, linear_features=num_hidden, spike_grad=spike_grad)
        self.log_std_decoder = nn.Linear(num_hidden, num_outputs)
        nn.init.kaiming_normal_(self.log_std_decoder.weight, mode='fan_out')

    def custom_grad(self, input_, grad_input, spikes):
        ## The hyperparameter slope is defined inside the function.
        gamma = 0.3
        grad = gamma * torch.max(torch.zeros_like(input_), 1 - torch.abs(input_))
        return grad

    def forward(self, x, spks=None, mems=None):

        next_mem2_rec = {}
        next_spk2_rec = {}

        # time-loop
        cur1 = self.fc1(x)
        next_spk2_rec['lif1'], next_mem2_rec['lif1'] = self.lif1(cur1, spks['lif1'], mems['lif1'])
        cur2 = self.fc2(next_spk2_rec['lif1'])
        next_spk2_rec['lif2'], next_mem2_rec['lif2'] = self.lif2(cur2, spks['lif2'], mems['lif2'])

        cur_mean = self.mean_linear(next_spk2_rec['lif2'])
        next_spk2_rec['mean_linear_lif'], next_mem2_rec['mean_linear_lif'] = self.mean_linear_lif(cur_mean, spks['mean_linear_lif'], mems['mean_linear_lif'])

        cur_std = self.log_std_linear(next_spk2_rec['lif2'])
        next_spk2_rec['log_std_linear_lif'], next_mem2_rec['log_std_linear_lif'] = self.log_std_linear_lif(cur_std, spks['log_std_linear_lif'], mems['log_std_linear_lif'])

        spk_mean_decoded = self.mean_decoder(next_spk2_rec['mean_linear_lif'])
        spk_std_decoded = self.log_std_decoder(next_spk2_rec['log_std_linear_lif'])
        spk_std_decoded = torch.clamp(spk_std_decoded, min=LOG_SIG_MIN, max=LOG_SIG_MAX)

        return spk_mean_decoded, spk_std_decoded, next_mem2_rec, next_spk2_rec
    
    def sample(self, state, sampling, spks=None, mem=None, training=False):

        if sampling == True:
            state = state.unsqueeze(0)

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
class CriticRSNN(nn.Module):
    def __init__(self, num_inputs=63, num_outputs=18, num_hidden=512, num_steps=10, beta=.95):
        super(CriticRSNN, self).__init__()
        self.num_steps = num_steps
        spike_grad = surrogate.custom_surrogate(self.custom_grad)
        #spike_grad = surrogate.atan()

        # QNet 1
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out')
        self.lif1 = snn.RLeaky(beta=beta, learn_threshold=True, linear_features=num_hidden, spike_grad=spike_grad)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_out')
        self.lif2 = snn.RLeaky(beta=beta, learn_threshold=True, linear_features=num_hidden, spike_grad=spike_grad)
        self.output_decoder_1 = nn.Linear(num_hidden, 1)
        nn.init.kaiming_normal_(self.output_decoder_1.weight, mode='fan_out')

        # QNet 2
        self.fc1_2 = nn.Linear(num_inputs, num_hidden)
        nn.init.kaiming_normal_(self.fc1_2.weight, mode='fan_out')
        self.lif1_2 = snn.RLeaky(beta=beta, learn_threshold=True, linear_features=num_hidden, spike_grad=spike_grad)
        self.fc2_2 = nn.Linear(num_hidden, num_hidden)
        nn.init.kaiming_normal_(self.fc2_2.weight, mode='fan_out')
        self.lif2_2 = snn.RLeaky(beta=beta, learn_threshold=True, linear_features=num_hidden, spike_grad=spike_grad)
        self.output_decoder_2 = nn.Linear(num_hidden, 1)
        nn.init.kaiming_normal_(self.output_decoder_2.weight, mode='fan_out')

    def custom_grad(self, input_, grad_input, spikes):
        ## The hyperparameter slope is defined inside the function.
        gamma = 0.3
        grad = gamma * torch.max(torch.zeros_like(input_), 1 - torch.abs(input_))
        return grad

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
    
    
class PolicyRNN(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(PolicyRNN, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.lstm = nn.RNN(hidden_dim, hidden_dim, batch_first=True)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

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

    def forward_for_simple_dynamics(self, state, h_prev, sampling, len_seq= None):

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
        return super(PolicyRNN, self).to(device)
    

class CriticRNN(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(CriticRNN, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):

        xu = torch.cat([state, action], 1)
        
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2
    

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
        self.fc3 = nn.Linear(num_hidden, num_hidden)
        self.fc4 = nn.Linear(num_hidden, num_hidden)

        self.mean_linear = nn.Linear(num_hidden, num_outputs)
        self.log_std_linear = nn.Linear(num_hidden, num_outputs)

    def forward(self, x):

        # time-loop
        cur1 = self.fc1(x)
        cur2 = self.fc2(cur1)
        cur3 = self.fc3(cur2)
        cur4 = self.fc4(cur3)

        cur_mean = self.mean_linear(cur4)
        cur_std = self.log_std_linear(cur4)
        cur_std = torch.clamp(cur_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)

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
        log_prob = log_prob.sum(-1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias

        return action, log_prob, mean

# Define critic SNN Network
class CriticANN(nn.Module):
    def __init__(self, num_inputs=63, num_outputs=18, num_hidden=512, num_steps=25, beta=.9):
        super(CriticANN, self).__init__()
        self.num_steps = num_steps

        # QNet 1
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.fc3 = nn.Linear(num_hidden, num_hidden)
        self.fc4 = nn.Linear(num_hidden, 1)

        # QNet 2
        self.fc1_2 = nn.Linear(num_inputs, num_hidden)
        self.fc2_2 = nn.Linear(num_hidden, num_hidden)
        self.fc3_2 = nn.Linear(num_hidden, num_hidden)
        self.fc4_2 = nn.Linear(num_hidden, 1)

    def forward(self, state, action):

        x = torch.cat([state, action], dim=-1)

        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        q1 = self.fc4(out)

        out = self.fc1_2(x)
        out = self.fc2_2(out)
        out = self.fc3_2(out)
        q2 = self.fc4_2(out)

        return q1, q2