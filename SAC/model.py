import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class Actor(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, model, action_space=None):
        super(Actor, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)

        if model == "rnn":
            self.rnn = nn.RNN(hidden_dim, hidden_dim, batch_first=True)
        elif model == "gru":
            self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        else:
            raise NotImplementedError

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

    def forward(self, state, h_prev, sampling, len_seq= None):

        x = F.tanh(self.linear1(state))

        if sampling == False:
            assert len_seq!=None, "Proved the len_seq"
            x = pack_padded_sequence(x, len_seq, batch_first= True, enforce_sorted= False)

        #Tap RNN input for fixedpoint analysis
        rnn_in = x

        x, (h_current) = self.rnn(x, (h_prev))

        if sampling == False:
           x, _ = pad_packed_sequence(x, batch_first= True)

        if sampling == True:
            x = x.squeeze(1)

        # x = F.relu(x)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)

        return mean, log_std, h_current, x, rnn_in

    def sample(self, state, h_prev, sampling, len_seq=None):

        mean, log_std, h_current, x, rnn_in = self.forward(state, h_prev, sampling, len_seq)
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
            mean = mean.reshape(-1, mean.size()[-1])[mask_seq]
            log_std = log_std.reshape(-1, log_std.size()[-1])[mask_seq]

        if sampling == True:
            mask_seq = [] #If sampling is True return a dummy mask seq

        std = log_std.exp()

        # white noise
        normal = Normal(mean, std)
        noise = normal.rsample()

        # reparameterization trick
        y_t = torch.tanh(noise) 
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(noise)

        # Enforce the action_bounds
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias

        return action, log_prob, mean, h_current, mask_seq, x, rnn_in

    def forward_for_simple_dynamics(self, state, h_prev, sampling, len_seq= None):

        x = F.tanh(self.linear1(state))

        #Tap the output of the first linear layer
        x_l1 = x

        if sampling == False:
            assert len_seq!=None, "Proved the len_seq"
            x = pack_padded_sequence(x, len_seq, batch_first= True, enforce_sorted= False)

        x, _ = self.rnn(x, (h_prev))

        if sampling == False:
           x, _ = pad_packed_sequence(x, batch_first= True)

        # x = F.relu(x)

        return x, x_l1


    def forward_lstm(self, state, h_prev, sampling, len_seq= None):

        x = F.tanh(self.linear1(state))

        if sampling == False:
            assert len_seq!=None, "Proved the len_seq"

            x = pack_padded_sequence(x, len_seq, batch_first= True, enforce_sorted= False)

        x, (h_current) = self.rnn(x, (h_prev))

        if sampling == False:
           x, len_x_seq = pad_packed_sequence(x, batch_first= True)

        if sampling == True:
            x = x.squeeze(1)

        return x

    def forward_for_neural_pert(self, state, h_prev, neural_pert= None):

        x = F.tanh(self.linear1(state))

        #Tap RNN input for fixedpoint analysis
        rnn_in = x

        x, (h_current) = self.rnn(x, (h_prev))

        #Add the neural perturbation to the RNN output
        x = x+neural_pert

        x = x.squeeze(1)

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)


        action = torch.tanh(mean) * self.action_scale + self.action_bias

        return action.detach().cpu().numpy()[0], h_current.detach(), x.detach().cpu().numpy(), rnn_in.detach().cpu().numpy()

class Critic(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):

        xu = torch.cat([state, action], 1)
        
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        return x1


class DoubleCritic(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(DoubleCritic, self).__init__()

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