import random
import numpy as np
from itertools import chain
import torch

class PolicyReplayMemory:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = state
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        pass

    def __len__(self):
        return len(self.buffer)

class PolicyReplayMemoryRNN(PolicyReplayMemory):
    def __init__(self, capacity, seed):
        super(PolicyReplayMemoryRNN, self).__init__(capacity, seed)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        batch_list = list(chain(*batch))
        state, action, reward, next_state, done, h_current, c_current = map(np.stack, zip(*batch_list))

        policy_state_batch = [[list(element)[0] for element in sample]for sample in batch]
        policy_state_batch = list(map(torch.FloatTensor, policy_state_batch))

        return state, action, reward, next_state, done, h_current, c_current, policy_state_batch

class PolicyReplayMemoryLSTM(PolicyReplayMemory):
    def __init__(self, capacity, seed):
        super(PolicyReplayMemoryLSTM, self).__init__(capacity, seed)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        state = [[list(element)[0] for element in sample]for sample in batch]
        state = list(map(torch.FloatTensor, state))

        action = [[list(element)[1] for element in sample]for sample in batch]
        action = list(map(torch.FloatTensor, action))

        reward = [[list(element)[2] for element in sample]for sample in batch]
        reward = list(map(torch.FloatTensor, reward))

        next_state = [[list(element)[3] for element in sample]for sample in batch]
        next_state = list(map(torch.FloatTensor, next_state))

        done = [[list(element)[4] for element in sample]for sample in batch]
        done = list(map(torch.FloatTensor, done))

        hi_lst = []
        ci_lst = []
        ho_lst = []
        co_lst = []

        for sample in batch:
            hi_lst.append(list(sample[0])[5])
            ci_lst.append(list(sample[0])[6])
            ho_lst.append(list(sample[0])[7])
            co_lst.append(list(sample[0])[8])

        hi_lst = torch.cat(hi_lst, dim= -2).detach()
        ci_lst = torch.cat(ci_lst, dim= -2).detach()
        ho_lst = torch.cat(ho_lst, dim= -2).detach()
        co_lst = torch.cat(co_lst, dim= -2).detach()

        hidden_in = (hi_lst, ci_lst)
        hidden_out = (ho_lst, co_lst)       

        return state, action, reward, next_state, done, hidden_in, hidden_out