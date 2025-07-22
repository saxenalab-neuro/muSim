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
        batch = random.sample(self.buffer, batch_size)
        batch_list = list(chain(*batch))
        state, action, reward, next_state, done, h_current, neural_activity, na_idx = map(np.stack, zip(*batch_list))

        policy_state_batch = [[list(element)[0] for element in sample] for sample in batch]
        policy_state_batch = list(map(torch.FloatTensor, policy_state_batch))

        return state, action, reward, next_state, done, h_current, policy_state_batch, neural_activity, na_idx

    def __len__(self):
        return len(self.buffer)
