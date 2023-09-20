import random
import numpy as np
from itertools import chain
import torch
import torch.nn.functional as F

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

class PolicyReplayMemorySNN(PolicyReplayMemory):
    def __init__(self, capacity, seed):
        super(PolicyReplayMemorySNN, self).__init__(capacity, seed)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        state = [[tup[0] for tup in trajectory] for trajectory in batch]
        for i, t in enumerate(state):
            state[i] = torch.FloatTensor(state[i])
            # manually padding rn
            state[i] = F.pad(state[i], (0,0, 300-state[i].shape[0], 0), "constant", 0)
        action = [[tup[1] for tup in trajectory] for trajectory in batch]
        for i, t in enumerate(action):
            action[i] = torch.Tensor(action[i])
            action[i] = F.pad(action[i], (0,0, 300-action[i].shape[0], 0), "constant", 0)
        reward = [[tup[2] for tup in trajectory] for trajectory in batch]
        for i, t in enumerate(reward):
            reward[i] = torch.Tensor(reward[i])
            reward[i] = F.pad(reward[i], (300-reward[i].shape[0], 0), "constant", 0)
        next_state = [[tup[3] for tup in trajectory] for trajectory in batch]
        for i, t in enumerate(next_state):
            next_state[i] = torch.Tensor(next_state[i])
            next_state[i] = F.pad(next_state[i], (0,0, 300-next_state[i].shape[0], 0), "constant", 0)
        done = [[tup[4] for tup in trajectory] for trajectory in batch]
        for i, t in enumerate(done):
            done[i] = torch.Tensor(done[i])
            done[i] = F.pad(done[i], (300-done[i].shape[0], 0), "constant", 0)

        return state, action, reward, next_state, done

class PolicyReplayMemoryANN(PolicyReplayMemory):
    def __init__(self, capacity, seed):
        super(PolicyReplayMemorySNN, self).__init__(capacity, seed)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        state = [tup[0] for tup in batch]
        action = [tup[1] for tup in batch]
        reward = [tup[2] for tup in batch]
        next_state = [tup[3] for tup in batch]
        done = [tup[4] for tup in batch]

        return state, action, reward, next_state, done