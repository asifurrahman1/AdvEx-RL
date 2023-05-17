import random
import numpy as np
from operator import itemgetter

class ConstraintReplayMemory:
    ''' 
        Replay buffer for training recovery policy and associated safety critic
    '''
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.pos_idx = np.zeros(self.capacity)

    def push(self, state, action, reward, constraint_violation, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, constraint_violation, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, constraint_violation, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, constraint_violation, next_state, done

    def __len__(self):
        return len(self.buffer)