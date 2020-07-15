import random
import numpy as np
from collections import deque


class trajectory_buffer(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=self.capacity)
        # * [obs, next_obs, act, rew, don, val]

    def store(self, obs, next_obs, act, rew, don, val):
        obs = np.expand_dims(obs, 0)
        next_obs = np.expand_dims(next_obs, 0)
        self.memory.append([obs, next_obs, act, rew, don, val])

    def get(self):
        obs, next_obs, act, rew, don, val = zip(* self.memory)
        act = np.expand_dims(act, 1)
        rew = np.expand_dims(rew, 1)
        don = np.expand_dims(don, 1)
        val = np.expand_dims(val, 1)
        return np.concatenate(obs, 0), np.concatenate(next_obs, 0), act, rew, don, val

    def __len__(self):
        return len(self.memory)

    def clear(self):
        self.memory.clear()