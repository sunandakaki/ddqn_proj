import random
from collections import deque

class UniformReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.memory = deque(maxlen=buffer_size)

    def push(self, transition):
        """Add a transition tuple (s, a, s', r, done)."""
        self.memory.append(transition)

    def sample(self):
        """Uniformly sample a batch of transitions."""
        batch = random.sample(self.memory, self.batch_size)
        return batch

    @property
    def num_entries(self):
        return len(self.memory)