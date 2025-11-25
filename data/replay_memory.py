from collections import deque
import random

DEFAULT_CAPACITY = 2000


class ReplayMemory:
    def __init__(self, capacity: int = DEFAULT_CAPACITY):
        self.capacity = capacity
        self.memory = deque(maxlen=self.capacity)

    def push(self, memory) -> None:
        self.memory.append(memory)

    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)
