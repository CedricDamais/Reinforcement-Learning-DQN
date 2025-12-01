import numpy as np
import torch


class EfficientReplayMemory:
    def __init__(self, capacity, device, frame_stack=4, image_size=84):
        self.device = device
        self.capacity = capacity
        self.frame_stack = frame_stack
        self.ptr = 0
        self.size = 0

        self.observations = np.zeros((capacity, image_size, image_size), dtype=np.uint8)
        self.actions = np.zeros((capacity, 1), dtype=np.int64)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.bool_)

    def push(self, state_frame, action, reward, done):
        self.observations[self.ptr] = state_frame
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        indices = np.random.randint(0, self.size, size=batch_size)

        states = []
        next_states = []
        for idx in indices:
            states.append(self._get_stack(idx))
            next_states.append(self._get_stack((idx + 1) % self.size))

        return (
            torch.tensor(np.stack(states), device=self.device, dtype=torch.float32)
            / 255.0,
            torch.tensor(self.actions[indices], device=self.device),
            torch.tensor(np.stack(next_states), device=self.device, dtype=torch.float32)
            / 255.0,
            torch.tensor(self.rewards[indices], device=self.device),
            torch.tensor(self.dones[indices], device=self.device),
        )

    def _get_stack(self, idx):
        frames = []
        for i in range(self.frame_stack):
            current_idx = (idx - i) % self.capacity
            if i > 0 and self.dones[(idx - i - 1) % self.capacity]:
                for _ in range(self.frame_stack - len(frames)):
                    frames.append(np.zeros_like(self.observations[0]))
                break
            frames.append(self.observations[current_idx])
        return np.stack(frames[::-1])

    def __len__(self):
        return self.size
