import numpy as np
import torch


class EfficientReplayMemory:
    def __init__(self, capacity, device, frame_stack=4, image_size=84):
        self.device = device
        self.capacity = capacity
        self.frame_stack = frame_stack
        self.ptr = 0
        self.size = 0

        # Pre-allocated buffers (store single frames as uint8)
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
            torch.tensor(np.stack(states), device=self.device, dtype=torch.float32) / 255.0,
            torch.tensor(self.actions[indices], device=self.device),
            torch.tensor(np.stack(next_states), device=self.device, dtype=torch.float32) / 255.0,
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
import numpy as np
import torch


class EfficientReplayMemory:
    def __init__(self, capacity, device, frame_stack=4, image_size=84):
        self.device = device
        self.capacity = capacity
        self.frame_stack = frame_stack
        self.ptr = 0
        self.size = 0

        # 1. Pre-allocate huge arrays (uint8 is 1 byte, float32 is 4 bytes)
        # We store OBSERVATIONS (1 frame), not STATES (4 frames)
        self.observations = np.zeros((capacity, image_size, image_size), dtype=np.uint8)
        self.actions = np.zeros((capacity, 1), dtype=np.int64)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.bool_)

    def push(self, state_frame, action, reward, done):
        """
        Save a transition.
        Args:
            state_frame: The SINGLE NEW FRAME (84x84) from the environment.
                         Not the stack of 4!
        """
        # Save raw data
        self.observations[self.ptr] = state_frame
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        # 1. Pick random indices
        indices = np.random.randint(0, self.size, size=batch_size)

        # 2. Construct the States and Next States on the fly
        states = []
        next_states = []

        for idx in indices:
            # Get State (History of 4 frames ending at idx)
            states.append(self._get_stack(idx))

            # Get Next State (History of 4 frames ending at idx+1)
            # If current frame is DONE, next_state is essentially invalid/terminal
            # But the Bellman update handles that via the (1-done) mask.
            # We still stack it to keep tensor shapes consistent.
            next_states.append(self._get_stack((idx + 1) % self.size))

        # 3. Convert to efficient batch tensors
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
        """
        Reconstructs the stack of 4 frames ending at 'idx'.
        Handles episode boundaries (if done=True in history).
        """
        frames = []
        for i in range(self.frame_stack):
            # Look backwards: idx, idx-1, idx-2, idx-3
            # Use modulo to wrap around buffer safely
            current_idx = (idx - i) % self.capacity

            # Edge Case: We wrapped around to the end of the buffer
            # OR we hit a 'done' flag from a previous game.
            # If we hit a done flag at 'current_idx-1', it means 'current_idx' is start of new game.
            # We shouldn't include frames from the old game.
            if i > 0 and self.dones[(idx - i - 1) % self.capacity]:
                # Fill remaining history with zeros (black screen)
                # because we hit the start of the episode
                for _ in range(self.frame_stack - len(frames)):
                    frames.append(np.zeros_like(self.observations[0]))
                break

            frames.append(self.observations[current_idx])

        # Frames are [t, t-1, t-2, t-3]. We want [t-3, t-2, t-1, t]
        return np.stack(frames[::-1])

    def __len__(self):
        return self.size
