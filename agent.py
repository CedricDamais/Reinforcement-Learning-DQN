import random
import torch
import sys
import os

sys.path.insert(0, os.getcwd())
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import GradScaler
from collections import namedtuple

from model import DQN
from config import Config
from data.efficient_memory import EfficientReplayMemory

Transition = namedtuple(
    "Transition", ("state", "action", "next_state", "reward", "done")
)


class DQNAgent:
    def __init__(self, input_shape, n_actions, device=Config.DEVICE):
        self.device = device
        self.n_actions = n_actions

        self.policy_net = DQN(input_shape, n_actions).to(device)
        self.target_net = DQN(input_shape, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.RMSprop(
            self.policy_net.parameters(), lr=Config.LR, alpha=0.95, eps=0.01
        )
        self.scaler = GradScaler() if device.type == "cuda" else None

        self.memory = EfficientReplayMemory(Config.MEMORY_SIZE, device=Config.DEVICE)

    def select_action(self, state, epsilon):
        """Selects action using epsilon-greedy policy"""
        if random.random() < epsilon:
            return random.randrange(self.n_actions)
        else:
            with torch.no_grad():
                if isinstance(state, torch.Tensor):
                    state_tensor = (
                        state.to(
                            self.device, dtype=torch.float32, non_blocking=True
                        ).unsqueeze(0)
                        / 255.0
                    )
                else:
                    state_tensor = (
                        torch.from_numpy(state)
                        .to(self.device, dtype=torch.float32, non_blocking=True)
                        .unsqueeze(0)
                        / 255.0
                    )
                return self.policy_net(state_tensor).argmax().item()

    def optimize_model(self):
        """Performs one step of gradient descent on the batch"""
        # Use efficient memory sample which returns tensors on device
        if len(self.memory) < Config.BATCH_SIZE:
            return None

        state_batch, action_batch, next_state_batch, reward_batch, done_batch = (
            self.memory.sample(Config.BATCH_SIZE)
        )

        # Ensure shapes/dtypes
        if action_batch.dim() == 2 and action_batch.shape[1] == 1:
            action_batch = action_batch.squeeze(1)
        action_batch = action_batch.to(device=self.device, dtype=torch.long).unsqueeze(
            1
        )

        if reward_batch.dim() == 2 and reward_batch.shape[1] == 1:
            reward_batch = reward_batch.squeeze(1)
        reward_batch = reward_batch.to(device=self.device, dtype=torch.float32)

        done_mask = done_batch.to(device=self.device, dtype=torch.bool).squeeze(1)

        # Compute Q(s, a) using Policy Net
        current_q_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute Max Q(s', a') using Target Net
        next_state_values = torch.zeros(
            Config.BATCH_SIZE, device=self.device, dtype=torch.float32
        )
        with torch.no_grad():
            next_state_values[~done_mask] = self.target_net(
                next_state_batch[~done_mask]
            ).max(1)[0]

        # y = r + gamma * max Q
        expected_q_values = (next_state_values * Config.GAMMA) + reward_batch

        loss = F.mse_loss(current_q_values, expected_q_values.unsqueeze(1))

        self.optimizer.zero_grad()
        if self.scaler:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            for param in self.policy_net.parameters():
                if param.grad is not None:
                    param.grad.data.clamp_(-1, 1)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            for param in self.policy_net.parameters():
                if param.grad is not None:
                    param.grad.data.clamp_(-1, 1)
            self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.policy_net.eval()

    def push_memory(self, state, action, reward, done):
        """
        Accepts a full state (stack of 4 frames) or a single frame.
        EfficientReplayMemory stores single frames only and will reconstruct stacks on sampling.
        """
        if hasattr(state, "ndim") and state.ndim == 3 and state.shape[0] == 4:
            last_frame = state[-1]
        else:
            last_frame = state

        if isinstance(last_frame, torch.Tensor):
            last_frame = last_frame.cpu().numpy()
        if last_frame.dtype != np.uint8:
            last_frame = (last_frame * 255).astype(np.uint8)

        self.memory.push(last_frame, int(action), float(reward), bool(done))
