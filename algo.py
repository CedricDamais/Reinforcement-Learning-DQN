import random
from collections import namedtuple

import torch
import torch.nn.functional as F

from config import Config

Transition = namedtuple(
    "Transition", ("state", "action", "next_state", "reward", "done")
)


def select_action(state, policy_net, epsilon, n_actions, device):
    """
    Selects action using epsilon-greedy policy[cite: 120].
    Optimized for faster inference.
    """
    if random.random() < epsilon:
        return random.randrange(n_actions)  # Explore
    else:
        with torch.no_grad():
            # More efficient tensor handling with single normalization
            if isinstance(state, torch.Tensor):
                state_tensor = state.to(device, dtype=torch.float32, non_blocking=True).unsqueeze(0) / 255.0
            else:
                state_tensor = (
                    torch.from_numpy(state)
                    .to(device, dtype=torch.float32, non_blocking=True)
                    .unsqueeze(0) / 255.0
                )

            return policy_net(state_tensor).argmax().item()  # Faster than max(1)[1]


def optimize_model(policy_net, target_net, memory, optimizer, device, scaler=None):
    """
    Performs one step of gradient descent on the batch [cite: 135-141].
    """
    if len(memory) < Config.BATCH_SIZE:
        return

    transitions = memory.sample(Config.BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # Optimized tensor creation - batch operations
    state_batch = (
        torch.stack(batch.state).to(device=device, dtype=torch.float32) / 255.0
    )
    action_batch = torch.tensor(
        batch.action, device=device, dtype=torch.long
    ).unsqueeze(1)
    reward_batch = torch.tensor(batch.reward, device=device, dtype=torch.float32)

    # Handle final states (where next_state is None) - more efficiently
    non_final_mask = torch.tensor(
        [s is not None for s in batch.next_state], device=device, dtype=torch.bool
    )
    non_final_next_states_list = [s for s in batch.next_state if s is not None]

    if non_final_next_states_list:
        non_final_next_states = (
            torch.stack(non_final_next_states_list).to(
                device=device, dtype=torch.float32
            )
            / 255.0
        )
    else:
        # create an empty float tensor with the correct channel/shape
        non_final_next_states = torch.empty(
            (0,) + state_batch.shape[1:], device=device, dtype=torch.float32
        )

    # Mixed precision forward pass
    if scaler is not None:
        from torch.amp import autocast

        with autocast("cuda"):
            # 1. Compute Q(s, a) using Policy Net
            current_q_values = policy_net(state_batch).gather(1, action_batch)

            # 2. Compute Max Q(s', a') using Target Net (Bellman Equation)
            next_state_values = torch.zeros(
                Config.BATCH_SIZE, device=device, dtype=torch.float16
            )
            with torch.no_grad():
                if len(non_final_next_states) > 0:
                    next_state_values[non_final_mask] = target_net(
                        non_final_next_states
                    ).max(1)[0]

            # Compute Target: y = r + gamma * max Q
            expected_q_values = (next_state_values * Config.GAMMA) + reward_batch

            # 3. Compute Loss (MSE) [cite: 141]
            loss = F.mse_loss(current_q_values, expected_q_values.unsqueeze(1))

        # 4. Update Weights with mixed precision
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        # Clamp gradients for stability (Standard DQN practice)
        for param in policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        scaler.step(optimizer)
        scaler.update()
    else:
        # Standard precision training
        # 1. Compute Q(s, a) using Policy Net
        current_q_values = policy_net(state_batch).gather(1, action_batch)

        # 2. Compute Max Q(s', a') using Target Net (Bellman Equation)
        next_state_values = torch.zeros(
            Config.BATCH_SIZE, device=device, dtype=torch.float32
        )
        with torch.no_grad():
            if len(non_final_next_states) > 0:
                next_state_values[non_final_mask] = target_net(
                    non_final_next_states
                ).max(1)[0]

        # Compute Target: y = r + gamma * max Q
        expected_q_values = (next_state_values * Config.GAMMA) + reward_batch

        # 3. Compute Loss (MSE) [cite: 141]
        loss = F.mse_loss(current_q_values, expected_q_values.unsqueeze(1))

        # 4. Update Weights
        optimizer.zero_grad()
        loss.backward()
        # Clamp gradients for stability (Standard DQN practice)
        for param in policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()
