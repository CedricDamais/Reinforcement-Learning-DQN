import random
from collections import namedtuple

import numpy as np
import torch
import torch.nn.functional as F

from config import Config

Transition = namedtuple(
    "Transition", ("state", "action", "next_state", "reward", "done")
)


def select_action(state, policy_net, epsilon, n_actions, device):
    """
    Selects action using epsilon-greedy policy[cite: 120].
    """
    if random.random() < epsilon:
        return random.randrange(n_actions)  # Explore
    else:
        with torch.no_grad():
            # convert state to tensor on device, normalize if integer image type
            if isinstance(state, torch.Tensor):
                s = state
            else:
                s = torch.as_tensor(state)

            if not s.is_floating_point():
                s = s.float() / 255.0

            state_tensor = s.to(device).unsqueeze(0)
            q_values = policy_net(state_tensor)
            return q_values.max(1)[1].item()  # Exploit


def optimize_model(policy_net, target_net, memory, optimizer, device):
    """
    Performs one step of gradient descent on the batch [cite: 135-141].
    """
    if len(memory) < Config.BATCH_SIZE:
        return

    transitions = memory.sample(Config.BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # Convert batch arrays to tensors (stack if already tensors)
    # Note: We create (Batch, 4, 84, 84) and convert to float32 normalized in [0,1]
    if isinstance(batch.state[0], torch.Tensor):
        state_batch = torch.stack(batch.state)
        state_batch = state_batch.to(device=device).float() / 255.0
    else:
        state_batch = (
            torch.tensor(np.array(batch.state), dtype=torch.float32, device=device)
            / 255.0
        )
    action_batch = torch.tensor(batch.action, device=device).unsqueeze(1)
    reward_batch = torch.tensor(batch.reward, device=device)

    # Handle final states (where next_state is None)
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device,
        dtype=torch.bool,
    )
    non_final_next_states_list = [s for s in batch.next_state if s is not None]
    if len(non_final_next_states_list) > 0:
        if isinstance(non_final_next_states_list[0], torch.Tensor):
            non_final_next_states = (
                torch.stack(non_final_next_states_list).to(device=device).float()
                / 255.0
            )
        else:
            non_final_next_states = (
                torch.tensor(
                    np.array(non_final_next_states_list),
                    dtype=torch.float32,
                    device=device,
                )
                / 255.0
            )
    else:
        # create an empty float tensor with the correct channel/shape
        non_final_next_states = torch.empty(
            (0,) + state_batch.shape[1:], device=device, dtype=torch.float32
        )

    # 1. Compute Q(s, a) using Policy Net
    current_q_values = policy_net(state_batch).gather(1, action_batch)

    # 2. Compute Max Q(s', a') using Target Net (Bellman Equation)
    next_state_values = torch.zeros(Config.BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]

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
