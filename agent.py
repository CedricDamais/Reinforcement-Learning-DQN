import random
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import GradScaler
from collections import namedtuple

from model import DQN
from config import Config
from data.replay_memory import ReplayMemory
from data.memory import Memory

Transition = namedtuple(
    "Transition", ("state", "action", "next_state", "reward", "done")
)


class DQNAgent:
    def __init__(self, input_shape, n_actions, device=Config.DEVICE):
        self.device = device
        self.n_actions = n_actions

        # Initialize networks
        self.policy_net = DQN(input_shape, n_actions).to(device)
        self.target_net = DQN(input_shape, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer and Scaler
        self.optimizer = optim.RMSprop(
            self.policy_net.parameters(), lr=Config.LR, alpha=0.95, eps=0.01
        )
        self.scaler = GradScaler("cuda") if device.type == "cuda" else None

        # Memory
        self.memory = ReplayMemory(Config.MEMORY_SIZE)

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
        if len(self.memory) < Config.BATCH_SIZE:
            return

        transitions = self.memory.sample(Config.BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        # Optimized tensor creation - batch operations
        state_batch = (
            torch.stack(batch.state).to(device=self.device, dtype=torch.float32) / 255.0
        )
        action_batch = torch.tensor(
            batch.action, device=self.device, dtype=torch.long
        ).unsqueeze(1)
        reward_batch = torch.tensor(
            batch.reward, device=self.device, dtype=torch.float32
        )

        # Handle final states (where next_state is None)
        non_final_mask = torch.tensor(
            [s is not None for s in batch.next_state],
            device=self.device,
            dtype=torch.bool,
        )
        non_final_next_states_list = [s for s in batch.next_state if s is not None]

        if non_final_next_states_list:
            non_final_next_states = (
                torch.stack(non_final_next_states_list).to(
                    device=self.device, dtype=torch.float32
                )
                / 255.0
            )
        else:
            non_final_next_states = torch.empty(
                (0,) + state_batch.shape[1:], device=self.device, dtype=torch.float32
            )

        # Compute Q(s, a) using Policy Net
        current_q_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute Max Q(s', a') using Target Net
        next_state_values = torch.zeros(
            Config.BATCH_SIZE, device=self.device, dtype=torch.float32
        )
        with torch.no_grad():
            if len(non_final_next_states) > 0:
                next_state_values[non_final_mask] = self.target_net(
                    non_final_next_states
                ).max(1)[0]

        # y = r + gamma * max Q
        expected_q_values = (next_state_values * Config.GAMMA) + reward_batch

        loss = F.mse_loss(current_q_values, expected_q_values.unsqueeze(1))

        self.optimizer.zero_grad()
        if self.scaler:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.policy_net.eval()

    def push_memory(self, state, action, next_state, reward, done):
        state_int8 = torch.tensor(state, dtype=torch.uint8, device="cpu")
        next_state_int8 = (
            None
            if next_state is None
            else torch.tensor(next_state, dtype=torch.uint8, device="cpu")
        )
        self.memory.push(
            Memory(
                state=state_int8,
                action=int(action),
                next_state=next_state_int8,
                reward=float(reward),
                done=bool(done),
            )
        )
