import numpy as np
import torch
import torch.optim as optim
from torch.amp import GradScaler
from tqdm import tqdm
from model import DQN
from config import Config
from algo import select_action, optimize_model
from environment import make_env
from data.replay_memory import ReplayMemory
from data.memory import Memory


def train():
    env = make_env(Config.ENV_NAME)
    print("Populating initial memory for validation set...")
    n_actions = env.action_space.n

    policy_net = DQN((4, 84, 84), n_actions).to(Config.DEVICE)
    target_net = DQN((4, 84, 84), n_actions).to(Config.DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.RMSprop(
        policy_net.parameters(), lr=Config.LR, alpha=0.95, eps=0.01
    )
    # Mixed precision training for faster computation
    scaler = GradScaler("cuda") if Config.DEVICE.type == "cuda" else None
    memory = ReplayMemory(Config.MEMORY_SIZE)

    state, _ = env.reset()
    state = np.array(state)
    for _ in range(100):  # Play 100 random steps
        action = env.action_space.sample()

        next_state, reward, terminated, truncated, _ = env.step(action)
        next_state = np.array(next_state)
        done = terminated or truncated

        store_state = None if done else next_state
        state_int8 = torch.tensor(state, dtype=torch.uint8, device="cpu")
        next_state_int8 = (
            None
            if store_state is None
            else torch.tensor(store_state, dtype=torch.uint8, device="cpu")
        )

        memory.push(
            Memory(
                state=state_int8,
                action=int(action),
                next_state=next_state_int8,
                reward=float(reward),
                done=bool(done),
            )
        )
        state = next_state if not done else np.array(env.reset()[0])

    validation_samples = memory.sample(32)
    validation_states = (
        torch.stack([x.state for x in validation_samples]).float().to(Config.DEVICE)
        / 255.0
    )

    episode_rewards = []
    average_q_values = []

    steps_done = 0
    episode_rewards = []

    print(f"Starting training on {Config.ENV_NAME} with {Config.DEVICE}...")

    pbar = tqdm(total=Config.TOTAL_FRAMES)
    while steps_done < Config.TOTAL_FRAMES:
        state, _ = env.reset()
        state = np.array(state)
        episode_reward = 0

        while True:
            epsilon = max(
                Config.EPS_END,
                Config.EPS_START
                - (Config.EPS_START - Config.EPS_END)
                * min(1.0, steps_done / Config.EPS_DECAY_FRAMES),
            )

            action = select_action(state, policy_net, epsilon, n_actions, Config.DEVICE)

            next_state, reward, done, _, _ = env.step(action)
            next_state = np.array(next_state)
            episode_reward += reward

            # More efficient tensor creation - avoid repeated conversions
            state_tensor = torch.from_numpy(state).to(dtype=torch.uint8)
            next_state_tensor = (
                None if done else torch.from_numpy(next_state).to(dtype=torch.uint8)
            )

            memory.push(
                Memory(
                    state=state_tensor,
                    action=int(action),
                    next_state=next_state_tensor,
                    reward=float(reward),
                    done=bool(done),
                )
            )

            state = next_state

            # Only train every LEARNING_FREQ steps
            if steps_done % Config.LEARNING_FREQ == 0:
                optimize_model(
                    policy_net, target_net, memory, optimizer, Config.DEVICE, scaler
                )

            if steps_done % Config.VALIDATION_FREQ == 0:
                with torch.no_grad():
                    q_values = policy_net(validation_states)
                    avg_q = q_values.max(1)[0].mean().item()
                    average_q_values.append(avg_q)

            if steps_done % Config.TARGET_UPDATE_FREQ == 0:
                target_net.load_state_dict(policy_net.state_dict())
                torch.save(policy_net.state_dict(), "policy_net.pth")

            steps_done += 1
            pbar.update(1)
            if done:
                episode_rewards.append(episode_reward)
                pbar.set_description(
                    f"Step {steps_done}, Reward: {episode_reward}, Epsilon: {epsilon:.2f}"
                )
                break
    pbar.close()
    torch.save(policy_net.state_dict(), "policy_net.pth")
    print("Model saved to policy_net.pth")

    env.close()


if __name__ == "__main__":
    train()
