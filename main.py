from collections import deque
import time

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from agent import DQNAgent
from config import Config
from environment import make_env


def train():
    env = make_env(Config.ENV_NAME)
    print("Populating initial memory for validation set...")
    n_actions = env.action_space.n

    # Initialize Agent
    agent = DQNAgent((4, 84, 84), n_actions, Config.DEVICE)

    # TensorBoard writer
    writer = SummaryWriter(log_dir=f"runs/{Config.ENV_NAME}_{int(time.time())}")

    # Populate memory with random actions
    state, _ = env.reset()
    state = np.array(state)
    for _ in range(100):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)
        next_state = np.array(next_state)
        done = terminated or truncated

        # EfficientReplayMemory expects single frames; agent.push_memory handles extracting last frame
        agent.push_memory(state, action, reward, done)

        state = next_state if not done else np.array(env.reset()[0])

    # Create validation set (EfficientReplayMemory returns tensors already normalized on the device)
    validation_states, _, _, _, _ = agent.memory.sample(32)

    episode_rewards = []
    average_q_values = []
    recent_rewards = deque(maxlen=100)
    steps_done = 0

    def get_vram_usage():
        if Config.DEVICE.type == "cuda":
            return f"VRAM: {torch.cuda.memory_allocated() / 1024**3:.2f}GB/{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB"
        return "CPU mode"

    print(f"Starting training on {Config.ENV_NAME} with {Config.DEVICE}...")
    print(f"Initial {get_vram_usage()}")

    pbar = tqdm(total=Config.TOTAL_FRAMES)
    game_reward = 0  # Accumulator for the full game score (across lives)
    game_reward_orig = 0  # Accumulate original (pre-clipping) reward for comparison

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

            action = agent.select_action(state, epsilon)

            next_state, reward, done, _, info = env.step(action)
            next_state = np.array(next_state)
            episode_reward += reward
            game_reward += reward
            # Log original vs clipped reward per step (throttle to every 10 steps to avoid spamming)
            if info is not None and "original_reward" in info:
                orig_r = float(info["original_reward"]) if info["original_reward"] is not None else 0.0
                game_reward_orig += orig_r
                if steps_done % 10 == 0:
                    writer.add_scalar("Train/StepReward/Original", orig_r, steps_done)
                    writer.add_scalar("Train/StepReward/Clipped", float(reward), steps_done)
            else:
                # Fall back to logging clipped-only if original not present
                if steps_done % 10 == 0:
                    writer.add_scalar("Train/StepReward/Clipped", float(reward), steps_done)

            agent.push_memory(state, action, reward, done)

            state = next_state

            adaptive_learning_freq = max(
                Config.LEARNING_FREQ, int(Config.LEARNING_FREQ * (2 - epsilon))
            )

            if steps_done % adaptive_learning_freq == 0:
                loss = agent.optimize_model()
                if loss is not None:
                    writer.add_scalar("Train/Loss", loss, steps_done)

            if steps_done % Config.VALIDATION_FREQ == 0:
                with torch.no_grad():
                    q_values = agent.policy_net(validation_states)
                    avg_q = q_values.max(1)[0].mean().item()
                    average_q_values.append(avg_q)
                    writer.add_scalar("Train/AvgQ", avg_q, steps_done)

                    avg_reward_100 = (
                        np.mean(recent_rewards) if len(recent_rewards) >= 10 else 0
                    )
                    print(
                        f"\n[Step {steps_done}] Avg Q-value: {avg_q:.3f}, Avg Game Reward (100 games): {avg_reward_100:.2f}, {get_vram_usage()}"
                    )

            if steps_done % Config.TARGET_UPDATE_FREQ == 0:
                agent.update_target_network()
                agent.save("policy_net.pth")

            steps_done += 1
            pbar.update(1)

            if done:
                # Check if it's a real game over (0 lives left)
                lives = info.get("lives", 0)
                if lives == 0:
                    episode_rewards.append(game_reward)
                    recent_rewards.append(game_reward)

                    avg_reward = (
                        np.mean(recent_rewards) if recent_rewards else game_reward
                    )

                    if len(recent_rewards) >= 10:
                        pbar.set_description(
                            f"Step {steps_done}, Game Reward: {game_reward:.1f}, Avg100: {avg_reward:.1f}, ε: {epsilon:.3f} | {get_vram_usage()}"
                        )
                    else:
                        pbar.set_description(
                            f"Step {steps_done}, Game Reward: {game_reward:.1f}, ε: {epsilon:.3f} | {get_vram_usage()}"
                        )

                    # Log the per-game reward as an episode metric
                    writer.add_scalar("Train/EpisodeReward", game_reward, steps_done)
                    writer.add_scalar("Train/EpisodeReward/Original", game_reward_orig, steps_done)
                    game_reward = 0  # Reset for the next game
                    game_reward_orig = 0

                break

    pbar.close()
    writer.close()
    agent.save("policy_net.pth")
    print("Model saved to policy_net.pth")
    env.close()


if __name__ == "__main__":
    train()
