"""
Quick evaluation script for DQN agent
Runs 30 episodes with epsilon=0.05 and reports average reward
"""

import numpy as np
import torch
from tqdm import tqdm

from model import DQN
from environment import make_env
from config import Config


def evaluate_agent(model_path="policy_net.pth", num_episodes=30, epsilon=0.05):
    """Evaluate the trained agent"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained model
    env = make_env(Config.ENV_NAME)
    n_actions = env.action_space.n
    input_shape = env.observation_space.shape  # (4, 84, 84) for stacked frames

    policy_net = DQN(input_shape, n_actions).to(device)
    policy_net.load_state_dict(torch.load(model_path, map_location=device))
    policy_net.eval()

    print(f"Evaluating DQN agent for {num_episodes} episodes with ε={epsilon}")
    print("=" * 60)

    rewards = []
    episodes_completed = 0

    with torch.no_grad():
        state, _ = env.reset()
        state = (
            torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0) / 255.0
        )
        current_game_reward = 0

        pbar = tqdm(total=num_episodes, desc="Episodes")
        while episodes_completed < num_episodes:
            if np.random.random() > epsilon:
                q_values = policy_net(state)
                action = q_values.max(1)[1].item()
            else:
                action = np.random.randint(n_actions)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            current_game_reward += reward

            if not done:
                state = (
                    torch.tensor(
                        next_state, dtype=torch.float32, device=device
                    ).unsqueeze(0)
                    / 255.0
                )
            else:
                # Check if it's a real game over (0 lives left)
                lives = info.get("lives", 0)
                if lives == 0:
                    rewards.append(current_game_reward)
                    episodes_completed += 1
                    pbar.update(1)
                    current_game_reward = 0

                    # Reset for new game
                    state, _ = env.reset()
                    state = (
                        torch.tensor(
                            state, dtype=torch.float32, device=device
                        ).unsqueeze(0)
                        / 255.0
                    )
                else:
                    # Life lost, continue game (EpisodicLifeEnv requires reset to advance)
                    state, _ = env.reset()
                    state = (
                        torch.tensor(
                            state, dtype=torch.float32, device=device
                        ).unsqueeze(0)
                        / 255.0
                    )

        pbar.close()

    env.close()

    avg_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    min_reward = np.min(rewards)
    max_reward = np.max(rewards)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Episodes: {num_episodes}")
    print(f"Epsilon: {epsilon}")
    print(f"Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Max Reward: {max_reward:.1f}")
    print(f"Min Reward: {min_reward:.1f}")
    print("=" * 60)

    # Store result for report
    result_text = f"""DQN Evaluation Results (ε=0.05, 30 episodes):
- Average Reward: {avg_reward:.2f} ± {std_reward:.2f}
- Max Reward: {max_reward:.1f}
- Min Reward: {min_reward:.1f}
- Individual Rewards: {rewards}
"""

    with open("evaluation_results.txt", "w") as f:
        f.write(result_text)

    print("Results saved to evaluation_results.txt for your report!")

    return avg_reward, std_reward, rewards


if __name__ == "__main__":
    evaluate_agent()
