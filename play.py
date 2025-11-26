import time
import torch
import numpy as np

from model import DQN
from environment import make_env
from config import Config


def play():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_name = Config.ENV_NAME

    # Difficulty 0: Normal (Large Paddles)
    # Difficulty 1: Hard (Small Paddles)
    difficulty = 0
    mode = 0
    print(
        f"Loading environment {env_name} with Difficulty {difficulty}, Mode {mode}..."
    )

    env = make_env(env_name, render_mode="human", difficulty=difficulty, mode=mode)
    n_actions = env.action_space.n

    policy_net = DQN((4, 84, 84), n_actions).to(device)

    state_dict = torch.load("policy_net.pth", map_location=device)
    policy_net.load_state_dict(state_dict)
    policy_net.eval()  # Set to evaluation mode that super important to disable dropout/batchnorm

    print("Playing! Press Ctrl+C to stop.")

    games_played = 0
    current_game_score = 0
    state, _ = env.reset()
    state = np.array(state)

    while games_played < 5:
        with torch.no_grad():
            state_tensor = torch.tensor(state, device=device).unsqueeze(0) / 255.0
            action = policy_net(state_tensor).max(1)[1].item()

        next_state, reward, done, _, info = env.step(action)
        next_state = np.array(next_state)
        state = next_state
        current_game_score += reward

        time.sleep(0.02)

        if done:
            # Check if it's a real game over (0 lives left) or just a life lost
            lives = info.get("lives", 0)
            if lives == 0:
                print(f"Game {games_played + 1} Score: {current_game_score}")
                games_played += 1
                current_game_score = 0
                state, _ = env.reset()
            else:
                # Life lost, continue game (EpisodicLifeEnv requires reset to advance)
                state, _ = env.reset()

    env.close()


if __name__ == "__main__":
    play()
