import time
import torch
import numpy as np
from gymnasium.wrappers import RecordVideo

from model import DQN
from environment import make_env
from config import Config


def play():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_name = Config.ENV_NAME

    difficulty = 0
    mode = 0
    print(
        f"Loading environment {env_name} with Difficulty {difficulty}, Mode {mode}..."
    )

    # Create environment with rgb_array for recording
    env = make_env(env_name, render_mode="rgb_array", difficulty=difficulty, mode=mode)

    # Wrap with RecordVideo to record all episodes
    env = RecordVideo(
        env,
        video_folder="results/breakout/videos",
        episode_trigger=lambda x: True,  # Record all episodes
        name_prefix="breakout_gameplay",
    )

    # Also create a human render environment for visualization
    human_env = make_env(
        env_name, render_mode="human", difficulty=difficulty, mode=mode
    )
    n_actions = env.action_space.n

    policy_net = DQN((4, 84, 84), n_actions).to(device)

    state_dict = torch.load("results/breakout/policy_net.pth", map_location=device)
    policy_net.load_state_dict(state_dict)
    policy_net.eval()  # Set to evaluation mode that super important to disable dropout/batchnorm

    print("Playing! Press Ctrl+C to stop.")

    games_played = 0
    current_game_score = 0
    state, _ = env.reset()
    state = np.array(state)

    # Also reset human env for visualization
    human_state, _ = human_env.reset()

    print("Recording and playing! Press Ctrl+C to stop.")
    print(f"Videos will be saved to: results/breakout/videos/")

    while games_played < 5:
        with torch.no_grad():
            state_tensor = torch.tensor(state, device=device).unsqueeze(0) / 255.0
            action = policy_net(state_tensor).max(1)[1].item()

        # Step both environments
        next_state, reward, done, _, info = env.step(action)
        human_env.step(action)

        next_state = np.array(next_state)
        state = next_state
        current_game_score += reward

        time.sleep(0.02)

        if done:
            lives = info.get("lives", 0)
            if lives == 0:
                print(f"Game {games_played + 1} Score: {current_game_score}")
                games_played += 1
                current_game_score = 0
                state, _ = env.reset()
                human_state, _ = human_env.reset()
            else:
                state, _ = env.reset()
                human_state, _ = human_env.reset()

    env.close()
    human_env.close()
    print("\nRecording complete! Check results/breakout/videos/ for the video files.")


if __name__ == "__main__":
    play()
