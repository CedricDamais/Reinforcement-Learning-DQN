import time
import torch
import numpy as np
from model import DQN
from config import Config
from environment import make_env
from algo import select_action

def visualize():
    env = make_env(Config.ENV_NAME, render_mode="human")
    n_actions = env.action_space.n

    policy_net = DQN((4, 84, 84), n_actions).to(Config.DEVICE)
    
    try:
        policy_net.load_state_dict(torch.load("policy_net.pth", map_location=Config.DEVICE))
        print("Loaded model from policy_net.pth")
    except FileNotFoundError:
        print("No trained model found. Please run main.py first.")
        return

    policy_net.eval()

    while True:
        state, _ = env.reset()
        state = np.array(state)
        done = False
        total_reward = 0

        while not done:
            # Select action with epsilon=0.05 (small exploration) or 0 (greedy)
            action = select_action(state, policy_net, epsilon=0.05, n_actions=n_actions, device=Config.DEVICE)
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = np.array(next_state)
            
            # Add a small delay to make it watchable if it's too fast (though 'human' mode usually handles this)
            # time.sleep(0.01) 

        print(f"Episode finished with reward: {total_reward}")

if __name__ == "__main__":
    visualize()
