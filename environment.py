from timeit import repeat
import ale_py
from collections import deque
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import (
    FrameStackObservation,
    GrayscaleObservation,
    ResizeObservation,
)


class MaxAndSkipEnv(gym.Wrapper):
    """
    Return only every 4th frame (skipping) and return the max between the two
    last frames to handle flickering (Paper Section 5).
    """

    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        obs_buffer = deque(maxlen=2)
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            obs_buffer.append(obs)
            total_reward += reward
            done = terminated or truncated
            if done:
                break

        max_frame = np.max(np.stack(obs_buffer), axis=0)
        return max_frame, total_reward, done, truncated, info


def make_env(env_name, render_mode="rgb_array"):
    """
    Applies the specific preprocessing steps from Section 4.1.
    1. Grayscale (Source [156])
    2. Resize to 84x84 (Source [157])
    3. Stack 4 Frames (Source [159])
    """
    env = gym.make(env_name, render_mode=render_mode, repeat_action_probability=0.0)
    env = MaxAndSkipEnv(env, skip=4)  # Source [187]
    env = GrayscaleObservation(env, keep_dim=False)  # Source [156]
    env = ResizeObservation(env, (84, 84))  # Source [157]
    env = FrameStackObservation(env, stack_size=4)  # Source [159]
    return env
