from timeit import repeat
import ale_py
from collections import deque
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import (
    FrameStackObservation,
    GrayscaleObservation,
    ResizeObservation,
    TransformReward,
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


class EpisodicLifeEnv(gym.Wrapper):
    """
    Make end-of-life == end-of-episode, but only reset on true game over.
    Done by DeepMind for the DQN and co.
    """

    def __init__(self, env):
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        self.was_real_done = done

        lives = info.get("lives", 0)
        if lives < self.lives and lives > 0:
            # A life was lost, but the game is not over.
            # We signal 'terminated' to the agent so it learns this is bad/end of segment.
            terminated = True

        self.lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            # If we are here, it means the agent "finished" an episode because it lost a life,
            # but the environment is not actually done. We need to continue.
            # Usually, we just step with 'NOOP' (0) to advance.
            obs, _, terminated, truncated, info = self.env.step(0)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)

        self.lives = info.get("lives", 0)
        return obs, info


class FireResetEnv(gym.Wrapper):
    """
    Take action on reset for environments that are fixed until firing.
    """

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Fire action is usually 1
        obs, _, terminated, truncated, info = self.env.step(1)
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        return obs, info


class RecordOriginalReward(gym.Wrapper):
    """
    Record the original reward returned by the underlying env into the info dict
    so that later wrappers (e.g., TransformReward) can modify the reward while we
    still access the unmodified reward for diagnostics/logging.
    """

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info) if info is not None else {}
        info["original_reward"] = float(reward)
        return obs, reward, terminated, truncated, info


def make_env(env_name, render_mode="rgb_array", difficulty=0, mode=0):
    """
    Applies the specific preprocessing steps from Section 4.1. of the DQN paper:
    1. Grayscale
    2. Resize to 84x84
    3. Stack 4 Frames
    """
    env = gym.make(
        env_name,
        render_mode=render_mode,
        difficulty=difficulty,
        mode=mode,
        repeat_action_probability=0.0,
    )
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)

    # Only apply FireResetEnv if the action space implies it (usually action 1 is FIRE)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)

    # Record original reward for logging before applying any transforms
    env = RecordOriginalReward(env)
    # Clip rewards to {-1, 0, +1} using TransformReward with sign function
    env = TransformReward(env, lambda reward: np.sign(reward))
    env = GrayscaleObservation(env, keep_dim=False)
    env = ResizeObservation(env, (84, 84))
    env = FrameStackObservation(env, stack_size=4)
    return env
