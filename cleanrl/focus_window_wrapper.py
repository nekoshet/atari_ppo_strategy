import gymnasium as gym
import numpy as np
from ppo_atari_utils import get_surrounding_window
from gymnasium.spaces import Box

class FocusWindowWrapper(gym.Wrapper):
    def __init__(self, env, window_size):
        super().__init__(env)
        self.window_size = window_size
        window_low = self.observation_space.low[1]
        window_high = self.observation_space.low[1]
        new_low = np.concatenate([self.env.observation_space.low, window_low[None, ...]], axis=0)
        new_high = np.concatenate([self.env.observation_space.high, window_high[None, ...]], axis=0)
        self.observation_space = Box(
            low=new_low, high=new_high, dtype=self.observation_space.dtype
        )
        pass

    def observation(self, obs, info):
        rel_obs = obs[1]
        i, j = info['focus_pos']
        window = get_surrounding_window(rel_obs, i, j, self.window_size)
        window_obs = np.zeros_like(rel_obs)
        window_obs[:window.shape[0], :window.shape[1]] = window
        window_obs = np.expand_dims(window_obs, axis=0)
        output = np.concatenate([obs, window_obs], axis=0)
        return output

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs, info), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.observation(obs, info), reward, terminated, truncated, info
