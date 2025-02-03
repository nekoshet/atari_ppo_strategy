import gymnasium as gym
import numpy as np
from cleanrl.ppo_atari_utils import get_surrounding_window

class FocusWindowWrapper(gym.Wrapper):
    def __init__(self, env, window_size):
        super().__init__(env)
        self.window_size = window_size

    def observation(self, obs, info):
        rel_obs = obs[1]
        i, j = info['focus_pos']
        window = get_surrounding_window(rel_obs, i, j, self.window_size)
        window_obs = np.zeros_like(rel_obs)
        window_obs[:window.shape[0], :window.shape[1]] = window
        output = np.concatenate([obs, window_obs], axis=0)
        return output

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs, info), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.observation(obs, info), reward, terminated, truncated, info
