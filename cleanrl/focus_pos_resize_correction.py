import gymnasium as gym
import numpy as np

class FocusPosResizeCorrection(gym.Wrapper):
    def __init__(self, env, new_size):
        super().__init__(env)
        self.new_size = new_size

    def _get_adjusted_focus_pos(self, obs, focus_pos):
        ratio = np.array(self.new_size) / obs.shape[:2]
        return (focus_pos * ratio).astype(int)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info['focus_pos'] = self._get_adjusted_focus_pos(obs, info['focus_pos'])
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info['focus_pos'] = self._get_adjusted_focus_pos(obs, info['focus_pos'])
        return obs, reward, terminated, truncated, info
