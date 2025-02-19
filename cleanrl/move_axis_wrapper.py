import gymnasium as gym
import numpy as np
from gymnasium.core import ObsType, WrapperObsType

from ppo_atari_utils import get_surrounding_window
from gymnasium.spaces import Box

class MoveAxisWrapper(gym.ObservationWrapper):
    def __init__(self, env, source_axis, target_axis):
        super().__init__(env)
        self.source_axis = source_axis
        self.target_axis = target_axis
        self.observation_space = Box(
            low=np.moveaxis(self.observation_space.low, self.source_axis, self.target_axis),
            high=np.moveaxis(self.observation_space.high, self.source_axis, self.target_axis),
            dtype=self.observation_space.dtype
        )

    def observation(self, observation: ObsType) -> WrapperObsType:
        return np.moveaxis(observation, self.source_axis, self.target_axis)
