import gymnasium as gym
import numpy as np
from gymnasium.core import ObsType, WrapperObsType
from matplotlib import pyplot as plt


class DisplayObservation(gym.ObservationWrapper):
    def observation(self, observation: ObsType) -> WrapperObsType:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for obs, ax in zip(observation, axes):
            obs = np.concatenate([obs, np.zeros_like(obs[..., 0])[..., None]], axis=-1)
            ax.matshow(obs * 100)
        plt.tight_layout()
        plt.show(block=True)
        return observation

    def observation2(self, observation: ObsType) -> WrapperObsType:
        plt.matshow(observation * 10)
        plt.tight_layout()
        plt.show(block=True)
        return observation
