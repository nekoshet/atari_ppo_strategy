import random

import gymnasium as gym
import numpy as np
from gymnasium.core import ObsType, WrapperObsType
from matplotlib import pyplot as plt


class DisplayObservation(gym.ObservationWrapper):
    def observation(self, observation: ObsType) -> WrapperObsType:
        self.show_images(observation)
        return observation

    @staticmethod
    def show_image(observation: ObsType):
        plt.matshow(observation)
        plt.tight_layout()
        plt.show(block=True)

    @staticmethod
    def show_collector_3(observation: ObsType):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for obs, ax in zip(observation, axes):
            obs = np.concatenate([obs, np.zeros_like(obs[..., 0])[..., None]], axis=-1)
            ax.matshow(obs * 100)
        plt.tight_layout()
        plt.show(block=True)

    @staticmethod
    def show_images(observation: ObsType):
        fig, axes = plt.subplots(1, 3, figsize=(12, 5))
        for obs, ax in zip(observation, axes):
            ax.matshow(obs)
        plt.tight_layout()
        plt.show(block=True)
