import gymnasium as gym
from gymnasium.core import ObsType, WrapperObsType
from matplotlib import pyplot as plt


class DisplayObservation(gym.ObservationWrapper):
    def observation(self, observation: ObsType) -> WrapperObsType:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for obs, ax in zip(observation, axes):
            ax.matshow(obs)
        plt.tight_layout()
        plt.show()
        return observation
