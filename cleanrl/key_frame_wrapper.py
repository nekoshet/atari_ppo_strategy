
import numpy as np

import gymnasium as gym
from gymnasium.spaces import Box

from matplotlib import pyplot as plt


class KeyFrame(gym.ObservationWrapper, gym.utils.RecordConstructorArgs):
    """Observation wrapper that stacks the observations in a rolling manner.

    For example, if the number of stacks is 4, then the returned observation contains
    the most recent 4 observations. For environment 'Pendulum-v1', the original observation
    is an array with shape [3], so if we stack 4 observations, the processed observation
    has shape [4, 3].

    Note:
        - To be memory efficient, the stacked observations are wrapped by :class:`LazyFrame`.
        - The observation space must be :class:`Box` type. If one uses :class:`Dict`
          as observation space, it should apply :class:`FlattenObservation` wrapper first.
        - After :meth:`reset` is called, the frame buffer will be filled with the initial observation.
          I.e. the observation returned by :meth:`reset` will consist of `num_stack` many identical frames.

    Example:
        >>> import gymnasium as gym
        >>> from gymnasium.wrappers import FrameStack
        >>> env = gym.make("CarRacing-v2")
        >>> env = FrameStack(env, 4)
        >>> env.observation_space
        Box(0, 255, (4, 96, 96, 3), uint8)
        >>> obs, _ = env.reset()
        >>> obs.shape
        (4, 96, 96, 3)
    """

    def __init__(
        self,
        env: gym.Env,
        key_frame_interval: int
    ):
        """Observation wrapper that stacks the observations in a rolling manner.

        Args:
            env (Env): The environment to apply the wrapper
            num_stack (int): The number of frames to stack
            lz4_compress (bool): Use lz4 to compress the frames internally
        """
        gym.utils.RecordConstructorArgs.__init__(
            self, key_frame_interval=key_frame_interval
        )
        gym.ObservationWrapper.__init__(self, env)

        self.key_frame_interval = key_frame_interval

        self.last_key_frame = None
        self.step_count = None

        low = np.repeat(self.observation_space.low[np.newaxis, ...], 2, axis=0)
        high = np.repeat(
            self.observation_space.high[np.newaxis, ...], 2, axis=0
        )
        self.observation_space = Box(
            low=low, high=high, dtype=self.observation_space.dtype
        )

    @staticmethod
    def show_observation(obs):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 10))

        # Plot first matrix
        obs = [np.swapaxes(obs, 0, -1)[:, :, :3] for obs in obs]
        obs = [np.swapaxes(obs, 0, 1) for obs in obs]
        ax1.imshow(obs[0])
        ax1.set_title('First Matrix')

        # Plot second matrix
        ax2.imshow(obs[1])
        ax2.set_title('Second Matrix')

        # Adjust layout and display
        plt.tight_layout()
        plt.show()

    def observation(self, observation):
        """Converts the wrappers current frames to lazy frames.

        Args:
            observation: Ignored

        Returns:
            :class:`LazyFrames` object for the wrapper's frame buffer,  :attr:`self.frames`
        """
        output = np.stack([self.last_key_frame, observation], axis=0)
        # \self.show_observation(output)
        return output

    def step(self, action):
        """Steps through the environment, appending the observation to the frame buffer.

        Args:
            action: The action to step through the environment with

        Returns:
            Stacked observations, reward, terminated, truncated, and information from the environment
        """
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1
        if self.step_count % self.key_frame_interval == 0:
            self.last_key_frame = observation
        return self.observation(observation), reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Reset the environment with kwargs.

        Args:
            **kwargs: The kwargs for the environment reset

        Returns:
            The stacked observations
        """
        obs, info = self.env.reset(**kwargs)
        self.step_count = 0

        self.last_key_frame = obs

        return self.observation(obs), info
