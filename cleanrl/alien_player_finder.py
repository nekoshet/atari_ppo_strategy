import gymnasium as gym
import numpy as np
from numba import jit
import random
from matplotlib import pyplot as plt

PLAYER_GREEN_VALUE = 144
GREEN_RGB_INDEX = 1
MIN_PLAYER_HEIGHT = 7
PLAYER_HEIGHT = 10
MAX_PLAYER_ROW = 172

@jit(nopython=True)
def find_vertical_line(img, target_g_value, k):
    # get height, width
    height, width = img.shape

    # Early return if k is larger than image height
    if k > height:
        return -1, -1

    # Iterate through each column
    for x in range(width):
        # Count consecutive matches
        current_count = 0

        # Check each pixel in the column
        for y in range(height):
            # Check value
            if img[y, x] == target_g_value:
                current_count += 1

                # If we found k consecutive matches
                if current_count == k:
                    # Return coordinates of the start of the sequence
                    return y - k + 1, x
            else:
                # Reset counter if we find a non-matching pixel
                current_count = 0

    # If no sequence found
    return -1, -1

class AlienPlayerFinder(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.last_player_position = np.zeros(2, dtype=int)

    @staticmethod
    def _find_player_location(observation):
        pos = find_vertical_line(observation[:, :, GREEN_RGB_INDEX], PLAYER_GREEN_VALUE, MIN_PLAYER_HEIGHT)
        if pos != (-1, -1):
            pos = pos[0] + int(PLAYER_HEIGHT / 2), pos[1]
        return pos if pos != (-1, -1) else None

    def _set_player_location(self, observation):
        player_pos = self._find_player_location(observation[:MAX_PLAYER_ROW])
        if player_pos is not None:
            self.last_player_position = player_pos

        # if random.random() < 0.01:
        #     print(self.last_player_position)
        #     plt.imshow(observation)
        #     plt.show(block=True)

        return self.last_player_position

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info['focus_pos'] = self._set_player_location(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info['focus_pos'] = self._set_player_location(obs)
        return obs, reward, terminated, truncated, info
