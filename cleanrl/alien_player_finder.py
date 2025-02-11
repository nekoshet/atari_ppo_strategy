import gymnasium as gym
import numpy as np

PLAYER_COLOR = np.array([132, 144, 252], dtype=np.uint8)
MIN_PLAYER_HEIGHT = 4
MAX_PLAYER_ROW = 172

class AlienPlayerFinder(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.last_player_position = np.zeros(2, dtype=int)

    @staticmethod
    def _find_player_in_box(observation, min_row, max_row, min_column, max_column):
        # Create a boolean mask where True indicates matching RGB values
        matches = np.all(observation == PLAYER_COLOR, axis=2)

        # For each column
        for row in range(min_row, max_row - MIN_PLAYER_HEIGHT + 1):
            for column in range(min_column, max_column):
                if np.all(matches[row:row + MIN_PLAYER_HEIGHT, column]):
                    return np.array([row, column])
        return None

    def _find_player(self, observation):
        # first look near last location
        top = max(self.last_player_position[0] - 10, 0)
        bottom = max(self.last_player_position[0] + 20, MAX_PLAYER_ROW)
        left = min(self.last_player_position[1] - 10, 0)
        right = min(self.last_player_position[1] + 10, observation.shape[1])
        player_pos = self._find_player_in_box(observation, top, bottom, left, right)
        # if not found, look in all window
        if player_pos is None:
            player_pos = self._find_player_in_box(observation, 0, MAX_PLAYER_ROW,
                                                  0, observation.shape[1])
        return player_pos

    def _set_player_location(self, observation):
        player_pos = self._find_player(observation)
        if player_pos is not None:
            self.last_player_position = player_pos

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._set_player_location(obs)
        info['focus_pos'] = self.last_player_position
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._set_player_location(obs)
        info['focus_pos'] = self.last_player_position
        return obs, reward, terminated, truncated, info
