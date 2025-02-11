from typing import Any, SupportsFloat
import gymnasium as gym
import numpy as np
from gymnasium.core import WrapperObsType, WrapperActType

PLAYER_COLOR = np.array([144, 112, 252], dtype=np.uint8)
MIN_PLAYER_HEIGHT = 4

class AlienPlayerFinder(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.last_player_position = np.zeros(2, dtype=int)

    def _find_player(self, observation):
        # Create a boolean mask where True indicates matching RGB values
        matches = np.all(observation == PLAYER_COLOR, axis=2)

        # get height and width
        height, width = observation.shape[:2]

        # For each column
        for x in range(width):
            # For each possible starting position in the column
            for y in range(height - MIN_PLAYER_HEIGHT + 1):
                # Check if k consecutive pixels match
                if np.all(matches[y:y + MIN_PLAYER_HEIGHT, x]):
                    self.last_player_position = np.array([x, y])
                    return True
        return False

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        pass

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        pass
