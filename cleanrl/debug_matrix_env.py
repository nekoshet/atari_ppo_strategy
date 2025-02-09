import gymnasium as gym
import numpy as np
from key_frame_wrapper import KeyFrame
from focus_window_wrapper import FocusWindowWrapper
from display_observation_wrapper import DisplayObservation


class DebugEnv(gym.Env):
    def __init__(self, rows=9, cols=9):
        super().__init__()

        # Define action and observation spaces
        self.action_space = gym.spaces.Discrete(1)  # Single no-op action
        self.observation_space = gym.spaces.Box(
            low=0, high=np.inf, shape=(rows, cols), dtype=np.float32
        )

        # Environment parameters
        self.rows = rows
        self.cols = cols
        self.step_counter = 0

    def _get_obs(self):
        # Create observation matrix with (step_counter, row, column)
        obs = np.zeros((self.rows, self.cols, 3), dtype=np.uint8)
        for r in range(self.rows):
            for c in range(self.cols):
                obs[r, c] = (self.step_counter, r, c)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_counter = 0

        # Create observation matrix with (step_counter, row, column)
        obs = self._get_obs()

        return obs, {'focus_pos': np.array([5, 5])}

    def step(self, action):
        # Increment step counter
        self.step_counter += 1

        # Create observation matrix
        obs = self._get_obs()

        # Always terminated after one step for debugging
        terminated = False
        truncated = False

        return obs, 0.0, terminated, truncated, {'focus_pos': np.array([5, 5])}

    def render(self):
        # Optional render method
        pass


# Example usage
def main():
    env = DebugEnv()
    env = KeyFrame(env, 3)
    env = FocusWindowWrapper(env, 5)
    env = DisplayObservation(env)
    obs, _ = env.reset()

    for _ in range(5):
        print(obs[:, :, 2])
        obs, reward, terminated, truncated, _ = env.step(0)

if __name__ == "__main__":
    main()
