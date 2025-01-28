import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pygame

class CollectorEnv(gym.Env):
    """
    A custom Gymnasium environment where an agent moves in a grid world collecting coins.

    Args:
        size (int): The size of the square grid world (N x N)
    """

    metadata = {
        "render_modes": ["human", "console"],
        "render_fps": 4,
    }

    def __init__(self, size=5, max_steps=100, seed=1, render_mode=None):
        super(CollectorEnv, self).__init__()

        assert render_mode is None or render_mode in self.metadata["render_modes"]

        # save args
        self.render_mode = render_mode
        self.size = size
        self.max_steps = max_steps
        self.seed = seed

        # rendering stuff
        self.window = None
        self.clock = None
        self.cell_size = 60  # pixels
        self.window_size = size * self.cell_size
        # Colors for rendering
        self.COLORS = {
            0: (255, 255, 255),  # Empty - White
            1: (0, 0, 255),      # Player - Blue
            2: (255, 215, 0)     # Coin - Gold
        }

        # Action space: 0=up, 1=right, 2=down, 3=left
        self.action_space = spaces.Discrete(4)

        # Observation space: matrix of size N x N with:
        # 0 = empty space
        # 1 = player
        # 2 = coin
        self.observation_space = spaces.Box(
            low=0,
            high=2,
            shape=(size, size),
            dtype=np.uint8
        )

        # Initialize state variables
        self.steps = None
        self.player_position = None
        self.coin_position = None
        self.rng = None

        # Initialize empty grid
        self.grid = np.zeros((self.size, self.size), dtype=np.uint8)

    def _set_grid(self):
        self.grid.fill(0)
        self.grid[tuple(self.player_position)] = 1
        self.grid[tuple(self.coin_position)] = 2

    def _place_new_coin(self):
        """Place a new coin in a random empty position."""
        self.coin_position = None
        while self.coin_position is None:
            new_coin_position = self.rng.integers(0, self.size, size=2)
            if not np.array_equal(new_coin_position, self.player_position):
                self.coin_position = new_coin_position

    def reset(self):
        """Reset the environment to initial state."""
        super().reset(seed=self.seed)

        # reset steps
        self.steps = 0

        # reset rng
        self.rng = np.random.default_rng(seed=self.seed)

        # Place player randomly
        self.player_position = np.array([
            self.rng.integers(0, self.size),
            self.rng.integers(0, self.size)
        ])

        # Place coin in random position (not where player is)
        self._place_new_coin()

        # set grid
        self._set_grid()

        # Initialize pygame if using human render mode
        if self.render_mode == "human" and self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            self.clock = pygame.time.Clock()

        # Render the initial state if needed
        if self.render_mode == "human":
            self._render_frame()

        return self.grid.copy(), {}

    def step(self, action):
        """Execute one time step within the environment."""
        # Initialize reward and done flag
        reward = 0

        # Calculate new position based on action
        if action == 0:  # up
            self.player_position[0] = max(0, self.player_position[0] - 1)
        elif action == 1:  # right
            self.player_position[1] = min(self.size - 1, self.player_position[1] + 1)
        elif action == 2:  # down
            self.player_position[0] = min(self.size - 1, self.player_position[0] + 1)
        elif action == 3:  # left
            self.player_position[1] = max(0, self.player_position[1] - 1)

        # Check if new position has coin
        if np.array_equal(self.player_position, self.coin_position):
            reward = 1

        # Place new coin if collected
        if reward > 0:
            self._place_new_coin()

        # inc steps
        self.steps += 1
        done = (self.steps >= self.max_steps)

        # set grid
        self._set_grid()

        # Render the frame if in human mode
        if self.render_mode == "human":
            self._render_frame()

        return self.grid.copy(), reward, done, False, {}

    def _render_frame(self):
        """Render the current frame using pygame."""
        if self.window is None:
            return

        # Create surface from state matrix
        surface = pygame.Surface((self.window_size, self.window_size))

        # Draw each cell
        for i in range(self.size):
            for j in range(self.size):
                pygame.draw.rect(
                    surface,
                    self.COLORS[self.grid[i, j]],
                    (j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size)
                )

        # Copy to window and update display
        self.window.blit(surface, (0, 0))
        pygame.display.flip()

        # Maintain constant game speed
        self.clock.tick(self.metadata["render_fps"])

    def render(self):
        """Render the environment."""
        if self.render_mode == "console":
            # Console rendering
            for row in self.grid:
                print(' '.join(['P' if cell == 1 else 'C' if cell == 2 else '.' for cell in row]))
            print(f"Step: {self.step}/{self.max_steps}")
            print()
        elif self.render_mode == "human":
            return self._render_frame()

    def close(self):
        """Clean up resources."""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None


def main():
    # Create environment with human rendering
    env = CollectorEnv(size=3, seed=42, render_mode="human")

    # Reset and run the environment
    obs, _ = env.reset()

    while True:
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)

        if reward > 0:
            print("Coin collected!")

        if done:
            break

    env.close()


if __name__ == '__main__':
    main()
