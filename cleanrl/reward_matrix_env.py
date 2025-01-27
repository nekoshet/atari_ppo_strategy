import gymnasium as gym
import numpy as np
from gymnasium import spaces
import time


class RewardMatrixEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 10,
    }

    def __init__(self, render_mode=None):
        super().__init__()

        # Constants
        self.size = 84
        self.silver_density = 0.04
        self.gold_density = 0.01

        # Action and observation spaces
        self.action_space = spaces.Discrete(4)  # Up, Right, Down, Left
        self.observation_space = spaces.Box(
            low=0, high=2, shape=(self.size, self.size), dtype=np.int8
        )

        # Rendering setup
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.font = None

        # Initialize state variables
        self.reset()

    def update_grid(self):
        self.grid.fill(0)
        self.grid[tuple(self.player_pos)] = 1
        for pos in self.silver_positions:
            self.grid[pos] = 2

        for pos in self.gold_positions:
            self.grid[pos] = 3

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Create empty grid
        self.grid = np.zeros((self.size, self.size), dtype=np.int8)

        # Place player
        self.player_pos = self.np_random.integers(0, self.size, size=2)

        # Place silver coins (value 1)
        num_silver = int(self.size * self.size * self.silver_density)
        self.silver_positions = set()
        while len(self.silver_positions) < num_silver:
            pos = tuple(self.np_random.integers(0, self.size, size=2))
            if pos != tuple(self.player_pos):
                self.silver_positions.add(pos)

        # Place gold coins (value 2)
        num_gold = int(self.size * self.size * self.gold_density)
        self.gold_positions = set()
        while len(self.gold_positions) < num_gold:
            pos = tuple(self.np_random.integers(0, self.size, size=2))
            if pos != tuple(self.player_pos) and pos not in self.silver_positions:
                self.gold_positions.add(pos)

        self.step_count = 0
        self.score = 0

        self.update_grid()

        return self.grid.copy(), {}

    def step(self, action):
        # Move player
        if action == 0:  # Up
            self.player_pos[0] = max(0, self.player_pos[0] - 1)
        elif action == 1:  # Right
            self.player_pos[1] = min(self.size - 1, self.player_pos[1] + 1)
        elif action == 2:  # Down
            self.player_pos[0] = min(self.size - 1, self.player_pos[0] + 1)
        elif action == 3:  # Left
            self.player_pos[1] = max(0, self.player_pos[1] - 1)

        # Collect coins
        pos_tuple = tuple(self.player_pos)
        reward = 0
        if pos_tuple in self.silver_positions:
            reward = 1
            self.silver_positions.remove(pos_tuple)
        elif pos_tuple in self.gold_positions:
            reward = 10
            self.gold_positions.remove(pos_tuple)

        self.score += reward
        self.step_count += 1

        # Move coins every 2 steps
        if self.step_count % 2 == 0:
            # Move silver coins
            new_silver_positions = set()
            for pos in self.silver_positions:
                new_pos = list(pos)
                direction = self.np_random.integers(0, 4)
                if direction == 0:  # Up
                    new_pos[0] = max(0, new_pos[0] - 1)
                elif direction == 1:  # Right
                    new_pos[1] = min(self.size - 1, new_pos[1] + 1)
                elif direction == 2:  # Down
                    new_pos[0] = min(self.size - 1, new_pos[0] + 1)
                elif direction == 3:  # Left
                    new_pos[1] = max(0, new_pos[1] - 1)
                new_silver_positions.add(tuple(new_pos))
            self.silver_positions = new_silver_positions

            # Move gold coins
            new_gold_positions = set()
            for pos in self.gold_positions:
                new_pos = list(pos)
                direction = self.np_random.integers(0, 4)
                if direction == 0:  # Up
                    new_pos[0] = max(0, new_pos[0] - 1)
                elif direction == 1:  # Right
                    new_pos[1] = min(self.size - 1, new_pos[1] + 1)
                elif direction == 2:  # Down
                    new_pos[0] = min(self.size - 1, new_pos[0] + 1)
                elif direction == 3:  # Left
                    new_pos[1] = max(0, new_pos[1] - 1)
                new_gold_positions.add(tuple(new_pos))
            self.gold_positions = new_gold_positions

        self.update_grid()

        if self.render_mode == "human":
            self.render()

        return self.grid.copy(), reward, False, False, {"score": self.score}

    def render(self):
        if self.render_mode == "human":
            try:
                import pygame

                if self.window is None:
                    pygame.init()
                    pygame.display.init()
                    self.window = pygame.display.set_mode(
                        (self.size * 10, self.size * 10 + 30))  # Added height for score
                    self.clock = pygame.time.Clock()
                    self.font = pygame.font.Font(None, 36)  # Initialize font

                canvas = pygame.Surface((self.size * 10, self.size * 10 + 30))  # Added height for score
                canvas.fill((255, 255, 255))  # White background

                # Draw grid
                for i in range(self.size):
                    for j in range(self.size):
                        if self.grid[i, j] == 1:  # Player
                            pygame.draw.circle(canvas, (255, 0, 0),
                                               (j * 10 + 5, i * 10 + 5), 4)
                        elif self.grid[i, j] == 2:  # Silver
                            pygame.draw.circle(canvas, (192, 192, 192),
                                               (j * 10 + 5, i * 10 + 5), 4)
                        elif self.grid[i, j] == 3:  # Gold
                            pygame.draw.circle(canvas, (255, 215, 0),
                                               (j * 10 + 5, i * 10 + 5), 4)

                # Draw score
                score_text = self.font.render(f'Score: {self.score}', True, (0, 0, 0))
                canvas.blit(score_text, (10, self.size * 10 + 5))

                self.window.blit(canvas, canvas.get_rect())
                pygame.event.pump()
                pygame.display.flip()
                self.clock.tick(self.metadata["render_fps"])

            except ImportError:
                print("pygame is not installed, cannot render in human mode")

    def close(self):
        if self.window is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()

def main():
    env = RewardMatrixEnv(render_mode="human")  # or None for no rendering
    observation, info = env.reset()

    while True:
        action = env.action_space.sample()  # Replace with your agent's action
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated:
            break

    env.close()


if __name__ == '__main__':
    main()
