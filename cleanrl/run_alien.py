# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_ataripy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from collector_env import CollectorEnv
from key_frame_wrapper import KeyFrame
from focus_window_wrapper import FocusWindowWrapper
from display_observation_wrapper import DisplayObservation
from alien_player_finder import AlienPlayerFinder
from focus_pos_resize_correction import FocusPosResizeCorrection
from move_axis_wrapper import MoveAxisWrapper

# constants
ATARI_KEY_FRAME_INTERVAL = 8
ATARI_WINDOW_SIZE = 31

def make_atari_env():
    env = gym.make('ALE/Alien-v5', frameskip=1)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = ClipRewardEnv(env)
    env = AlienPlayerFinder(env)
    env = FocusPosResizeCorrection(env, (84, 84))
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayScaleObservation(env)
    env = gym.wrappers.FrameStack(env, 3)
    env = MoveAxisWrapper(env, 0, -1)
    env = KeyFrame(env, ATARI_KEY_FRAME_INTERVAL)
    env = FocusWindowWrapper(env, ATARI_WINDOW_SIZE)
    env = DisplayObservation(env)
    return env


def main():
    # Create environment with human rendering
    env = make_atari_env()

    # Reset and run the environment
    obs, _ = env.reset()

    while True:
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        if done:
            break

    env.close()


if __name__ == '__main__':
    main()
