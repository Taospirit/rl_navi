import os
import time
import numpy as np
import logging
import torch
from datetime import datetime
from collections import namedtuple
from net import MultiDiscreteActor, Critic
from policy import PPO
from env import RobotEnv

env_config = 'configs/env_config_1.json'
need_render = 0
env = RobotEnv(env_config, render=need_render)

obs = env.reset()
for _ in range(100):
    act = env.get_rand_act()
    next_obs, rew, done, info = env.step(act)
    print(next_obs.shape, rew, done, info)