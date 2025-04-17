import json
from env import RobotEnv
import time
import numpy as np
import torch
from collections import namedtuple
from net import MultiDiscreteActor, Critic
from policy import PPO


train_seed = 42
max_timesteps = int(1e8)
torch.manual_seed(train_seed)
torch.cuda.manual_seed(train_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

env_config = 'configs/env_config.json'
need_render = 1
env = RobotEnv(env_config, render=need_render)

actor = MultiDiscreteActor(env.state_dim, env.action_dim)
critic = Critic(env.state_dim)
agent = PPO(actor, critic)

live_time = []
total_steps = 0
cnt = 0
obs = env.reset()
while total_steps < max_timesteps:
    max_len = agent.data_size()
    rews = 0
    step = 0
    for _ in range(max_len):
        act, log_prob = agent.act(obs)
        next_obs, rew, done, info = env.step(act)
        total_steps += 1
        agent.store({
            "obs": obs,
            "act": act,
            "rew": rew,
            "next_obs": next_obs,
            "done": done,
            "log_prob": log_prob
        })
        # print(obs, act, rew, done, log_prob)
        rews += rew
        step += 1
        if done:
            break
        obs = next_obs
        if need_render:
            env.render()
    if agent.ready():
        agent.update()
    cnt += 1
    print(f'learn {cnt}, rews {rews}, step {step}')
