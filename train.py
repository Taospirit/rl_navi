import json
from env import RobotEnv
import time
import numpy as np
import torch
import pygame
from collections import namedtuple
from net import MultiDiscreteActor, Critic
from policy import PPO


env_config = 'configs/env_config.json'

need_render = 0
env = RobotEnv(env_config, render=need_render)

# obs = env.reset()
# for _ in range(1):
#     act = np.random.choice([0, 1, -1], size=2)
#     next_obs, reward, done, info = env.step(act)
#     print(next_obs, act, done)
#     if need_render:
#         env.render()

train_seed = 42
max_timesteps = int(1e8)
torch.manual_seed(train_seed)
torch.cuda.manual_seed(train_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

actor = MultiDiscreteActor(env.state_dim, env.action_dim)
critic = Critic(env.state_dim)
# agent = PPO(model(actor, critic), **args.algo.to_dict())
agent = PPO(actor, critic)

# env.reset()
# running = True
# while running:
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             running = False

#     keys = pygame.key.get_pressed()
#     move = 0
#     rotate = 0
#     if keys[pygame.K_w]:
#         move = 1
#     elif keys[pygame.K_s]:
#         move = -1
#     if keys[pygame.K_a]:
#         rotate = 1
#     elif keys[pygame.K_d]:
#         rotate = -1

#     act = [move + 1, rotate + 1]
#     next_obs, reward, done, info = env.step(act)
#     if need_render:
#         env.render()
#     print(reward, done, info)

live_time = []
total_steps = 0
cnt = 0
obs = env.reset()
while total_steps < max_timesteps:
    max_len = agent.data_size()
    rews = 0
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
        # print(rew)
        rews += rew
        if done:
            break
        obs = next_obs
        if need_render:
            env.render()
    agent.update()
    cnt += 1
    print(f'learn {cnt}, rews {rews}')
