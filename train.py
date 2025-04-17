import json
from env import RobotEnv
import time
import numpy as np
import torch
import pygame
from tool import load_config
from collections import namedtuple
# from net import ActorNet, CriticNet
from net import MultiDiscreteActor, Critic
from policy import PPO


env_config = 'configs/env_config.json'

show = 0
env = RobotEnv(env_config, render=show)
obs = env.reset()
for _ in range(1):
    act = np.random.choice([0, 1, -1], size=2)
    next_obs, reward, done, info = env.step(act)
    print(next_obs, act, done)
    if show:
        env.render()

train_config = 'configs/train_config.yaml'

args = load_config(train_config)
torch.manual_seed(args.train_seed)
torch.cuda.manual_seed(args.train_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

actor = MultiDiscreteActor(env.state_dim, env.action_dim)
critic = Critic(env.state_dim)
# agent = PPO(model(actor, critic), **args.algo.to_dict())
agent = PPO(actor, critic)

live_time = []
total_steps = 0
cnt = 0


env.reset()
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()
    move = 0
    rotate = 0
    if keys[pygame.K_w]:
        move = 1
    elif keys[pygame.K_s]:
        move = -1
    if keys[pygame.K_a]:
        rotate = 1
    elif keys[pygame.K_d]:
        rotate = -1

    act = [move + 1, rotate + 1]
    next_obs, reward, done, info = env.step(act)
    env.render()
    print(reward, done, info)

# obs = env.reset()
# while total_steps < args.max_timesteps:
#     max_len = agent.data_size()
#     rews = 0
#     for _ in range(max_len):
#         act, log_prob = agent.act(obs)
#         next_obs, rew, done, info = env.step(act)
#         total_steps += 1
#         agent.store({
#             "obs": obs,
#             "act": act,
#             "rew": rew,
#             "next_obs": next_obs,
#             "done": done,
#             "log_prob": log_prob
#         })
#         # print(obs, act, rew, done, log_prob)
#         # print(rew)
#         rews += rew
#         if done:
#             break
#         obs = next_obs
#     agent.update()
#     cnt += 1
#     print(f'learn {cnt}, rews {rews}')
