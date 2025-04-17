import json
from env import RobotEnv
import time
import numpy as np
import torch


env_config = 'configs/env_config.json'

show = 0
env = RobotEnv(env_config, render=show)
obs = env.reset()
for _ in range(100):
    act = np.random.choice([0, 1, -1], size=2)
    next_obs, reward, done, info = env.step(act)
    print(next_obs, act, done)
    if show:
        env.render()
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

#     act = [move, rotate]
#     next_obs, reward, done, info = env.step(act)
#     env.render()

