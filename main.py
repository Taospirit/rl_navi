import json
import pygame
from env import RobotEnv
import time
import numpy as np

with open('config.json') as f:
    config = json.load(f)

show = True
env = RobotEnv(config, show)
obs = env.reset()
for _ in range(10000):
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

