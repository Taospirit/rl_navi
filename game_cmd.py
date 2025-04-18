import json
import time
import pygame
from env import RobotEnv

env_config = 'configs/env_config.json'
need_render = 1
env = RobotEnv(env_config, render=need_render)

env.reset()
running = True
rews = 0
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
    print(rews, done, info)
    rews += reward
    if need_render:
        env.render()
    if done:
        rews = 0
