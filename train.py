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

now = datetime.now()
time_format = now.strftime("%m%d-%H%M%S")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(f"train-{time_format}.log"),  # 保存到文件
        logging.StreamHandler()          # 同时输出到控制台
    ]
)
save_dir = f'saves-{time_format}'


train_seed = 42
max_timesteps = int(1e8)
torch.manual_seed(train_seed)
torch.cuda.manual_seed(train_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

env_config = 'configs/env_config.json'
need_render = 0
env = RobotEnv(env_config, render=need_render)

actor = MultiDiscreteActor(env.state_dim, env.action_dim)
critic = Critic(env.state_dim)
agent = PPO(actor, critic)

total_steps = 0
cnt = 0
obs = env.reset()
max_len = agent.data_size()
while total_steps < max_timesteps:
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
        obs = next_obs
        # print(obs, act, rew, done, log_prob)
        rews += rew
        step += 1

        if total_steps % 10000 == 0:
            save_path = f'{save_dir}/save_model_{total_steps}.pth'
            agent.save_model(save_path)

        if env.get_step_cnt() > 5000:
            obs = env.reset()
        if need_render:
            env.render()
        if done:
            break
    if agent.ready():
        cnt += 1
        agent.update()
    logging.info(f'learn {cnt}, rews {rews}, roll step {step}, total step {total_steps}')