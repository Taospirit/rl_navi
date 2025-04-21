import os
import time
import numpy as np
import logging
import torch
from datetime import datetime
from collections import namedtuple
from core.net import PPOActor, PPOCritic
from core.net import SACActor, SACCritic
from core.policy import PPO, SAC
from core.env import RobotEnv
from core.utils import load_config

# 加载配置文件
train_config = 'configs/train_config.yaml'
env_config = 'configs/env_config_1.yaml'
config = load_config(train_config)

# 设置保存路径
now = datetime.now()
time_format = now.strftime("%m%d-%H%M%S")
save_root = config.train.save_root
os.makedirs(save_root, exist_ok=True)
model_save_dir = f'{save_root}/saves-{time_format}'
log_save_file = f"{save_root}/train-{time_format}.log"

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(log_save_file),
        logging.StreamHandler()
    ]
)
logging.info(f'log file: {log_save_file}, save dir: {model_save_dir}')

# 设置随机种子
train_seed = config.train.seed
max_timesteps = config.train.max_timesteps
torch.manual_seed(train_seed)
torch.cuda.manual_seed(train_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 初始化环境
need_render = config.train.need_render
env = RobotEnv(env_config, render=need_render)

# 初始化网络和策略
hidden_size = config.network.hidden_size
actor = PPOActor(env.state_dim, env.action_dim, hidden_size=hidden_size)
critic = PPOCritic(env.state_dim, hidden_size=hidden_size)
agent = PPO(actor, critic, **config.ppo.to_dict())

# actor = SACActor(env.state_dim, env.action_dim, hidden_size=hidden_size)
# critic = SACCritic(env.state_dim, env.action_dim, hidden_size=hidden_size)
# agent = SAC(actor, critic, **config.sac.to_dict())

# 训练循环
total_steps = 0
cnt = 0
obs = env.reset()
max_len = agent.data_size()
while total_steps < max_timesteps:
    rews = 0
    step = 0
    for _ in range(max_len):
        action_mask = env.get_action_mask()
        act, log_prob = agent.act(obs, action_mask)
        next_obs, rew, done, info = env.step(act)
        total_steps += 1
        agent.store({
            "obs": obs,
            "act": act,
            "rew": rew,
            "next_obs": next_obs,
            "done": done,
            "log_prob": log_prob,
            "act_mask": action_mask
        })
        obs = next_obs
        rews += rew
        step += 1

        if total_steps % config.train.save_interval == 0:
            save_path = f'{model_save_dir}/save_model_{total_steps}.pth'
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
env.close()