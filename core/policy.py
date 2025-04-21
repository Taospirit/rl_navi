import os
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data import DataLoader, TensorDataset
from collections import deque
import torch.nn.functional as F
from copy import deepcopy


class ReplayBuffer:
    def __init__(self, size):
        self.memory = deque(maxlen=int(size))

    def append(self, **kwargs):
        self.memory.append(kwargs)

    def clear(self):
        self.memory.clear()

    def split(self):
        keys = self.memory[0].keys()
        split_res = {k: [item[k] for item in self.memory] for k in keys}
        for k in split_res:
            if not isinstance(split_res[k][0], torch.Tensor):
                split_res[k] = np.asarray(split_res[k])
        return split_res

    def is_full(self):
        return len(self.memory) == self.memory.maxlen

    def is_empty(self):
        return len(self.memory) == 0

def compute_gae(rewards, values, next_value, masks, gamma=0.99, lam=0.95):
    values = np.append(values, next_value)
    gae = 0
    adv = np.zeros_like(rewards)
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] * masks[t] - values[t]
        gae = delta + gamma * lam * masks[t] * gae
        adv[t] = gae
    return adv

class PPO:
    def __init__(self, actor, critic, buffer_size=2048, gamma=0.99, lam=0.95,
                 pi_lr=3e-4, v_lr=1e-3, clip_ratio=0.2, epochs=10, batch_size=64,
                 entropy_coef=0.01, value_coef=0.5):
        self.actor = actor
        self.critic = critic

        self.buffer = ReplayBuffer(buffer_size)
        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.epochs = epochs
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

        self.actor_optim = optim.Adam(actor.parameters(), lr=pi_lr)
        self.critic_optim = optim.Adam(critic.parameters(), lr=v_lr)

    def save_model(self, path):
        dir_path = osp.dirname(path)
        os.makedirs(dir_path, exist_ok=True)

        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optim_state_dict': self.actor_optim.state_dict(),
            'critic_optim_state_dict': self.critic_optim.state_dict(),
        }, path)
        print(f"模型已保存到 {path}")

    def load_model(self, path):
        assert osp.exists(path), f"path not exits: {path}"
        
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optim.load_state_dict(checkpoint['actor_optim_state_dict'])
        self.critic_optim.load_state_dict(checkpoint['critic_optim_state_dict'])
        print(f"已从 {path} 加载模型参数")

    def data_size(self):
        return self.buffer.memory.maxlen

    def store(self, dict):
        self.buffer.append(**dict)

    def act(self, obs, action_mask=None, eval=False):
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        if action_mask is not None:
            action_mask = torch.tensor(action_mask, dtype=torch.bool).unsqueeze(0)
        with torch.no_grad():
            actions, log_probs = self.actor.act(obs, action_mask, eval)
        return actions.squeeze(0).numpy(), log_probs.squeeze(0).numpy()

    def update(self):
        mem = self.buffer.split()
        self.buffer.clear()

        obs = torch.tensor(mem['obs'], dtype=torch.float32)
        acts = torch.tensor(mem['act'], dtype=torch.long)
        rews = np.asarray(mem['rew'])
        next_obs = torch.tensor(mem['next_obs'][-1], dtype=torch.float32).unsqueeze(0)
        dones = np.asarray(mem['done'])
        old_log_probs = torch.tensor(mem['log_prob'], dtype=torch.float32)
        act_mask = torch.tensor(mem['act_mask'], dtype=torch.bool)

        with torch.no_grad():
            vals = self.critic(obs).numpy()
            next_val = self.critic(next_obs).item()
            advs = compute_gae(rews, vals, next_val, 1 - dones, self.gamma, self.lam)
            advs = torch.tensor(advs, dtype=torch.float32)
            vals = torch.tensor(vals, dtype=torch.float32)
            returns = advs + vals

        dataset = TensorDataset(obs, acts, old_log_probs, advs, returns, act_mask)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for _ in range(self.epochs):
            for batch in loader:
                b_obs, b_acts, b_old_log_probs, b_advs, b_returns, b_masks = batch
                
                dists = self.actor.get_dists(b_obs, b_masks)
                new_log_probs = torch.stack(
                    [dist.log_prob(b_acts[:, i]) for i, dist in enumerate(dists)], dim=-1
                )
                entropy = torch.stack([dist.entropy() for dist in dists], dim=-1).mean()
                ratio = torch.exp(new_log_probs - b_old_log_probs)
                b_advs = b_advs.unsqueeze(-1).expand_as(ratio)
                surr1 = ratio * b_advs
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * b_advs
                policy_loss = -torch.min(surr1, surr2).sum(-1).mean()

                value_loss = (self.critic(b_obs) - b_returns).pow(2).mean()

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.actor_optim.zero_grad()
                self.critic_optim.zero_grad()
                loss.backward()
                self.actor_optim.step()
                self.critic_optim.step()
    
    def ready(self):
        return self.buffer.is_full()


class SAC:
    def __init__(self, actor, critic,
                 buffer_size=1000000, gamma=0.99, tau=0.005,
                 actor_lr=3e-4, critic_lr=3e-4, alpha_lr=3e-4,
                 batch_size=256, target_entropy=None):
        self.actor = actor
        self.critic = critic
        # self.critic2 = deepcopy(critic)  # 创建第二个 critic 网络
        self.target_critic = deepcopy(critic)  # 创建目标网络
        # self.target_critic2 = deepcopy(critic)  # 创建目标网络
        
        self.buffer = ReplayBuffer(buffer_size)
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        
        # 自动调整的温度参数
        self.log_alpha = nn.Parameter(torch.zeros(1))
        self.alpha = self.log_alpha.exp()
        self.target_entropy = target_entropy
        
        # 优化器
        self.actor_optim = optim.Adam(actor.parameters(), lr=actor_lr)
        self.critic_optim = optim.Adam(critic.parameters(), lr=critic_lr)
        # self.critic2_optim = optim.Adam(self.critic2.parameters(), lr=critic_lr)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=alpha_lr)
        
        # 初始化目标网络
        self._update_target_networks(tau=1.0)
    
    def _update_target_networks(self, tau=None):
        tau = self.tau if tau is None else tau
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        # for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
        #     target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    def save_model(self, path):
        dir_path = osp.dirname(path)
        os.makedirs(dir_path, exist_ok=True)
        
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'target_critic_state_dict': self.target_critic.state_dict(),
            # 'target_critic2_state_dict': self.target_critic2.state_dict(),
            'actor_optim_state_dict': self.actor_optim.state_dict(),
            'critic_optim_state_dict': self.critic_optim.state_dict(),
            # 'critic2_optim_state_dict': self.critic2_optim.state_dict(),
            'alpha_optim_state_dict': self.alpha_optim.state_dict(),
            'log_alpha': self.log_alpha,
        }, path)
        print(f"模型已保存到 {path}")
    
    def load_model(self, path):
        assert osp.exists(path), f"path not exits: {path}"
        
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
        # self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        # self.target_critic2.load_state_dict(checkpoint['target_critic2_state_dict'])
        self.actor_optim.load_state_dict(checkpoint['actor_optim_state_dict'])
        self.critic_optim.load_state_dict(checkpoint['critic_optim_state_dict'])
        # self.critic2_optim.load_state_dict(checkpoint['critic2_optim_state_dict'])
        self.alpha_optim.load_state_dict(checkpoint['alpha_optim_state_dict'])
        self.log_alpha.data.copy_(checkpoint['log_alpha'])
        self.alpha = self.log_alpha.exp()
        print(f"已从 {path} 加载模型参数")
    
    def data_size(self):
        return self.buffer.memory.maxlen
    
    def store(self, dict):
        self.buffer.append(**dict)
    
    def act(self, obs, action_mask=None, deterministic=False):
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        if action_mask is not None:
            action_mask = torch.tensor(action_mask, dtype=torch.bool).unsqueeze(0)
        with torch.no_grad():
            actions, log_probs = self.actor.act(obs, action_mask, deterministic)

        return actions.squeeze(0).numpy(), log_probs.squeeze(0).numpy()
    
    def update(self):
        if len(self.buffer.memory) < self.batch_size:
            return
        if self.target_entropy is None:
            self.target_entropy = -sum([np.log(n) for n in self.actor.action_dims])
        
        mem = self.buffer.split()
        indices = np.random.choice(len(mem['obs']), self.batch_size, replace=False)
        
        obs = torch.tensor(mem['obs'][indices], dtype=torch.float32)
        acts = torch.tensor(mem['act'][indices], dtype=torch.long)
        rews = torch.tensor(mem['rew'][indices], dtype=torch.float32)
        next_obs = torch.tensor(mem['next_obs'][indices], dtype=torch.float32)
        dones = torch.tensor(mem['done'][indices], dtype=torch.float32)
        act_mask = torch.tensor(mem['act_mask'][indices], dtype=torch.bool)
        
        # 更新 Q 网络
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.act(next_obs, act_mask)
            # next_q1 = self.target_critic(next_obs, next_actions)
            # next_q2 = self.target_critic(next_obs, next_actions)
            next_q1, next_q2 = self.target_critic(next_obs, next_actions)
            next_q = torch.min(next_q1, next_q2) - self.alpha * next_log_probs.sum(dim=-1, keepdim=True)
            target_q = rews + (1 - dones) * self.gamma * next_q.squeeze(-1)
            target_q = target_q.unsqueeze(-1)
        
        # 计算critic loss
        cur_q1, cur_q2 = self.critic(obs, acts)
        loss_q1 = F.mse_loss(cur_q1, target_q)
        loss_q2 = F.mse_loss(cur_q2, target_q)
        critic_loss = 0.5 * (loss_q1 + loss_q2)
        
        # 更新critic
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        
        # 计算actor loss
        actions, log_probs = self.actor.act(obs, act_mask)
        q1, q2 = self.critic(obs, actions)
        q_min = torch.min(q1, q2)
        
        # 重新计算log_probs用于actor loss
        actor_loss = (self.alpha.detach() * log_probs.sum(dim=-1, keepdim=True) - q_min).mean()
        
        # 更新actor
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        
        # 更新温度参数
        if self.target_entropy is not None:
            entropy = log_probs.sum(-1).detach()
            alpha_loss = -(self.log_alpha * (entropy + self.target_entropy)).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp()
        
        # 更新目标网络
        with torch.no_grad():
            self._update_target_networks()
    
    def ready(self):
        return len(self.buffer.memory) >= self.batch_size
