import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data import DataLoader, TensorDataset
from collections import deque


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
        self.entropy_coef = 1.0
        self.value_coef = value_coef

        self.actor_optim = optim.Adam(actor.parameters(), lr=pi_lr)
        self.critic_optim = optim.Adam(critic.parameters(), lr=v_lr)

    def data_size(self):
        return self.buffer.memory.maxlen

    def store(self, dict):
        self.buffer.append(**dict)

    def act(self, obs, deterministic=False):
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            actions, log_probs = self.actor.act(obs, deterministic)
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

        with torch.no_grad():
            vals = self.critic(obs).numpy()
            next_val = self.critic(next_obs).item()
            advs = compute_gae(rews, vals, next_val, 1 - dones, self.gamma, self.lam)
            advs = torch.tensor(advs, dtype=torch.float32)
            vals = torch.tensor(vals, dtype=torch.float32)
            returns = advs + vals

        dataset = TensorDataset(obs, acts, old_log_probs, advs, returns)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for _ in range(self.epochs):
            for batch in loader:
                b_obs, b_acts, b_old_log_probs, b_advs, b_returns = batch
                logits = self.actor(b_obs)
                dists = [Categorical(logits=l) for l in logits]

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
