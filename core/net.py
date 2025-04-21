import os
import time
import numpy as np
import matplotlib.pyplot as plt
from itertools import count
from collections import namedtuple
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

    
class PPOActor(nn.Module):
    def __init__(self, obs_dim, action_dims, hidden_size=128):
        super().__init__()
        self.action_dims = action_dims
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU()
        )
        self.heads = nn.Linear(hidden_size, self.action_dims.sum())

    def forward(self, x):
        x = self.shared(x)
        logits = self.heads(x)
        logits = logits.split(self.action_dims.tolist(), dim=-1)
        return logits
    
    def get_dists(self, x, action_mask=None):
        # obs shape: [batch_size, obs_dim] -> [bs, 28]
        # logits shape: [action_dim, batch_size, action_num] -> [2, bs, 3]
        # action_mask shape: [batch_size, action_dim, action_num] -> [bs, 2, 3]
        logits = self.forward(x)
        if action_mask is not None:
            neg_inf = torch.tensor(-float('inf'), device=logits[0].device)
            logits = [torch.where(mask[:, :l.shape[-1]], l, neg_inf) 
                        for l, mask in zip(logits, action_mask.transpose(0, 1))]
        dists = [Categorical(logits=logit) for logit in logits]
        return dists

    def act(self, x, action_mask=None, eval=False):
        dists = self.get_dists(x, action_mask)      
        if eval:
            actions = [dist.probs.argmax(-1) for dist in dists]
        else:
            actions = [dist.sample() for dist in dists]  
        log_probs = [dist.log_prob(act) for dist, act in zip(dists, actions)]
        return torch.stack(actions, dim=-1), torch.stack(log_probs, dim=-1)


class PPOCritic(nn.Module):
    def __init__(self, obs_dim, hidden_size=128):
        super().__init__()
        self.v = nn.Sequential(
            nn.Linear(obs_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return self.v(x).squeeze(-1)
    

# class MultiDiscreteActorCritic(nn.Module):
#     def __init__(self, obs_dim, action_dims, hidden_size=128, shared=True):
#         super().__init__()
#         self.action_dims = action_dims
#         self.shared = nn.Sequential(
#             nn.Linear(obs_dim, hidden_size), nn.ReLU(),
#             nn.Linear(hidden_size, hidden_size), nn.ReLU()
#         )
#         self.pi = nn.Linear(hidden_size, self.action_dims.sum())
#         self.v = nn.Linear(hidden_size, 1)
    
#     def get_pi(self, x, action_mask=None):
#         logits = self.pi(self.shared(x))
#         logits = logits.split(self.action_dims.tolist(), dim=-1)
#         if action_mask is not None:
#             logits = [torch.where(mask[:, :l.shape[-1]], l, torch.tensor(-float('inf')).to(l.device)) 
#                         for l, mask in zip(logits, action_mask.transpose(0, 1))]
#         dists = [Categorical(logits=logit) for logit in logits]
#         return dists
    
#     def get_value(self, x):
#         return self.v(self.shared(x))
    
#     def act(self, x, action_mask=None, deterministic=False):
#         dists = self.get_dists(x, action_mask)        
#         actions = [dist.probs.argmax(-1) if deterministic else dist.sample() for dist in dists]
#         log_probs = [dist.log_prob(act) for dist, act in zip(dists, actions)]
#         return torch.stack(actions, dim=-1), torch.stack(log_probs, dim=-1)


class SACActor(nn.Module):
    def __init__(self, obs_dim, action_dims, hidden_size=128):
        super().__init__()
        self.action_dims = action_dims
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU()
        )
        self.heads = nn.Linear(hidden_size, self.action_dims.sum())
    
    def forward(self, x):
        x = self.shared(x)
        logits = self.heads(x)
        logits = logits.split(self.action_dims.tolist(), dim=-1)
        return logits
    
    def get_dists(self, x, action_mask=None):
        logits = self.forward(x)
        if action_mask is not None:
            neg_inf = torch.tensor(-float('inf'), device=logits[0].device)
            logits = [torch.where(mask[:, :l.shape[-1]], l, neg_inf) 
                    for l, mask in zip(logits, action_mask.transpose(0, 1))]
        dists = [Categorical(logits=logit) for logit in logits]
        return dists
    
    def act(self, x, action_mask=None, eval=False):
        dists = self.get_dists(x, action_mask)
        if eval:
            actions = [dist.probs.argmax(-1) for dist in dists]
        else:
            actions = [dist.sample() for dist in dists]
        log_probs = [dist.log_prob(act) for dist, act in zip(dists, actions)]
        return torch.stack(actions, dim=-1), torch.stack(log_probs, dim=-1)


class SACCritic(nn.Module):
    def __init__(self, obs_dim, action_dims, hidden_size=128):
        super().__init__()
        self.action_dims = action_dims
        self.input_dim = obs_dim + action_dims.sum()  # 拼接动作

        self.q1 = nn.Sequential(
            nn.Linear(self.input_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 1)  # 标量 Q 值
        )
        self.q2 = deepcopy(self.q1)

    def forward(self, obs, actions):
        # one-hot encode actions for each dimension
        one_hots = []
        for i, dim in enumerate(self.action_dims):
            one_hot = torch.nn.functional.one_hot(actions[:, i], num_classes=dim)
            one_hots.append(one_hot.float())
        action_embed = torch.cat(one_hots, dim=-1)
        x = torch.cat([obs, action_embed], dim=-1)
        return self.q1(x), self.q2(x)