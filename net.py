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

# # ppo discrete
# class ActorNet(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super().__init__()
#         self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh(),
#                                 nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
#                                 nn.Linear(hidden_dim, output_dim))
    
#     def forward(self, s):
#         return self.net(s)
    
#     def pi(self, s, softmax_dim=-1):
#         x = self.forward(s)
#         prob = F.softmax(x, dim=softmax_dim)
#         return prob
    
#     def action(self, s, softmax_dim=-1, deterministic=False):
#         prob = self.pi(s, softmax_dim)
#         import pdb; pdb.set_trace()
#         if deterministic:
#             act = torch.argmax(prob)
#             return act.item(), None
#         else:
#             dist = torch.distributions.Categorical(prob)
#             act = dist.sample()
#             log_prob = dist.log_prob(act)
#             return act.item(), log_prob.item()

# class CriticNet(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super().__init__()
#         self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(),
#                                  nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
#                                  nn.Linear(hidden_dim, output_dim))

#     def forward(self, s):
#         return self.net(s)
    
class MultiDiscreteActor(nn.Module):
    def __init__(self, obs_dim, action_dims, hidden_size=128):
        super().__init__()
        self.action_dims = action_dims
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU()
        )
        self.heads = nn.ModuleList([nn.Linear(hidden_size, dim) for dim in action_dims])

    def forward(self, x):
        x = self.shared(x)
        logits = [head(x) for head in self.heads]
        return logits
    
    def get_dists(self, x, action_mask=None):
        # obs shape: [batch_size, obs_dim] -> [bs, 28]
        # logits shape: [action_dim, batch_size, action_num] -> [2, bs, 3]
        # action_mask shape: [batch_size, action_dim, action_num] -> [bs, 2, 3]
        logits = self.forward(x)
        if action_mask is not None:
            logits = [torch.where(mask[:, :l.shape[-1]], l, torch.tensor(-float('inf')).to(l.device)) 
                        for l, mask in zip(logits, action_mask.transpose(0, 1))]
        dists = [Categorical(logits=logit) for logit in logits]
        return dists

    def act(self, x, action_mask=None, deterministic=False):
        dists = self.get_dists(x, action_mask)        
        actions = [dist.probs.argmax(-1) if deterministic else dist.sample() for dist in dists]
        log_probs = [dist.log_prob(act) for dist, act in zip(dists, actions)]
        return torch.stack(actions, dim=-1), torch.stack(log_probs, dim=-1)


class Critic(nn.Module):
    def __init__(self, obs_dim, hidden_size=128):
        super().__init__()
        self.v = nn.Sequential(
            nn.Linear(obs_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return self.v(x).squeeze(-1)