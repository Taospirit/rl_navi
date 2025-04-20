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

    
class MultiDiscreteActor(nn.Module):
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
    

class MultiDiscreteActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dims, hidden_size=128, shared=True):
        super().__init__()
        self.action_dims = action_dims
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU()
        )
        self.pi = nn.Linear(hidden_size, self.action_dims.sum())
        self.v = nn.Linear(hidden_size, 1)
    
    def get_pi(self, x, action_mask=None):
        logits = self.pi(self.shared(x))
        logits = logits.split(self.action_dims.tolist(), dim=-1)
        if action_mask is not None:
            logits = [torch.where(mask[:, :l.shape[-1]], l, torch.tensor(-float('inf')).to(l.device)) 
                        for l, mask in zip(logits, action_mask.transpose(0, 1))]
        dists = [Categorical(logits=logit) for logit in logits]
        return dists
    
    def get_value(self, x):
        return self.v(self.shared(x))
    
    def act(self, x, action_mask=None, deterministic=False):
        dists = self.get_dists(x, action_mask)        
        actions = [dist.probs.argmax(-1) if deterministic else dist.sample() for dist in dists]
        log_probs = [dist.log_prob(act) for dist, act in zip(dists, actions)]
        return torch.stack(actions, dim=-1), torch.stack(log_probs, dim=-1)