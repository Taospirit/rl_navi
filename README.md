# RL-Navi

A 2D navigation simulator based on reinforcement learning, using PPO algorithm to train robots for autonomous navigation in complex environments.

![Demo](image.png)

## File Structure

- `env.py`: Navigation environment simulator
- `train.py`: PPO training script
- `net.py`: Neural network model
- `policy.py`: PPO policy implementation
- `eval.py`: Model evaluation script
- `configs/`: Environment configuration files
- `saves/`: Model save directory

## Requirements

```
pygame
numpy
torch
shapely
opencv-python
```

## Usage

1. Train the model:
```bash
python train.py
```

2. Evaluate the model:
```bash
python eval.py
```

3. Customize environment:
Modify `configs/env_config_*.json` to configure environment parameters
