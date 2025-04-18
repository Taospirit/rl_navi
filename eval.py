from env import RobotEnv
from net import MultiDiscreteActor, Critic
from policy import PPO

env_config = 'configs/env_config_1.json'
need_render = 1
env = RobotEnv(env_config, render=need_render, save_video="rl_navi.mp4")

# model_path = 'saves-0418-191948/save_model_20000.pth'
model_path = 'save_model_210000.pth'
actor = MultiDiscreteActor(env.state_dim, env.action_dim)
critic = Critic(env.state_dim)
agent = PPO(actor, critic)
agent.load_model(model_path)

loop_cnt = 10
for ep in range(loop_cnt):
    obs = env.reset()
    done = False
    total_rews = 0
    step = 0
    while not done:
        # act, _ = agent.act(obs, deterministic=True)
        act, _ = agent.act(obs)
        next_obs, rew, done, info = env.step(act)
        obs = next_obs
        total_rews += rew
        step += 1
        print(f"ep {ep}, step {step}, obs {obs.shape}, act {act}, rew {rew}")
        if need_render:
            env.render()
        if step > 3000:
            step = 0
            env.reset()
    print(f"ep {ep}, rew {total_rews}, step {step}")
env.close()