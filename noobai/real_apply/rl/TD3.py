import sys

project_dir = "noobai".join(__file__.split("noobai")[:-1])

sys.path.append(project_dir)
import time

import gymnasium as gym
from tqdm import tqdm
import wandb

from noobai.model.rl.TD3 import TD3
from noobai.real_apply.rl.config import TD3Config
from noobai.model.rl.parallel_env import GymEnvContinuousWrapper as gecw
from noobai.model.rl.replay_buffer import FIFOOfflineReplayBuffer

config = TD3Config()
config.device = "cpu"
config.env_name = "HalfCheetah-v4"
config.batch_size = 256

wandb.init(project="rl", config=config)

env = gym.make(config.env_name)
# state, action, next_state, reward, done_bool
env_wrapper = gecw(
    env,
    config.env_num_workers,
    sample_data_mode=[gecw.state, gecw.action, gecw.next_state, gecw.reward, gecw.done],
)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
agent = TD3(
    state_dim,
    action_dim,
    max_action,
    policy_noise=config.policy_noise * max_action,
    noise_clip=config.noise_clip * max_action,
    policy_freq=config.policy_freq,
    tau=config.tau,
    discount=config.discount,
    device=config.device,
)
replay_buffer = FIFOOfflineReplayBuffer(int(1e6))
env_wrapper.start(agent.actor, replay_buffer, device=config.device)

t_bar = tqdm(total=config.train_eps, ncols=80, colour="green")
for train_step in range(config.train_eps):
    while replay_buffer.size() < config.batch_size:
        time.sleep(0.001)
    stats = agent.train(replay_buffer, batch_size=config.batch_size)
    wandb.log(stats)

    t_bar.update()
env_wrapper.stop()
