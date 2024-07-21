import sys

project_dir = "noobai".join(__file__.split("noobai")[:-1])

sys.path.append(project_dir)
import time
from threading import Event

import torch
import gymnasium as gym
from tqdm import tqdm
import numpy as np

from noobai.model.rl.TD3 import TD3
from noobai.real_apply.rl.config import TD3Config
from noobai.model.rl.parallel_env import (
    DataMode,
    GymEnvContinuousWrapper as gecw,
    fill_replay_buffer_with_random_data,
    GymEnvSimpleWrapper as gesw,
)
from noobai.model.rl.replay_buffer import FIFOOfflineReplayBuffer

config = TD3Config()
config.device = "cuda"
config.env_name = "HalfCheetah-v4"
config.batch_size = 256
config.policy_freq = 2
config.env_num_workers = 1

sample_data_mode = [
    DataMode.state,
    DataMode.action,
    DataMode.next_state,
    DataMode.reward,
    DataMode.done,
]

env = gym.make(config.env_name)
# env_wrapper = gecw(env, config.env_num_workers, sample_data_mode=sample_data_mode)
env_wrapper = gesw(env, sample_data_mode=sample_data_mode)

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


def get_action(state):
    action = agent.actor(state)
    action = (
        action + torch.randn_like(agent.actor(state)) * config.expl_noise * max_action
    )
    action = torch.clamp(action, -max_action, max_action)
    return action


fill_replay_buffer_with_random_data(
    env, replay_buffer, 25000, sample_data_mode=sample_data_mode
)

import wandb

wandb.init(project="rl", config=config)

t_bar = tqdm(total=config.train_eps, ncols=80, colour="green")

eval_step = 5000
eval_time = 4
def eval():
    state, _ = env.reset()
    done = False
    total_reward = 0
    for _ in tqdm(range(eval_time), ncols=80, colour="blue", leave=False):
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            total_reward += reward
    total_reward /= eval_time
    return total_reward

# initial_flag = False
# stop_event = Event()
# env_wrapper.start(
#     agent.actor, replay_buffer, device=config.device, interrupt_event=stop_event
# )
# for train_step in range(config.train_eps):
#     while (initial_flag and replay_buffer.new_in_after_sample < config.batch_size) or (
#         not initial_flag and replay_buffer.new_in_after_sample < 20
#     ):
#         time.sleep(0.001)
#     stop_event.set()
#     stats = agent.train(replay_buffer, batch_size=config.batch_size)
#     stop_event.clear()
#     wandb.log(stats)

#     t_bar.update()
# env_wrapper.stop()


for train_step in range(config.train_eps):
    done, ep_reward = env_wrapper.step(get_action, replay_buffer, device=config.device)
    
    if done:
        t_bar.set_postfix({"ep_reward": ep_reward})
        stats["train_episode_reward"] = ep_reward
    stats = agent.train(replay_buffer, batch_size=config.batch_size)
    t_bar.update()
    if train_step % eval_step == 0:
        eval_reward = eval()
        stats["eval_reward"] = eval_reward
        tqdm.write(f"Train step: {train_step} Eval_reward: {eval_reward}")
    wandb.log(stats)
