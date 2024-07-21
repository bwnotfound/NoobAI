import sys

project_dir = "noobai".join(__file__.split("noobai")[:-1])

sys.path.append(project_dir)
import time
from threading import Event, Semaphore
from dataclasses import asdict

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
config.env_num_workers = 32
config.num_of_new_step_per_train = 32
config.train_per_step = 2
config.warmup_steps = 50000

sample_data_mode = [
    DataMode.state,
    DataMode.action,
    DataMode.next_state,
    DataMode.reward,
    DataMode.done,
]

env = gym.make(config.env_name)
env_wrapper = gecw(env, config.env_num_workers, sample_data_mode=sample_data_mode)
# env_wrapper = gesw(env, sample_data_mode=sample_data_mode)

config.num_states = state_dim = env.observation_space.shape[0]
config.num_actions = action_dim = env.action_space.shape[0]
config.max_action = max_action = float(env.action_space.high[0])
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
replay_buffer = FIFOOfflineReplayBuffer(config.capacity)


def get_action(state):
    action = agent.actor(state)
    action = (
        action + torch.randn_like(agent.actor(state)) * config.expl_noise * max_action
    )
    action = torch.clamp(action, -max_action, max_action)
    return action


fill_replay_buffer_with_random_data(
    env, replay_buffer, config.warmup_steps, sample_data_mode=sample_data_mode
)

import wandb

wandb.init(project="rl", config=asdict(config))

t_bar = tqdm(total=config.train_eps, ncols=80, colour="green")


def eval():
    total_reward = 0
    for _ in tqdm(range(config.eval_eps), ncols=80, colour="blue", leave=False):
        state, _ = env.reset()
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            total_reward += reward
    total_reward /= config.eval_eps
    return total_reward


num_of_new_step_per_train = Semaphore(config.num_of_new_step_per_train)
env_wrapper.start(
    get_action,
    replay_buffer,
    device=config.device,
    num_of_new_step_per_train=num_of_new_step_per_train,
)
for train_step in range(config.train_eps):
    if train_step % config.train_per_step == 0:
        while num_of_new_step_per_train._value > 0:
            time.sleep(0.001)
        num_of_new_step_per_train.release(config.num_of_new_step_per_train)
    stats = agent.train(replay_buffer, batch_size=config.batch_size)
    if train_step % config.eval_freq == 0:
        eval_reward = eval()
        stats["eval_reward"] = eval_reward  
        tqdm.write(f"Train step: {train_step} Eval_reward: {eval_reward}")
    wandb.log(stats)
    t_bar.update()
env_wrapper.stop()


# for train_step in range(config.train_eps):
#     done, ep_reward = env_wrapper.step(get_action, replay_buffer, device=config.device)
#     stats = agent.train(replay_buffer, batch_size=config.batch_size)
#     if done:
#         t_bar.set_postfix({"ep_reward": ep_reward})
#         stats["train_episode_reward"] = ep_reward
#     t_bar.update()
#     if train_step % config.eval_freq == 0:
#         eval_reward = eval()
#         stats["eval_reward"] = eval_reward
#         tqdm.write(f"Train step: {train_step} Eval_reward: {eval_reward}")
#     wandb.log(stats)
