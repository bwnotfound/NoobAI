import sys
import math

project_dir = "noobai".join(__file__.split("noobai")[:-1])

sys.path.append(project_dir)
# import time
from dataclasses import asdict

import torch
import gymnasium as gym
from tqdm import tqdm
import numpy as np

from noobai.model.rl.Delay import Delay
from noobai.real_apply.rl.config import DelayConfig
from noobai.model.rl.replay_buffer import FIFOOfflineReplayBuffer

config = DelayConfig()
config.device = "cpu"
config.env_name = "HalfCheetah-v4"
config.batch_size = 256
config.policy_freq = 2
# config.env_num_workers = 32
config.num_of_new_step_per_train = 32
config.train_per_step = 2
config.warmup_steps = 50000
config.capacity = 1000000

config.actor_hidden_dim = 256
config.critic_hidden_dim = 256
config.actor_lr = 3e-4
config.critic_lr = 3e-4

config.vision_delay_frame = 1
config.act_delay_frame = 1

# config.eval_freq = 10

env = gym.make(config.env_name)

config.num_states = state_dim = env.observation_space.shape[0]
config.num_actions = action_dim = env.action_space.shape[0]
config.max_action = max_action = float(env.action_space.high[0])

agent = Delay(
    state_dim,
    config.actor_hidden_dim,
    config.critic_hidden_dim,
    config.actor_lr,
    config.critic_lr,
    action_dim,
    max_action,
    config.vision_delay_frame,
    config.act_delay_frame,
    policy_noise=config.policy_noise * max_action,
    noise_clip=config.noise_clip * max_action,
    policy_freq=config.policy_freq,
    tau=config.tau,
    discount=config.discount,
    device=config.device,
)
replay_buffer = FIFOOfflineReplayBuffer(config.capacity)


@torch.no_grad()
def get_action(state, return_numpy=True):
    if not isinstance(state, torch.Tensor):
        state = torch.FloatTensor(state).unsqueeze(0).to(config.device)
    action = agent.actor(state)
    action = (
        action + torch.randn_like(agent.actor(state)) * config.expl_noise * max_action
    )
    action = torch.clamp(action, -max_action, max_action)
    if return_numpy:
        action = action.squeeze(0).cpu().numpy()
    return action


done = True
first = True
t_bar = tqdm(total=config.warmup_steps, ncols=80, colour="green")
for _ in range(config.warmup_steps):
    if done:
        state, _  = env.reset()
        done = False
        if not first:
            t_bar.set_postfix_str(f"totol_reward: {total_reward}")
        total_reward = 0
        step = 0
        if not first:
            for i in range(config.act_delay_frame, len(actions) - config.vision_delay_frame):
                data = [states[i], actions[i - config.act_delay_frame], states[i + 1], rewards[i], dones[i]]
                replay_buffer.add(data)
        actions, states, rewards, dones = [], [state], [], []
        first = False
    
    if step - config.vision_delay_frame < 0:
        action = env.action_space.sample()
    else:
        action = env.action_space.sample()
        # action = get_action(states[step - config.vision_delay_frame])
    action = np.array(action)
    actions.append(action)
    if step - config.act_delay_frame < 0:
        cur_action = env.action_space.sample()
    else:
        cur_action = actions[step - config.act_delay_frame]
    next_state, reward, terminated, truncated, info = env.step(cur_action)
    done = terminated or truncated
    states.append(next_state)
    rewards.append(reward)
    dones.append(done)
    total_reward += reward
    state = next_state
    step += 1
    t_bar.update()
t_bar.close()

import wandb

wandb.init(project="rl-Delay", config=asdict(config))

def eval():
    total_reward = 0
    for _ in tqdm(range(config.eval_eps), ncols=80, colour="blue", leave=False):
        done = True
        first = True
        while True:
            if done:
                if not first:
                    break
                state, _  = env.reset()
                done = False
                step = 0
                actions, states, rewards, dones = [], [state], [], []
                first = False
            
            if step - config.vision_delay_frame < 0:
                action = env.action_space.sample()
            else:
                # action = env.action_space.sample()
                action = get_action(states[step - config.vision_delay_frame])
            action = np.array(action)
            actions.append(action)
            if step - config.act_delay_frame < 0:
                cur_action = env.action_space.sample()
            else:
                cur_action = actions[step - config.act_delay_frame]
            next_state, reward, terminated, truncated, info = env.step(cur_action)
            done = terminated or truncated
            states.append(next_state)
            rewards.append(reward)
            dones.append(done)
            total_reward += reward
            state = next_state
            step += 1
        
    total_reward /= config.eval_eps
    return total_reward


t_bar = tqdm(total=config.train_eps, ncols=80, colour="green")
train_step = 0
while train_step < config.train_eps:
    done = True
    first = True
    while True:
        if done:
            if not first:
                for i in range(config.act_delay_frame, len(actions) - config.vision_delay_frame):
                    data = [states[i], actions[i - config.act_delay_frame], states[i + 1], rewards[i], dones[i]]
                    replay_buffer.add(data)
                break
            state, _  = env.reset()
            done = False
            total_reward = 0
            step = 0
            actions, states, rewards, dones = [], [state], [], []
            first = False
        
        if step - config.vision_delay_frame < 0:
            action = env.action_space.sample()
        else:
            # action = env.action_space.sample()
            action = get_action(states[step - config.vision_delay_frame])
        action = np.array(action)
        actions.append(action)
        if step - config.act_delay_frame < 0:
            cur_action = env.action_space.sample()
        else:
            cur_action = actions[step - config.act_delay_frame]
        next_state, reward, terminated, truncated, info = env.step(cur_action)
        done = terminated or truncated
        states.append(next_state)
        rewards.append(reward)
        dones.append(done)
        total_reward += reward
        state = next_state
        step += 1
    first = True
    for _ in range(math.ceil(step / config.num_of_new_step_per_train * config.train_per_step)):
        stats = agent.train(replay_buffer, batch_size=config.batch_size)
        if first:
            first = False
            stats["ep_reward"] = total_reward
        t_bar.update()
        train_step += 1
        if train_step % config.eval_freq == 0:
            eval_reward = eval()
            stats["eval_reward"] = eval_reward
            tqdm.write(f"Train step: {train_step} Eval_reward: {eval_reward}")
        wandb.log(stats)
