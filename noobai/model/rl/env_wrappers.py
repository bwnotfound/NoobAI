import copy
import time
from threading import Thread, Event, Semaphore

# from queue import Queue

import numpy as np
import torch
from tqdm import tqdm

from noobai.model.rl.replay_buffer import BaseReplayBuffer


class DataMode:
    state = 1
    action = 2
    reward = 3
    next_state = 4
    # next_action = 5
    # next_reward = 6
    done = 7
    cum_reward = 8  # accmulate reward calc by passed func.
    # info = 9


default_sample_data_mode = [
    DataMode.state,
    DataMode.action,
    DataMode.next_state,
    DataMode.reward,
    DataMode.done,
]


class GymEnvSimpleWrapper:
    def __init__(self, env, sample_data_mode=None, cum_reward_func=None):
        self.env = copy.deepcopy(env)
        if sample_data_mode is None:
            sample_data_mode = default_sample_data_mode
        self.sample_data_mode = sample_data_mode
        if DataMode.cum_reward in sample_data_mode:
            assert (
                cum_reward_func is not None
            ), 'cum_reward_func is not None when the mode of "value" is in sample_data_mode.'
        self.cum_reward_func = cum_reward_func
        self.need_cum_reward = DataMode.cum_reward in self.sample_data_mode

        self.env_is_done = True
        self.state = None
        if self.need_cum_reward:
            (
                self.states,
                self.rewards,
                self.dones,
                self.actions,
                self.next_states,
            ) = ([], [], [], [], [])
        self.ep_reward = 0

    def step(self, model, replay_buffer: BaseReplayBuffer, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.env_is_done:
            if self.need_cum_reward and len(self.dones) > 0:
                cum_rewards = self.cum_reward_func(self.rewards)
                assert len(cum_rewards) == len(
                    self.rewards
                ), f"cum_reward_func returned list length is {len(cum_rewards)} and rewards length is {len(self.rewards)}"
                for j in range(len(self.rewards)):
                    data = []
                    for data_mode in self.sample_data_mode:
                        if data_mode == DataMode.state:
                            data.append(self.states[j])
                        elif data_mode == DataMode.action:
                            data.append(self.actions[j])
                        elif data_mode == DataMode.reward:
                            data.append(self.rewards[j])
                        elif data_mode == DataMode.next_state:
                            data.append(self.next_states[j])
                        elif data_mode == DataMode.done:
                            data.append(self.dones[j])
                        elif data_mode == DataMode.cum_reward:
                            data.append(cum_rewards[j])
                        else:
                            raise RuntimeError(
                                f"Cannot specified data_mode: {data_mode}"
                            )
                    replay_buffer.add(data)
            self.state, _ = self.env.reset()
            self.env_is_done = False
            self.ep_reward = 0

        model_input = torch.FloatTensor(np.array(self.state)).to(device).unsqueeze(0)
        model_output = model(model_input)
        if isinstance(model_output, (list, tuple)):
            model_output = model_output[0]
        action_numpy = model_output.squeeze(0).detach().cpu().numpy()
        next_state, reward, terminated, truncated, info = self.env.step(action_numpy)
        done = terminated or truncated
        if self.need_cum_reward:
            self.states.append(self.state)
            self.rewards.append(reward)
            self.dones.append(done)
            self.actions.append(action_numpy)
            self.next_states.append(next_state)
        else:
            data = []
            for data_mode in self.sample_data_mode:
                if data_mode == DataMode.state:
                    data.append(self.state)
                elif data_mode == DataMode.action:
                    data.append(action_numpy)
                elif data_mode == DataMode.reward:
                    data.append(reward)
                elif data_mode == DataMode.next_state:
                    data.append(next_state)
                elif data_mode == DataMode.done:
                    data.append(done)
                else:
                    raise RuntimeError(f"Cannot specified data_mode: {data_mode}")
            replay_buffer.add(data)
        self.env_is_done = done
        self.ep_reward += reward
        self.state = next_state
        return done, self.ep_reward


class GymEnvBatchWrapper:
    def __init__(self, env, num_envs=1):
        if isinstance(env, list):
            self.envs = copy.deepcopy(env)
        else:
            self.envs = [copy.deepcopy(env) for _ in range(num_envs)]

    def reset(self, return_running_envs_index=True):
        states, infos = [], []
        self.running_envs_index = [i for i in range(len(self.envs))]
        for env in self.envs:
            state, info = env.reset()
            states.append(state)
            infos.append(info)
        if return_running_envs_index:
            return np.array(states), infos, copy.deepcopy(self.running_envs_index)
        else:
            return np.array(states), infos

    def step(self, actions, return_running_envs_index=True):
        states, rewards, dones, infos = [], [], [], []
        pop_ids = []
        result = None
        if return_running_envs_index:
            for i, index in enumerate(self.running_envs_index):
                env = self.envs[index]
                state, reward, terminated, truncated, info = env.step(actions[i])
                done = terminated or truncated
                states.append(state)
                rewards.append(reward)
                dones.append(done)
                infos.append(info)
                if done:
                    pop_ids.append(i)

            result = (
                np.array(states),
                np.array(rewards),
                np.array(dones),
                infos,
                copy.deepcopy(self.running_envs_index),
                pop_ids,
            )
        else:
            for index in range(len(self.envs)):
                if index not in self.running_envs_index:
                    states.append(None)
                    rewards.append(None)
                    dones.append(None)
                    infos.append(None)
                    continue
                if done:
                    pop_ids.append(i)

            result = (states, rewards, dones, infos)
        for i in pop_ids[::-1]:
            self.running_envs_index.pop(i)
        return result

    @property
    def base_env(self):
        return self.envs[0]


class GymEnvContinuousWrapper:

    def __init__(
        self,
        env,
        num_workers=1,
        use_multithread=False,
        sample_data_mode=None,
        cum_reward_func=None,
    ):
        """
        param:
            env: Should be able to be deepcopy to duplicate.
            sample_data_mode: Determine the construction of each step data. Default: see code.
            cum_reward_func: Receive a list of reward and return the same size list of value.
            WARNNING: cum_reward_func only work when sample_data_mode contain the value.
        """
        self.base_env = copy.deepcopy(env)
        self.num_workers = num_workers
        self.use_multithread = use_multithread
        if sample_data_mode is None:
            sample_data_mode = default_sample_data_mode
        self.sample_data_mode = sample_data_mode
        if DataMode.cum_reward in sample_data_mode:
            assert (
                cum_reward_func is not None
            ), 'cum_reward_func is not None when the mode of "value" is in sample_data_mode.'
        self.cum_reward_func = cum_reward_func
        self.need_cum_reward = DataMode.cum_reward in self.sample_data_mode

        self.stop_event = Event()

    def start(
        self,
        model,
        replay_buffer: BaseReplayBuffer,
        device=None,
        num_of_new_step_per_train: Semaphore = None,
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.stop_event.clear()

        def run():
            states, rewards, dones, actions, next_states = (
                [[] for _ in range(self.num_workers)],
                [[] for _ in range(self.num_workers)],
                [[] for _ in range(self.num_workers)],
                [[] for _ in range(self.num_workers)],
                [[] for _ in range(self.num_workers)],
            )
            last_states = [None for _ in range(self.num_workers)]
            env_is_dones = [True for _ in range(self.num_workers)]
            env_list = [copy.deepcopy(self.base_env) for _ in range(self.num_workers)]

            while not self.stop_event.is_set():
                for i in range(self.num_workers):
                    if not env_is_dones[i]:
                        continue
                    if self.need_cum_reward and len(dones[i]) > 0:
                        cum_rewards = self.cum_reward_func(rewards[i])
                        assert len(cum_rewards) == len(
                            rewards
                        ), f"cum_reward_func returned list length is {len(cum_rewards)} and rewards length is {len(rewards)}"
                        for j in range(len(rewards[i])):
                            data = []
                            for data_mode in self.sample_data_mode:
                                if data_mode == DataMode.state:
                                    data.append(states[i][j])
                                elif data_mode == DataMode.action:
                                    data.append(actions[i][j])
                                elif data_mode == DataMode.reward:
                                    data.append(rewards[i][j])
                                elif data_mode == DataMode.next_state:
                                    data.append(next_states[i][j])
                                elif data_mode == DataMode.done:
                                    data.append(dones[i][j])
                                elif data_mode == DataMode.cum_reward:
                                    data.append(cum_rewards[i][j])
                                else:
                                    raise RuntimeError(
                                        f"Cannot specified data_mode: {data_mode}"
                                    )
                            if num_of_new_step_per_train is not None:
                                num_of_new_step_per_train.acquire()
                            replay_buffer.add(data)
                    state, _ = env_list[i].reset()
                    last_states[i] = state
                    env_is_dones[i] = False
                last_states_numpy = np.array(last_states)
                model_input = torch.FloatTensor(last_states_numpy).to(device)
                model_output = model(model_input)
                if isinstance(model_output, (list, tuple)):
                    model_output = model_output[0]
                action_list_numpy = model_output.detach().cpu().numpy()
                for i in range(self.num_workers):
                    state, reward, terminated, truncated, info = env_list[i].step(
                        action_list_numpy[i]
                    )
                    done = terminated or truncated
                    if self.need_cum_reward:
                        actions[i].append(action_list_numpy[i])
                        states[i].append(last_states[i])
                        rewards[i].append(reward)
                        dones[i].append(done)
                        next_states[i].append(np.array(state))
                    else:
                        data = []
                        for data_mode in self.sample_data_mode:
                            if data_mode == DataMode.state:
                                data.append(last_states[i])
                            elif data_mode == DataMode.action:
                                data.append(action_list_numpy[i])
                            elif data_mode == DataMode.reward:
                                data.append(reward)
                            elif data_mode == DataMode.next_state:
                                data.append(state)
                            elif data_mode == DataMode.done:
                                data.append(done)
                            else:
                                raise RuntimeError(
                                    f"Cannot specified data_mode: {data_mode}"
                                )
                        if num_of_new_step_per_train is not None:
                            num_of_new_step_per_train.acquire()
                        replay_buffer.add(data)
                    env_is_dones[i] = done
                    last_states[i] = state

        Thread(target=run).start()

    def stop(self):
        self.stop_event.set()


def fill_replay_buffer_with_random_data(
    env,
    replay_buffer: BaseReplayBuffer,
    num_steps,
    action_func=None,
    sample_data_mode=None,
    show_progress=True,
):
    env = copy.deepcopy(env)
    if sample_data_mode is None:
        sample_data_mode = default_sample_data_mode
    if action_func is None:
        action_func = env.action_space.sample
    if show_progress:
        t_bar = tqdm(total=num_steps, colour="green")
    state, done, total_reward = None, True, 0
    for _ in range(num_steps):
        if done:
            state, _ = env.reset()
            done = False
            t_bar.set_postfix({"Total_reward": total_reward})
            total_reward = 0
        action = action_func()
        action = np.array(action)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        data = []
        for data_mode in sample_data_mode:
            if data_mode == DataMode.state:
                data.append(state)
            elif data_mode == DataMode.action:
                data.append(action)
            elif data_mode == DataMode.reward:
                data.append(reward)
            elif data_mode == DataMode.next_state:
                data.append(next_state)
            elif data_mode == DataMode.done:
                data.append(done)
            else:
                raise RuntimeError(f"Cannot specified data_mode: {data_mode}")
        replay_buffer.add(data)
        state = next_state
        t_bar.update()
    t_bar.close()
