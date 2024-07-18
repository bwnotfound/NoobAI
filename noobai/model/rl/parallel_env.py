import copy

from threading import Thread, Event

# from queue import Queue

import numpy as np
import torch

from noobai.model.rl.reply_buffer import BaseReplyBuffer


class GymEnvBatchWrapper:
    def __init__(self, env, num_envs=1):
        if isinstance(env, list):
            self.envs = env
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
    state = 1
    action = 2
    reward = 3
    next_state = 4
    # next_action = 5
    # next_reward = 6
    done = 7
    cum_reward = 8  # accmulate reward calc by passed func.
    info = 9

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
        self.base_env = env
        self.num_workers = num_workers
        self.use_multithread = use_multithread
        if sample_data_mode is None:
            sample_data_mode = [
                GymEnvContinuousWrapper.state,
                GymEnvContinuousWrapper.action,
                GymEnvContinuousWrapper.next_state,
                GymEnvContinuousWrapper.reward,
                GymEnvContinuousWrapper.done,
            ]
        self.sample_data_mode = sample_data_mode
        if GymEnvContinuousWrapper.cum_reward in sample_data_mode:
            assert (
                cum_reward_func is not None
            ), 'cum_reward_func is not None when the mode of "value" is in sample_data_mode.'
        self.cum_reward_func = cum_reward_func
        self.need_cum_reward = (
            GymEnvContinuousWrapper.cum_reward in self.sample_data_mode
        )

        self.stop_event = Event()

    def start(self, model, reply_buffer: BaseReplyBuffer, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.stop_event.clear()

        def run():
            states, rewards, dones, infos, actions, next_states = (
                [[] for _ in range(self.num_workers)],
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
                                if data_mode == GymEnvContinuousWrapper.state:
                                    data.append(states[i][j])
                                elif data_mode == GymEnvContinuousWrapper.action:
                                    data.append(actions[i][j])
                                elif data_mode == GymEnvContinuousWrapper.reward:
                                    data.append(rewards[i][j])
                                elif data_mode == GymEnvContinuousWrapper.next_state:
                                    data.append(next_states[i][j])
                                elif data_mode == GymEnvContinuousWrapper.done:
                                    data.append(dones[i][j])
                                elif data_mode == GymEnvContinuousWrapper.cum_reward:
                                    data.append(cum_rewards[i][j])
                                elif data_mode == GymEnvContinuousWrapper.info:
                                    data.append(infos[i][j])
                                else:
                                    raise RuntimeError(
                                        f"Cannot specified data_mode: {data_mode}"
                                    )
                            reply_buffer.add(data)
                    state, _ = env_list[i].reset()
                    last_states[i] = state
                    env_is_dones[i] = False
                last_states_numpy = np.array(
                    [np.array(last_state) for last_state in last_states]
                )
                model_input = torch.tensor(
                    [torch.tensor(last_state) for last_state in last_states_numpy]
                ).to(device)
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
                        states[i].append(np.array(last_states[i]))
                        rewards[i].append(reward)
                        dones[i].append(done)
                        infos[i].append(info)
                        env_is_dones[i] = done
                        next_states[i].append(np.array(state))
                    else:
                        data = []
                        for data_mode in self.sample_data_mode:
                            if data_mode == GymEnvContinuousWrapper.state:
                                data.append(np.array(last_states[i]))
                            elif data_mode == GymEnvContinuousWrapper.action:
                                data.append(action_list_numpy[i])
                            elif data_mode == GymEnvContinuousWrapper.reward:
                                data.append(reward)
                            elif data_mode == GymEnvContinuousWrapper.next_state:
                                data.append(state)
                            elif data_mode == GymEnvContinuousWrapper.done:
                                data.append(done)
                            elif data_mode == GymEnvContinuousWrapper.info:
                                data.append(info)
                            else:
                                raise RuntimeError(
                                    f"Cannot specified data_mode: {data_mode}"
                                )
                        reply_buffer.add(data)

        Thread(target=run).start()

    def stop(self):
        self.stop_event.set()
