import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l5 = nn.Linear(hidden_dim, hidden_dim)
        self.l6 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class Delay(object):
    def __init__(
        self,
        state_dim,
        actor_hidden_dim,
        critic_hidden_dim,
        actor_lr,
        critic_lr,
        action_dim,
        max_action,
        vision_delay_frame,
        act_delay_frame,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        device="cuda",
    ):
        '''
            输入输出和value都是针对输入的state，act将在未来生效
        '''
        self.device = device
        self.actor = Actor(state_dim, actor_hidden_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = Critic(state_dim, critic_hidden_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.vision_delay_frame = vision_delay_frame
        self.act_delay_frame = act_delay_frame

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().detach().numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # Sample replay buffer
        sampled_data = replay_buffer.sample(batch_size)
        sampled_data = [data.to(self.device) for data in sampled_data]
        sampled_data = [
            data if len(data.shape) > 1 else data.unsqueeze(1) for data in sampled_data
        ]
        state, action, next_state, reward, done = sampled_data
        with torch.no_grad():
            # Select action according to policy and add clipped noise
            next_action = self.actor_target(next_state)
            next_action = next_action + (
                torch.randn_like(next_action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)
            next_action = next_action.clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done) * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q
        )

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        stats = {
            "critic_loss": critic_loss.item(),
            "target_Q_mean": target_Q.mean().item(),
            "reward_mean": reward.mean().item(),
            "current_Q1_mean": current_Q1.mean().item(),
            "current_Q2_mean": current_Q2.mean().item(),
        }
        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor losse
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(
                self.critic.parameters(), self.critic_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

            for param, target_param in zip(
                self.actor.parameters(), self.actor_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )
            stats["actor_loss"] = actor_loss.item()
        return stats

    def dumps(self):
        return {
            "critic": self.critic.state_dict(),
            "critic_optim": self.critic_optimizer.state_dict(),
            "actor": self.actor.state_dict(),
            "actor_optim": self.actor_optimizer.state_dict(),
        }

    def loads(self, data):
        if "critic" in data:
            self.critic.load_state_dict(data["critic"])
        if "critic_optim" in data:
            self.critic_optimizer.load_state_dict(data["critic_optim"])
        self.critic_target = copy.deepcopy(self.critic)

        if "actor" in data:
            self.actor.load_state_dict(data["actor"])
        if "actor_optim" in data:
            self.actor_optimizer.load_state_dict(data["actor_optim"])
        self.actor_target = copy.deepcopy(self.actor)

    def state_dict(self):
        return self.dumps()

    def load_state_dict(self, data):
        return self.loads(data)
