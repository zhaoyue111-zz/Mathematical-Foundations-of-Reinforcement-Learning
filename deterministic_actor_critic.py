import numpy as np
import torch
from torch import nn

from env import GridWorldEnv


class DeterministicAC(object):
    def __init__(self, env: GridWorldEnv, gamma=0.99, lr_actor=1e-3, lr_critic=1e-3, noise_std=0.1):
        self.env = env
        self.action_space_size = self.env.num_actions
        self.state_space_size = self.env.num_states
        self.gamma = gamma
        self.noise_std = noise_std

        # 确定性策略 μ(s,θ)
        self.actor = nn.Sequential(
            nn.Linear(2, 16),  # state
            nn.ReLU(),
            nn.Linear(16, 1),  # action
            nn.Tanh()
        )

        # q(s,a,w)
        self.critic = nn.Sequential(
            nn.Linear(3, 32),  # [state,action]
            nn.ReLU(),
            nn.Linear(32, 1)  # q(state,action)
        )

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.policy=np.zeros((self.state_space_size, self.action_space_size))

    def decode_state(self, state_int):
        i = state_int // self.env.size
        j = state_int % self.env.size
        return torch.tensor(
            (i / (self.env.size - 1), j / (self.env.size - 1)),
            dtype=torch.float32
        )

    def behavior_action(self, state_tensor):
        """
        β(a|s) = μ(s) + 噪声
        :return: action_tensor (用于 critic), action_for_env (numpy 标量)
        """
        with torch.no_grad():
            mu = self.actor(state_tensor)  # μ(s,θ)
        noise = torch.randn_like(mu) * self.noise_std  # noise
        a_beta = mu + noise  # β(a|s)
        return a_beta, a_beta.item()

    def critic_q(self, state_tensor, action_tensor):
        '''
        :return: q(s,a)  [1, 1]
        '''
        sa = torch.cat([state_tensor, action_tensor], dim=-1)
        return self.critic(sa).squeeze()

    def solve(self, num_episodes=200):
        for _ in range(num_episodes):
            state_int = self.env.reset()
            state = self.decode_state(state_int)

            done = False
            while not done:
                action_tensor, action_env = self.behavior_action(state)

                next_state_int, reward, done = self.env.step(state_int, action_env)
                next_state = self.decode_state(next_state_int)

                q_sa = self.critic_q(state, action_tensor)  # q(s_t, a_t, w_t)

                with torch.no_grad():
                    if done:
                        target_q = torch.tensor(reward, dtype=torch.float32)
                    else:
                        mu_next = self.actor(next_state)  # μ(s_t+1, θ_t)
                        q_next = self.critic_q(next_state, mu_next)  # q(s_t+1, μ(s_t+1, θ_t), w_t)
                        target_q = torch.tensor(reward, dtype=torch.float32) + self.gamma * q_next

                # TD error
                delta = target_q - q_sa

                # Critic
                self.critic_opt.zero_grad()
                critic_loss = 0.5 * delta.pow(2)
                critic_loss.backward()
                self.critic_opt.step()

                # Actor
                self.actor_opt.zero_grad()
                mu_action = self.actor(state)  # μ(s_t,θ)
                q_mu = self.critic_q(state, mu_action)  # q(s_t, μ(s_t), w_t)
                actor_loss = - q_mu
                actor_loss.backward()
                self.actor_opt.step()

                state_int = next_state_int
                state = next_state