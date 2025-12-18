import numpy as np
import torch
from torch import nn

from env import GridWorldEnv
from utils import drow_policy

'''
off-policy actor-critic based on importance sampling
'''
class OffPolicyA2C(object):
    def __init__(self, env: GridWorldEnv, gamma=0.9, lr_actor=1e-2, lr_critic=1e-2):
        self.env = env
        self.action_space_size = self.env.num_actions
        self.state_space_size = self.env.num_states
        self.gamma = gamma

        # 目标策略 π(a|s, θ)
        self.pnet = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, self.action_space_size)
        )

        # 值函数 v(s, w)
        self.vnet = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

        self.value_optimizer = torch.optim.Adam(self.vnet.parameters(), lr=lr_critic)
        self.policy_optimizer = torch.optim.Adam(self.pnet.parameters(), lr=lr_actor)

        # 行为策略 β(a|s)：这里用均匀随机策略
        self.behaviour_policy = np.ones((self.state_space_size, self.action_space_size)) / self.action_space_size

        self.policy = np.zeros((self.state_space_size, self.action_space_size))
        self.state_value = np.zeros(self.state_space_size)

    def decode_state(self, state_int: int):
        i = state_int // self.env.size
        j = state_int % self.env.size
        return torch.tensor(
            (i / (self.env.size - 1), j / (self.env.size - 1)),
            dtype=torch.float32
        )

    def sample_from_behavior(self, state_int: int):
        """
        :return: action(int), beta_as(float)
        """
        beta_probs = self.behaviour_policy[state_int]
        action = np.random.choice(self.action_space_size, p=beta_probs)
        beta_as = beta_probs[action]
        return action, beta_as

    def pi_log_prob(self, state_tensor: torch.Tensor, action: int):
        '''
        :return: π(a_t|s_t,θ_t), log π(a_t|s_t,θ_t)
        '''
        logits = self.pnet(state_tensor)
        action_probs = torch.softmax(logits, dim=0)
        pi_a = action_probs[action]
        log_prob = torch.log(pi_a)          # ln π(a|s,θ)
        return pi_a, log_prob

    def solve(self, num_episodes=200):
        for _ in range(num_episodes):
            state_int = self.env.reset()
            state = self.decode_state(state_int)

            done = False
            while not done:
                action, beta_as = self.sample_from_behavior(state_int)  # a_t, β(a_t|s_t)

                next_state_int, reward, done = self.env.step(state_int, action)
                next_state = self.decode_state(next_state_int)

                pi_a, log_prob = self.pi_log_prob(state, action)  # π(a_t|s_t,θ_t), ln π(a_t|s_t,θ_t)

                rho = pi_a.detach() / beta_as

                v_s = self.vnet(state).squeeze()
                if done:
                    td_target = torch.tensor(reward, dtype=torch.float32)
                else:
                    with torch.no_grad():
                        v_s_next = self.vnet(next_state).squeeze()
                        td_target = torch.tensor(reward, dtype=torch.float32) + self.gamma * v_s_next

                # Advantage (TD error)
                delta = td_target - v_s

                # Critic
                self.value_optimizer.zero_grad()
                critic_loss=-delta.detach() * rho * v_s
                critic_loss.backward()
                self.value_optimizer.step()

                # Actor
                self.policy_optimizer.zero_grad()
                actor_loss = - rho * delta.detach() * log_prob
                actor_loss.backward()
                self.policy_optimizer.step()

                state_int = next_state_int
                state = next_state

    def get_policy_by_policy_net(self):
        for s in range(self.state_space_size):
            if s in self.env.terminal:
                self.policy[s, 4] = 1
                continue

            s_t = self.decode_state(s)
            logits = self.pnet(s_t)
            action_probs = torch.softmax(logits, dim=0)
            a = torch.argmax(action_probs).item()
            self.policy[s, a] = 1

        return self.policy

    def get_state_value(self):
        for s in range(self.state_space_size):
            s_t = self.decode_state(s)
            v = self.vnet(s_t).item()
            self.state_value[s] = v
        return self.state_value.reshape(self.env.size,self.env.size)


if __name__ == '__main__':
    env = GridWorldEnv(
        size=5,
        forbidden=[(1, 2), (3, 3)],
        terminal=[(4, 4)],
        r_boundary=-1,
        r_other=-0.04,
        r_terminal=1,
        r_forbidden=-1,
        r_stay=-0.1
    )

    vi = OffPolicyA2C(env=env)

    vi.solve(num_episodes=20)

    print("\n state value: ")
    print(vi.get_state_value())

    print("\n get policy by policy net:")
    drow_policy(vi.get_policy_by_policy_net(), env)