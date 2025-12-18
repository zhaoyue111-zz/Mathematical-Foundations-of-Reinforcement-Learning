import numpy as np
import torch
from torch import nn

from env import GridWorldEnv
from utils import drow_policy

'''
policy gradient by Monte Carlo
'''
class Reinforce(object):
    def __init__(self, env: GridWorldEnv, gamma=0.9, lr=1e-2):
        '''
        :param env:
        :param gamma: discount rate
        :param lr:   learning rate of optimizer
        '''
        self.env = env
        self.action_space_size = self.env.num_actions
        self.state_space_size = self.env.num_states

        self.gamma = gamma

        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, self.action_space_size)
        )
        self.policy = np.zeros((self.state_space_size, self.action_space_size))
        self.q_value = np.zeros((self.state_space_size, self.action_space_size))
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

    def decode_state(self, state):
        '''
        :param state: int
        :return: 归一化后的元组
        '''
        i = state // self.env.size
        j = state % self.env.size
        return torch.tensor((i / (self.env.size - 1), j / (self.env.size - 1)), dtype=torch.float32)

    def solve(self, num_episodes):
        for _ in range(num_episodes):
            state_int = self.env.reset()
            state = self.decode_state(state_int)

            done = False
            episode = []  # [[state_tensor,reward,done]...[...]]
            while not done:
                logits = self.net(state)
                action_probs = torch.softmax(logits, dim=0)
                action_dist = torch.distributions.Categorical(action_probs)  # 按分布采样
                action = action_dist.sample().item()

                next_state, reward, done = self.env.step(state_int, action)
                episode.append((state, action, reward))
                state_int = next_state
                state = self.decode_state(next_state)

            # value update
            returns = []
            G = 0
            for _, _, reward in reversed(episode):  # 使用MC的方法估计q_t
                G = reward + self.gamma * G
                returns.insert(0, G)

            # policy update
            self.optimizer.zero_grad()
            loss = 0
            for (state, action, _), G in zip(episode, returns):
                logits = self.net(state)
                action_probs = torch.softmax(logits, dim=0)
                action_dist = torch.distributions.Categorical(action_probs)
                log_prob = action_dist.log_prob(torch.tensor(action))  # In Π(a_t|s_t, θ)
                loss -= log_prob * G  # 负号是因为最小化 loss->最大化 J(θ)，梯度上升更新参数

            loss.backward()
            self.optimizer.step()

    def get_policy(self):
        for s in range(self.state_space_size):
            a = np.argmax(self.q_value[s])
            self.policy[s, a] = 1
        return self.policy

    def get_qvalues(self):
        for s in range(self.state_space_size):
            s_t = self.decode_state(s)
            logits = self.net(s_t).detach().numpy()
            self.q_value[s,:] = logits  # q_value是numpy类型，action_probs是tensor，必须转换
        return self.q_value


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

    # 注意samples要大一点，否则每个state被访问到的概率很小
    vi = Reinforce(env=env)

    vi.solve(num_episodes=200)

    print("\n state value: ")
    print(vi.get_qvalues())

    drow_policy(vi.get_policy(), env)