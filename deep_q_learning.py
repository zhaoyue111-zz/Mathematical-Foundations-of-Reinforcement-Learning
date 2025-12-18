import numpy as np
import torch
from torch import nn
from torch.utils import data

from env import GridWorldEnv
from utils import drow_policy


class DeepQLearning(object):
    def __init__(self, env: GridWorldEnv, gamma=0.9):
        self.env = env
        self.action_space_size = self.env.num_actions
        self.state_space_size = self.env.num_states

        self.gamma = gamma

        self.policy = np.ones((self.state_space_size, self.action_space_size)) / self.action_space_size

        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, self.action_space_size)
        )
        self.target_net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, self.action_space_size)
        )
        self.target_net.load_state_dict(self.net.state_dict())

        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=0.01, momentum=0.9)
        self.loss = nn.MSELoss()

    def data_iter(self, episode, batch_size=32, is_train=True):
        reward = []
        state = []
        action = []
        next_state = []
        for s, a, r, next_s in episode:
            reward.append(r)
            state.append((s // self.env.size, s % self.env.size))  # 有空间位置
            next_state.append((next_s // self.env.size, next_s % self.env.size))
            action.append(a)
        reward = torch.tensor(reward, dtype=torch.float32)
        state = torch.tensor(state, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.long)

        data_arrays = (state, action, reward, next_state)
        dataset = data.TensorDataset(*data_arrays)
        return data.DataLoader(dataset, batch_size=batch_size, shuffle=is_train, drop_last=False)

    def solve(self, epochs, update_frep):
        s = self.env.reset()
        a = np.random.choice(self.action_space_size, p=self.policy[s])
        episodes = []
        for _ in range(self.action_space_size * self.state_space_size + 1):
            episode = self.env.generate_episodes(self.policy, s, a, max_steps=1000)
            episodes.extend(episode)

        dataloader = self.data_iter(episodes)
        step = 0
        for epoch in range(epochs):
            for state, action, reward, next_state in dataloader:
                step += 1
                with torch.no_grad():
                    q_value = self.target_net(next_state)  # [B,N]
                    max_q = q_value.max(dim=1).values
                    y_target = reward + self.gamma * max_q  # [B,]

                y = self.net(state)
                y_ = y.gather(1, action.unsqueeze(1)).squeeze(1)
                l = self.loss(y_target, y_)
                self.optimizer.zero_grad()
                l.backward()
                self.optimizer.step()

                if step % update_frep == 0:
                    self.target_net.load_state_dict(self.net.state_dict())

    def get_policy(self):
        for s in range(self.state_space_size):
            if s in self.env.terminal:
                self.policy[s, 4] = 1
                break

            s_t = torch.tensor((s // self.env.size, s % self.env.size), dtype=torch.float32)
            q_value = self.net(s_t)
            a = q_value.argmax(dim=0).item()

            self.policy[s] = 0
            self.policy[s, a] = 1

        return self.policy


if __name__ == '__main__':
    env = GridWorldEnv(
        size=5,
        forbidden=[(1, 2), (3, 3)],
        terminal=[(4, 4)],
        r_boundary=-1,
        r_other=0,
        r_terminal=1,
        r_forbidden=-1,
        r_stay=-0.1
    )

    # 注意samples要大一点，否则每个state被访问到的概率很小
    vi = DeepQLearning(env=env, gamma=0.8)

    vi.solve(epochs=50, update_frep=20)
    policy = vi.get_policy()
    print(policy)

    drow_policy(policy, env)
