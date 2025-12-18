from collections import defaultdict

import numpy as np

from env import GridWorldEnv
from utils import drow_policy


class MonteCarloPolicyIteration(object):
    def __init__(self, env: GridWorldEnv, gamma=0.9,  epsilon=0.1, samples=1, mode="first visit"):
        self.env = env
        self.action_space_size = self.env.num_actions  # 上下左右原地
        self.reward_space_size = self.env.reward_space_size  # 执行每个动作的reward
        self.state_space_size = self.env.num_states

        self.reward_list = self.env.reward_list
        self.gamma = gamma
        self.epsilon = epsilon
        self.samples = samples
        self.mode = mode

        self.policy = np.ones((self.state_space_size, self.action_space_size)) / self.action_space_size
        self.state_value = np.zeros((self.env.size, self.env.size))
        self.qvalues = np.zeros((self.state_space_size, self.action_space_size))

        self.returns = np.zeros((self.state_space_size, self.action_space_size))  # 必须初始化为0，不是zeros_like
        self.nums = np.zeros((self.state_space_size, self.action_space_size))

    def solve(self, iterations=20):
        '''
        :param iterations: 迭代的次数
        :param epsilon: epsilon greedy：[0,1] epsilon=0：greedy，就选择best action；epsilon=1:stochastic，选择所有action的概率相同
        '''
        for i in range(iterations):
            for _ in range(self.samples):
                # 随机选择一个非终点状态作为起始状态,确保所有的状态都能被充分访问
                non_terminal_states = [i for i in range(self.state_space_size) if i not in self.env.terminal]
                s = np.random.choice(non_terminal_states)
                a = np.random.choice(self.action_space_size, p=self.policy[s])  # 按policy采样
                episode = self.env.generate_episodes(self.policy, s, a)
                self.update_q_from_episode(episode)

            for s in range(self.state_space_size):
                if s in self.env.terminal:
                    self.policy[s] = np.eye(self.action_space_size)[4]
                else:
                    best_a = np.argmax(self.qvalues[s])
                    if self.mode == "every visit":  # 如果是first visit，很多(s,t)可能被访问了很多次，但是却只用它做了一次action value的估计
                        # epsilon greedy
                        self.policy[s] = self.epsilon / self.action_space_size  # 给其他action小概率
                        self.policy[s, best_a] += 1 - self.epsilon  # 给最有可能的action大概率
                    elif self.mode == "first visit":  # 实际对应epsilon=0的情况
                        self.policy[s] = np.eye(self.action_space_size)[best_a]

            self.state_value = np.sum(self.policy * self.qvalues, axis=1).reshape(self.env.size, self.env.size)

    def update_q_from_episode(self, episode):
        G = 0
        visit = set()
        for s, a, r, _ in reversed(episode):  # 如果直接使用reversed(episode)就会同时把tuple内部也反转了
            G = r + self.gamma * G
            if self.mode == "first visit":
                if (s, a) not in visit:
                    self.returns[s, a] += G
                    self.nums[s, a] += 1
                    self.qvalues[s, a] = self.returns[s, a] / self.nums[s, a]
            elif self.mode == "every visit":
                self.returns[s, a] += G
                self.nums[s, a] += 1
                self.qvalues[s, a] = self.returns[s, a] / self.nums[s, a]
            else:
                raise Exception("Invalid mode")


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

    vi = MonteCarloPolicyIteration(env=env, gamma=0.9, epsilon=0.9, samples=10, mode="every visit")

    vi.solve(iterations=100)  # 只有mode="every visit"才需要传入epsilon

    print("\n state value: ")
    print(vi.state_value)

    drow_policy(vi.policy, env)
