
'''
区别：1. sarsa是从一个特定的开始状态出发，到达目标状态，只有这条episode是最优的，其他状态则不一定
2. sarsa是迭代式算法，每更新一次action value就要更新一次policy
'''
import random

import numpy as np
from prometheus_client import samples

from env import GridWorldEnv
from utils import drow_policy


class NStepSarsa(object):
    def __init__(self, env:GridWorldEnv, gamma=0.9, alpha=0.001, epsilon=0.1, samples=1, start_state=(0,0), n_step=5):
        '''
        :param env: 定义了网格的基础配置
        :param gamma: discount rate
        :param alpha:  learning rate
        :param epsilon:  epsilon greedy更新policy
        :param samples:  从起点到终点采样的路径数
        :param start_state:  起点
        :param n_step: action value往后加多少步   n=1 -> Sarsa  n=∞ -> MC learning，但不能传入∞
        '''
        self.env = env
        self.action_space_size = self.env.num_actions  # 上下左右原地
        self.state_space_size = self.env.num_states

        self.reward_list = self.env.reward_list
        self.gamma = gamma
        self.samples = samples
        self.alpha = alpha
        self.epsilon=epsilon
        self.start_state = self.env.state_id(start_state[0],start_state[1])
        self.n_step = n_step

        self.policy = np.ones((self.state_space_size, self.action_space_size)) / self.action_space_size
        self.qvalues = np.zeros((self.state_space_size, self.action_space_size))

    def solve(self):

        n = self.n_step

        for episode in range(self.samples):

            s = self.start_state
            a = np.random.choice(self.action_space_size, p=self.policy[s])

            # 轨迹缓存
            S = [s]
            A = [a]
            R = [0]  # R[0] 占位

            T = float('inf')  # 终止时间步
            t = 0

            while True:

                if t < T:
                    next_s, next_r, _ = self.env.step(s, a)
                    S.append(next_s)
                    R.append(next_r)

                    if next_s in self.env.terminal:
                        T = t + 1
                    else:
                        next_a = np.random.choice(self.action_space_size, p=self.policy[next_s])
                        A.append(next_a)

                # tau 是要更新的时间步
                tau = t - n + 1
                if tau >= 0:  # 类似于滑动窗口的left必须>=0
                    # 计算 n-step return
                    G = 0
                    for i in range(tau + 1, min(tau + n, T) + 1):  # rt+1...rt+n  t=tau
                        G += (self.gamma ** (i - tau - 1)) * R[i]

                    if tau + n < T:
                        G += (self.gamma ** n) * self.qvalues[S[tau + n]][A[tau + n]]

                    # 更新 Q
                    s_tau = S[tau]
                    a_tau = A[tau]
                    td_error = G - self.qvalues[s_tau][a_tau]
                    self.qvalues[s_tau][a_tau] += self.alpha * td_error

                    # 更新策略
                    best_a = np.argmax(self.qvalues[s_tau])
                    self.policy[s_tau] = self.epsilon / self.action_space_size
                    self.policy[s_tau, best_a] += 1 - self.epsilon

                if tau == T - 1:
                    break

                t += 1
                s = next_s
                if next_s not in self.env.terminal:
                    a = next_a

    # sarsa
    # def solve(self):
    #
    #     for i in range(self.samples):
    #         s = self.start_state
    #         a = np.random.choice(self.action_space_size, p=self.policy[s])
    #
    #         while s not in self.env.terminal:
    #             next_s, next_r, _ = self.env.step(s,a)
    #             next_a= np.random.choice(self.action_space_size, p=self.policy[next_s]) # 根据Πt(s_t+1)生成a_t+1
    #
    #             # updata q-value for (s_t,a_t)
    #             # qt+1(st, at) = qt(st, at) − αt(st, at)  [  qt(st, at) − (rt+1 + γqt(st+1, at+1))]
    #             td_target=next_r+self.gamma*self.qvalues[next_s][next_a]
    #             td_error=td_target-self.qvalues[s][a]  # 负号提出去
    #             self.qvalues[s][a]+=self.alpha*td_error
    #
    #             # update policy for s_t
    #             best_a=np.argmax(self.qvalues[s])
    #             self.policy[s] = self.epsilon / self.action_space_size
    #             self.policy[s, best_a] += 1 - self.epsilon
    #
    #             s, a = next_s, next_a


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
    vi=NStepSarsa(env=env, gamma=0.9, alpha=0.01, epsilon=0.1, samples=500, start_state=(0,0), n_step=1000)

    vi.solve()

    print("\n state value: ")
    print(vi.qvalues)

    drow_policy(vi.policy, env)