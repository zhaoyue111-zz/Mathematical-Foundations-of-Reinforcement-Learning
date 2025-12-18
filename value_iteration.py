import numpy as np

from env import GridWorldEnv
from utils import drow_policy


class ValueIteration(object):
    def __init__(self, env: GridWorldEnv,gamma=0.9):
        self.env = env
        self.action_space_size = self.env.num_actions  # 上下左右原地

        self.reward_list = self.env.reward_list
        self.PRsa = self.env.PRsa
        self.PSsa = self.env.PSsa
        self.reward_space_size = self.env.reward_space_size  # 执行每个动作的reward
        self.state_space_size = self.env.num_states  # 每个点都是一个状态

        self.gamma = gamma

        self.state_value = np.zeros(self.state_space_size)  # v(k-1)
        self.policy = np.zeros((self.state_space_size, self.action_space_size))

    def solve(self, threshold=0.001, steps=100):
        V = np.zeros(self.state_space_size)  # v(k)
        while steps > 0:
            steps -= 1
            V_new = np.zeros_like(V)
            policy_new = np.zeros((self.state_space_size, self.action_space_size))  # TODO：每轮迭代开始policy都应该为0
            q_table = np.zeros((self.state_space_size, self.action_space_size))  # 在每个状态执行每个动作的概率为0

            for state in range(self.state_space_size):
                qtable_s = []  # q_tabel的第state行

                if state in self.env.terminal:
                    V_new[state] = 0
                    policy_new[state, 4] = 1  # 原地
                    continue

                for action in range(self.action_space_size):
                    qvalue = np.dot(self.reward_list, self.PRsa[state, action])
                    qvalue += self.gamma * np.dot(V, self.PSsa[state, action])
                    qtable_s.append(qvalue)

                q_table[state, :] = qtable_s.copy()  # 更新q_tabel
                V_new[state] = np.max(qtable_s)  # v(k+1)=max qk(s,a)

                action_star = np.argmax(qtable_s)  # a*=argmax qk(s,a)
                policy_new[state, action_star] = 1  # 更新策略

            if np.linalg.norm(V_new - V) < threshold:  # |v(k)-v(k-1)|<=threshold
                break

            V = V_new

        self.policy = policy_new
        self.state_value = V

        return steps

if __name__ == '__main__':
    # transition_prob = {
    #     "forward": 0.8,
    #     "left": 0.05,
    #     "right": 0.15
    # }

    env = GridWorldEnv(
        size=5,
        forbidden=[(1, 2),(2,4)],
        terminal=[(4,4)],
        # transition_prob=transition_prob,
        r_boundary=-10,
        r_other=0,
        r_terminal=1,
        r_forbidden=-1
    )

    vi = ValueIteration(env=env,gamma=0.9)

    steps=vi.solve()
    print(steps)

    print("状态价值 V(s):")
    print(vi.state_value.reshape(env.size, env.size))

    print("\n策略 π(s):")
    print(vi.policy)

    drow_policy(vi.policy,env)