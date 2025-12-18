import numpy as np

from env import GridWorldEnv
from utils import drow_policy


class PolicyIteration(object):
    def __init__(self, env: GridWorldEnv, gamma=0.9):
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
        V = np.zeros(self.state_space_size)
        policy = np.ones((self.state_space_size, self.action_space_size)) / self.action_space_size  # 不能初始化为0，否则初步乘积权威0

        while steps > 0:
            steps -= 1

            # -------------------------
            # Policy Evaluation
            # -------------------------
            while True:
                V_new = np.zeros_like(V)  # 每次迭代更新V都要从头开始

                for state in range(self.state_space_size):
                    if state in self.env.terminal:
                        continue

                    for action in range(self.action_space_size):
                        qvalue = np.dot(self.reward_list, self.PRsa[state, action])
                        qvalue += self.gamma * np.dot(V, self.PSsa[state, action])
                        V_new[state] += policy[state, action] * qvalue

                if np.linalg.norm(V_new - V) < threshold:
                    break

                V = V_new

            # -------------------------
            # Policy Improvement
            # -------------------------
            policy_stable = True
            policy_new = np.zeros_like(policy)
            for state in range(self.state_space_size):
                if state in self.env.terminal:
                    policy_new[state, 4] = 1
                    continue

                qtable_s = []
                for action in range(self.action_space_size):
                    qvalue = np.dot(self.reward_list, self.PRsa[state, action])
                    qvalue += self.gamma * np.dot(V, self.PSsa[state, action])
                    qtable_s.append(qvalue)

                action_star = np.argmax(qtable_s)
                policy_new[state, action_star] = 1

                if not np.array_equal(policy_new[state], policy[state]):
                    policy_stable = False

            policy = policy_new

            if policy_stable:
                break

        self.policy = policy
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
        forbidden=[(1, 2), (2, 4)],
        terminal=[(4, 4)],
        # transition_prob=transition_prob,
        r_boundary=-10,
        r_other=0,
        r_terminal=1,
        r_forbidden=-1
    )

    vi = PolicyIteration(env=env,gamma=0.9)

    steps=vi.solve()
    print(steps)

    print("状态价值 V(s):")
    print(vi.state_value.reshape(env.size, env.size))

    print("\n策略 π(s):")
    print(vi.policy)

    drow_policy(vi.policy,env)