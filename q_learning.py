import numpy as np

from env import GridWorldEnv
from utils import drow_policy


class Q_Learning(object):
    def __init__(self, env: GridWorldEnv, gamma=0.9, alpha=0.001, epsilon=0.1, samples=1, start_state=(0, 0),mode="on policy"):
        '''
        :param env: 定义了网格的基础配置
        :param gamma: discount rate
        :param alpha:  learning rate
        :param samples:  从起点到终点采样的路径数
        :param start_state:  起点
        :param mode: 模式
        '''
        self.env = env
        self.action_space_size = self.env.num_actions  # 上下左右原地
        self.state_space_size = self.env.num_states

        self.reward_list = self.env.reward_list
        self.gamma = gamma
        self.samples = samples
        self.alpha = alpha
        self.epsilon = epsilon
        self.mode=mode
        self.start_state = self.env.state_id(start_state[0], start_state[1])

        self.behavior_policy = np.ones(
            (self.state_space_size, self.action_space_size)) / self.action_space_size  # 探索性很强
        self.target_policy = np.zeros((self.state_space_size, self.action_space_size))
        self.qvalues = np.zeros((self.state_space_size, self.action_space_size))


    def update_qvalues(self,s_t,a_t,s_next,r_next):
        max_q_next = np.max(self.qvalues[s_next])
        td_target = r_next + self.gamma * max_q_next
        td_error = td_target - self.qvalues[s_t][a_t]  # 负号提出去
        self.qvalues[s_t][a_t] += self.alpha * td_error


    def solve(self):
        if self.mode=="off policy":
            for _ in range(self.samples):
                s = self.start_state
                a = np.random.choice(self.action_space_size, p=self.behavior_policy[s])
                episode = self.env.generate_episodes(self.behavior_policy, s, a)

                for i in range(len(episode)):
                    s_t, a_t, r_next_t, s_next_t= episode[i]

                    self.update_qvalues(s_t,a_t,s_next_t,r_next_t)

                    # greedy
                    best_a = np.argmax(self.qvalues[s_t])
                    self.target_policy[s_t] = np.eye(self.action_space_size)[best_a]

        elif self.mode=="on policy":  # target_policy=behavior_policy
            for _ in range(self.samples):
                s = self.start_state

                while s not in self.env.terminal:
                    a = np.random.choice(self.action_space_size, p=self.behavior_policy[s])  # generate at following πt(st)
                    next_s, next_r, _ = self.env.step(s, a)  # generate rt+1, st+1 by interacting with the environment

                    # updata q-value for (s_t,a_t)
                    # qt+1(st, at) = qt(st, at) − αt(st, at)  [  qt(st, at) − (rt+1 + γ max(qt(st+1, a)))]
                    self.update_qvalues(s,a,next_s,next_r)

                    # update policy for s_t: epsilon greedy 因为要用policy生成数据，因此需要策略具有一定的探索性，因此使用epsilon greedy
                    best_a = np.argmax(self.qvalues[s])
                    self.behavior_policy[s] = self.epsilon / self.action_space_size
                    self.behavior_policy[s, best_a] += 1 - self.epsilon

                    self.target_policy=self.behavior_policy
                    s = next_s
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

    # 注意samples要大一点，否则每个state被访问到的概率很小
    vi = Q_Learning(env=env, gamma=0.8, alpha=0.01, samples=1000, start_state=(0, 0),mode="off policy")

    vi.solve()

    print("\n state value: ")
    print(vi.qvalues)

    drow_policy(vi.target_policy, env)