import numpy as np
from typing import Union


class GridWorldEnv(object):
    def __init__(self, size: int, terminal, forbidden, r_forbidden=-1, r_other=0, r_terminal=1, r_boundary=-1,r_stay=0,
                 transition_prob=None):
        self.size = size

        self.num_states = size * size
        self.num_actions = 5  # 上右下左原地

        self.r_forbidden = r_forbidden
        self.r_other = r_other
        self.r_terminal = r_terminal
        self.r_boundary = r_boundary
        self.r_stay = r_stay

        # 状态编号映射 假设size=3,(0,0)->0 (0,1)->1 (0,2)->2 (1,0)->3 (1,1)->4 ... (2,2)->8
        self.state_id = lambda x, y: x * size + y

        self.terminal = {self.state_id(x, y) for (x, y) in terminal}
        self.forbidden = {self.state_id(x, y) for (x, y) in forbidden}

        # 初始化
        self.PSsa = np.zeros((self.num_states, self.num_actions, self.num_states))  # +1表示超出边界
        self.PRsa = None  # p(r|s,a)
        self.reward_list = []
        self.reward_space_size = 0

        # 构造
        self.build_reward_list()
        self.build_transition_prob(transition_prob=transition_prob)
        self.build_PRsa()

    def build_transition_prob(self, transition_prob=None):  # 如果输入了transition_prob，必须强制三个key都要输入
        """
            transition_prob: dict, like
            {
                "forward": 0.8,
                "left":    0.1,
                "right":   0.1
            }
            """
        if transition_prob is None:  # 默认是dertiministric的
            transition_prob = {"forward": 1.0, "left": 0.0, "right": 0.0}

        moves = {
            0: (-1, 0),  # 上
            1: (0, 1),  # 右
            2: (1, 0),  # 下
            3: (0, -1),  # 左
            4: (0, 0)  # 原地
        }

        # 做第i个动作，左偏/右偏后执行的动作
        left_turn = {0: 3, 1: 0, 2: 1, 3: 2, 4: 4}  # 逆时针
        right_turn = {0: 1, 1: 2, 2: 3, 3: 0, 4: 4}  # 顺时针

        for i in range(self.size):
            for j in range(self.size):
                s = self.state_id(i, j)

                for a in range(self.num_actions):
                    if a == 4:
                        possible_dirs = {4: 1.0}  # 字典的键不可以重复，理论上是应该累加为1，但是最终只会保留一项
                    else:
                        possible_dirs = {
                            a: transition_prob["forward"],
                            left_turn[a]: transition_prob["left"],
                            right_turn[a]: transition_prob["right"]
                        }

                    for a2, prob in possible_dirs.items():
                        di, dj = moves[a2]
                        ni, nj = i + di, j + dj

                        # 越界则原地
                        if not (0 <= ni < self.size and 0 <= nj < self.size):
                            ni, nj = i, j
                        s_next = self.state_id(ni, nj)

                        self.PSsa[s, a, s_next] += prob

    def reward_func(self, s, s_next, action):
        """根据当前状态、下一个状态、采取的动作确定该步动作产生的奖励"""
        if s_next in self.terminal:
            return self.r_terminal
        if s_next in self.forbidden:
            return self.r_forbidden
        if s_next == s and action != 4:
            return self.r_boundary
        if s_next==s and action==4:
            return self.r_stay
        return self.r_other

    def build_PRsa(self):
        """
        构造 PRsa[state, action, reward_index]
        因为 reward 只依赖 next_state，所以：
        p(r|s,a) = sum_{s'} p(s'|s,a) * [reward(s') == r]
        """
        self.PRsa = np.zeros((self.num_states, self.num_actions, self.reward_space_size))

        for s in range(self.num_states):
            for a in range(self.num_actions):
                for s_next in range(self.num_states):
                    prob = self.PSsa[s, a, s_next]
                    if prob > 0:
                        r = self.reward_func(s, s_next, a)
                        r_idx = self.reward_list.index(r)
                        self.PRsa[s, a, r_idx] += prob  # deterministic MDP→每个(s,a)只有一个s_next,+=和=效果一样

    def build_reward_list(self):
        reward = set()
        reward.add(self.r_forbidden)
        reward.add(self.r_other)
        reward.add(self.r_terminal)
        reward.add(self.r_boundary)
        reward.add(self.r_stay)

        self.reward_list = sorted(list(reward))
        self.reward_space_size = len(self.reward_list)

    def step(self, state, action):
        '''
        :param action: 当前所在的state_id
        :param action: 当前采取的动作
        :return: state,reward,done  到达的下一个状态的state_id，获取的奖励，是否走到了终点
        '''
        if state in self.terminal:
            return state, self.r_terminal, True

        i, j = divmod(state, self.size)

        moves = {
            0: (-1, 0),  # 上
            1: (0, 1),  # 右
            2: (1, 0),  # 下
            3: (0, -1),  # 左
            4: (0, 0)  # 原地
        }
        di, dj = moves[action]
        ni, nj = i + di, j + dj

        if not (0 <= ni < self.size and 0 <= nj < self.size):
            next_state = self.state_id(i,j)
        else:
            next_state = self.state_id(ni,nj)

        reward = self.reward_func(state, next_state, action)

        done = next_state in self.terminal

        return next_state, reward, done

    def reset(self):
        return 0

    def generate_episodes(self, policy, start_state, start_action, max_steps=500):
        '''
        :param policy: 当前策略
        :param start_state: 当前状态的state_id
        :param start_action: 当前动作
        :return: [(state_id, action，reward,next_state_id),(...)]
        '''
        episode = []
        state = start_state
        action = start_action
        for _ in range(max_steps):
            next_state, reward, done = self.step(state, action)
            episode.append((state, action, reward, next_state))
            if done:
                break
            state = next_state
            action = np.random.choice(self.num_actions,
                                      p=policy[state])  # 从[0,action_space_size)随机选一个，每个action的概率为policy[state]
        return episode


if __name__ == '__main__':
    transition_prob = {
        "forward": 0.8,
        "left": 0.1,
        "right": 0.1
    }

    env = GridWorldEnv(size=3, terminal=[(2, 2)], forbidden=[(1, 1)])

    for s in range(env.num_states):
        for a in range(env.num_actions):
            print(f"s={s}, a={a}, next={env.PSsa[s, a]}")

    for s in range(env.num_states):
        for a in range(env.num_actions):
            print(f"s={s}, a={a}, PRsa={env.PRsa[s, a]}")
