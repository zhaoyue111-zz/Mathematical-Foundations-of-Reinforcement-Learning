import numpy as np
import torch
from torch import nn

from env import GridWorldEnv
from utils import drow_policy

'''
Simple Actor-Critic
Actor: reinforce
Critic: TD learning with value function approximation
'''
class QAC(object):
    def __init__(self, env: GridWorldEnv, gamma=0.9, lr_actor=1e-2, lr_critic=1e-2):
        self.env = env
        self.action_space_size = self.env.num_actions
        self.state_space_size = self.env.num_states
        self.gamma = gamma

        self.pnet = nn.Sequential(  # policy_net
            nn.Linear(2, 16),  # s -> Π(a|s)
            nn.ReLU(),
            nn.Linear(16, self.action_space_size)
        )
        self.qnet = nn.Sequential(  # q_value_net
            nn.Linear(2, 16),  # s -> q[s,a]
            nn.ReLU(),
            nn.Linear(16, self.action_space_size)
        )
        self.value_optimizer = torch.optim.Adam(self.qnet.parameters(), lr=lr_critic)
        self.policy_optimizer = torch.optim.Adam(self.pnet.parameters(), lr=lr_actor)


        self.policy = np.zeros((self.state_space_size, self.action_space_size))
        self.q_value = np.zeros((self.state_space_size, self.action_space_size))

    def decode_state(self, state):
        '''
        :param state: int
        :return: 归一化后的元组
        '''
        i = state // self.env.size
        j = state % self.env.size
        return torch.tensor((i / (self.env.size - 1), j / (self.env.size - 1)), dtype=torch.float32)

    def generate_action(self, state):
        '''
        :param state: tuple
        :return: int,float
        '''
        logits = self.pnet(state)
        action_probs = torch.softmax(logits, dim=0)  # π(a|s,θ)
        action_dist = torch.distributions.Categorical(action_probs)  # 按分布采样
        action = action_dist.sample()

        log_prob = action_dist.log_prob(action)  # In π(a|s,θ)  注意传入的是索引，会自动做log(action_probs[action_index])
        return action.item(), log_prob

    def solve(self, num_episodes=200):
        for _ in range(num_episodes):
            state_int = self.env.reset()
            state = self.decode_state(state_int)

            done = False
            while not done:
                action, log_prob = self.generate_action(state)  # a_t,s_t,In π(a_t|s_t,θ)

                next_state_int, reward, done = self.env.step(state_int, action)  # s_t+1,r_t+1
                next_state = self.decode_state(next_state_int)
                if not done:
                    next_action, _ = self.generate_action(next_state)  # a_t+1
                else:
                    next_action, action_prob = None, None

                # Critic (value update)
                qvalue = self.qnet(state)[action]  # q(s_t,a_t)
                if done:
                    td_target = torch.tensor(reward, dtype=torch.float32)
                else:
                    with torch.no_grad():  # semi gradient
                        qvalue_next = self.qnet(next_state)[next_action]  # q(s_t+1,a_t+1)
                        td_target = torch.tensor(reward, dtype=torch.float32) + self.gamma * qvalue_next

                delta = td_target - qvalue  # TD error

                self.value_optimizer.zero_grad()
                critic_loss = 0.5 * delta.pow(2)
                critic_loss.backward()
                self.value_optimizer.step()

                # Actor (policy update)
                qvalue = qvalue.detach() # 避免梯度污染
                self.policy_optimizer.zero_grad()
                actor_loss = -log_prob * qvalue
                actor_loss.backward()
                self.policy_optimizer.step()

                state_int = next_state_int
                state = next_state

    def get_policy_by_policy_net(self):
        for s in range(self.state_space_size):
            if s in self.env.terminal:
                self.policy[s,4]=1
                break

            s_t = self.decode_state(s)
            logits = self.pnet(s_t)
            action_probs = torch.softmax(logits, dim=0)
            a=torch.argmax(action_probs)
            self.policy[s,a]=1

        return self.policy

    def get_policy_by_state_value_net(self):
        for s in range(self.state_space_size):
            if s in self.env.terminal:
                self.policy[s,4]=1
                break

            a = np.argmax(self.q_value[s])
            self.policy[s, a] = 1
        return self.policy

    def get_qvalues(self):
        for s in range(self.state_space_size):
            s_t = self.decode_state(s)
            logits = self.qnet(s_t).detach().numpy()  # q(s,a)表示在状态s执行动作a后，未来所有折扣回报的期望值，不要取softmax然后取最大
            self.q_value[s, :] = logits
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
    vi = QAC(env=env)

    vi.solve(num_episodes=200)

    print("\n state value: ")
    print(vi.get_qvalues())

    print("\n get policy by policy net:")
    drow_policy(vi.get_policy_by_policy_net(), env)

    print("\n get policy by state value net:")
    drow_policy(vi.get_policy_by_state_value_net(), env)