import numpy as np
import torch
import torch.nn.functional as F
from utils import PolicyNetDiscrete, ValueNetDiscrete

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


class SACDiscrete:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_learning_rate,
                 critic_learning_rate, alpha_learning_rate, target_entropy, tau, gamma):
        self.actor = PolicyNetDiscrete(state_dim, hidden_dim, action_dim).to(device)
        self.critic_1 = ValueNetDiscrete(state_dim, hidden_dim, action_dim).to(device)
        self.critic_2 = ValueNetDiscrete(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic_1 = ValueNetDiscrete(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic_2 = ValueNetDiscrete(state_dim, hidden_dim, action_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer_1 = torch.optim.Adam(self.critic_1.parameters(), lr=critic_learning_rate)
        self.critic_optimizer_2 = torch.optim.Adam(self.critic_2.parameters(), lr=critic_learning_rate)
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float32)    # 可学习的熵系数，用alpha的log值,可以使训练结果比较稳定
        self.log_alpha.requires_grad = True     # 对熵值计算对数
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_learning_rate)
        self.gamma = gamma  # 折扣系数
        self.target_entropy = target_entropy    # 目标熵
        self.tau = tau      # 软更新参数

    def take_action(self, state):
        state = np.array(state, dtype=np.float32)
        state = torch.tensor(state).unsqueeze(0).to(device)  # 将状态转换为张量并添加批次维度
        probs = self.actor(state)   # 输出动作的概率分布
        action_dist = torch.distributions.Categorical(probs)    # 创建类别分布并采样
        action = action_dist.sample().item()
        return action

    def calc_target(self, rewards, next_states, dones):
        next_probs = self.actor(next_states)
        next_log_probs = torch.log(next_probs + 1e-8)
        # 对每个状态s，对所有可能的动作a的概率乘以其对数概率的乘积进行求和，得到该状态下策略的熵
        entropy = -torch.sum(next_probs * next_log_probs, dim=1, keepdim=True)
        q1_value = self.target_critic_1(next_states)
        q2_value = self.target_critic_2(next_states)
        min_q_value = torch.sum(next_probs * torch.min(q1_value, q2_value), dim=1, keepdim=True)
        next_value = min_q_value + self.log_alpha.exp() * entropy
        td_target = rewards + self.gamma * next_value * (1 - dones)
        return td_target

    def soft_update(self, net, target_net):
        for params, params_target in zip(net.parameters(), target_net.parameters()):
            params_target.data.copy_(params_target.data * (1 - self.tau) + params * self.tau)

    def update(self, transition_dict):
        states = torch.tensor(transition_dict["states"], dtype=torch.float32).to(device)
        actions = torch.tensor(transition_dict["actions"], dtype=torch.int64).view(-1, 1).to(device)
        rewards = torch.tensor(transition_dict["rewards"], dtype=torch.float32).view(-1, 1).to(device)
        next_states = torch.tensor(transition_dict["next_states"], dtype=torch.float32).to(device)
        dones = torch.tensor(transition_dict["dones"], dtype=torch.float32).view(-1, 1).to(device)

        td_target = self.calc_target(rewards, next_states, dones)
        critic_q_value1 = self.critic_1(states).gather(1, actions)
        critic_loss_1 = torch.mean(F.mse_loss(critic_q_value1, td_target.detach()))
        critic_q_value2 = self.critic_2(states).gather(1, actions)
        critic_loss_2 = torch.mean(F.mse_loss(critic_q_value2, td_target.detach()))
        self.critic_optimizer_1.zero_grad()
        self.critic_optimizer_2.zero_grad()
        critic_loss_1.backward()
        critic_loss_2.backward()
        self.actor_optimizer.step()
        self.critic_optimizer_1.step()
        self.critic_optimizer_2.step()

        probs = self.actor(states)
        log_probs = torch.log(probs + 1e-8)
        entropy = -torch.sum(probs * log_probs, dim=1, keepdim=True)
        q1_value = self.critic_1(states)
        q2_value = self.critic_2(states)
        min_q_value = torch.sum(probs * torch.min(q1_value, q2_value), dim=1, keepdim=True)
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - min_q_value)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        # 让熵值靠近目标熵值，当策略熵低于目标熵时，增加探索力度；反之则减少探索
        alpha_loss = torch.mean((entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)
