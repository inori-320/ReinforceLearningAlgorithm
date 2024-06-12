import numpy as np
import torch
import torch.nn.functional as F
from utils import PolicyNet, ValueNet

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


class PPO:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_learning_rate,
                 critic_learning_rate, lmbda, epochs, eps, gamma):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_learning_rate)
        self.gamma = gamma  # 折扣系数
        self.lmbda = lmbda  # GAE优势函数的缩放因子
        self.epochs = epochs    # 一条序列的数据用来训练多少轮
        self.eps = eps  # 截断范围，PPO-clip目标中的截断范围参数，用于限制新旧策略的概率比值，控制策略更新的幅度

    def take_action(self, state):
        state = np.array(state, dtype=np.float32)
        state = torch.tensor(state).unsqueeze(0).to(device)  # 将状态转换为张量并添加批次维度
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample().item()
        return action

    def update(self, transition_dict):
        states = torch.tensor(transition_dict["states"], dtype=torch.float32).to(device)
        actions = torch.tensor(transition_dict["actions"], dtype=torch.int64).view(-1, 1).to(device)
        rewards = torch.tensor(transition_dict["rewards"], dtype=torch.float32).view(-1, 1).to(device)
        next_states = torch.tensor(transition_dict["next_states"], dtype=torch.float32).to(device)
        dones = torch.tensor(transition_dict["dones"], dtype=torch.float32).view(-1, 1).to(device)

        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)   # 当前时刻的state_value
        td_delta = td_target - self.critic(states)
        td_delta = td_delta.detach().cpu().numpy()
        # 对时序差分结果计算GAE优势函数
        advantage_list = []
        advantage = 0.0
        # 逆序遍历时序差分结果，把最后一时刻的放前面
        for delta in td_delta[::-1]:
            advantage = self.gamma * self.lmbda * advantage + delta
            advantage_list.append(advantage)
        advantage_list.reverse()
        advantage = torch.tensor(advantage_list, dtype=torch.float32).to(device)
        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()

        # 一个序列训练epochs次，批量更新，在收集一定量的经验后，使用这些经验数据多次更新策略网络和价值网络
        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions))    # 当前策略在t时刻智能体处于状态s所采取的行为概率
            ratio = torch.exp(log_probs - old_log_probs)    # 计算概率的比值来控制新策略更新幅度
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            actor_loss = torch.mean(-torch.min(surr1, surr2))   # 策略网络的损失PPO-clip
            # 价值网络的当前时刻预测值，与目标价值网络当前时刻的state_value之差
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

