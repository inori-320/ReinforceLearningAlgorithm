from utils import PolicyNet, ValueNet
import torch
import numpy as np
import torch.nn.functional as F

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


class TD3:
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound, sigma,
                 actor_learning_rate, critic_learning_rate, tau, gamma, policy_delay, noise_clip, policy_noise):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.critic_1 = ValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic_2 = ValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.target_critic_1 = ValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic_2 = ValueNet(state_dim, hidden_dim, action_dim).to(device)
        # 把actor和critic网络的参数复制到target网络中
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        # 定义优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer_1 = torch.optim.Adam(self.critic_1.parameters(), lr=critic_learning_rate)
        self.critic_optimizer_2 = torch.optim.Adam(self.critic_2.parameters(), lr=critic_learning_rate)

        self.gamma = gamma  # TD折扣系数
        self.sigma = sigma  # 动作噪声标准差
        self.tau = tau  # 软更新参数
        self.action_bound = action_bound    # 动作范围
        self.policy_delay = policy_delay    # 延迟策略更新频率
        self.noise_clip = noise_clip    # 噪声裁剪值，添加噪声后对其进行裁剪，确保添加的噪声不会超出一定的范围
        self.policy_noise = policy_noise    # 策略噪声标准差
        self.total_iter = 0  # 迭代次数
        self.action_dim = action_dim

    def take_action(self, state):
        state = np.array(state, dtype=np.float32)
        state = torch.tensor(state).unsqueeze(0).to(device)
        action = self.actor(state).detach().cpu().numpy().flatten()     # 使用 Actor 网络生成一个动作并添加高斯噪声
        action += self.sigma * np.random.randn(self.action_dim)
        return np.clip(action, -self.action_bound, self.action_bound)

    # 软更新
    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_((1 - self.tau) * param_target.data + self.tau * param.data)

    def update(self, transition_dict):
        self.total_iter += 1
        states = torch.tensor(transition_dict["states"], dtype=torch.float32).to(device)
        actions = torch.tensor(transition_dict["actions"], dtype=torch.float32).to(device)
        rewards = torch.tensor(transition_dict["rewards"], dtype=torch.float32).view(-1, 1).to(device)
        next_states = torch.tensor(transition_dict["next_states"], dtype=torch.float32).to(device)
        dones = torch.tensor(transition_dict["dones"], dtype=torch.float32).view(-1, 1).to(device)

        with torch.no_grad():   # 禁用梯度计算
            # 生成策略噪声并进行裁剪
            # randn_like(actions)表示生成一个与actions张量形状相同的标准正态分布噪声张量，然后经过policy_noise调整后裁剪到noise_clip范围内
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            # 借用double DQN的思想，使用target_actor计算next_action，并添加噪声
            next_actions = (self.target_actor(next_states) + noise).clamp(-self.action_bound, self.action_bound)
            target_q1 = self.target_critic_1(next_states, next_actions)
            target_q2 = self.target_critic_2(next_states, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * torch.min(target_q1, target_q2)

        current_q1 = self.critic_1(states, actions)
        current_q2 = self.critic_2(states, actions)
        critic_loss_1 = F.mse_loss(current_q1, target_q)
        critic_loss_2 = F.mse_loss(current_q2, target_q)

        self.critic_optimizer_1.zero_grad()
        self.critic_optimizer_2.zero_grad()
        critic_loss_1.backward()
        critic_loss_2.backward()
        self.critic_optimizer_1.step()
        self.critic_optimizer_2.step()

        # 延迟更新actor网络和target网络
        if self.total_iter % self.policy_delay == 0:
            # 利用两个critic网络其中之一来计算状态动作对的评估值，并且这里希望actor朝向最大值前进，所以对-actor_loss求梯度下降
            actor_loss = -self.critic_1(states, self.actor(states)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.soft_update(self.actor, self.target_actor)
            self.soft_update(self.critic_1, self.target_critic_1)
            self.soft_update(self.critic_2, self.target_critic_2)
