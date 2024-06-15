from utils import PolicyNet, ValueNet
import torch
import numpy as np
import torch.nn.functional as F

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


class DDPG:
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound, sigma,
                 actor_learning_rate, critic_learning_rate, tau, gamma):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.critic = ValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.target_critic = ValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_actor.load_state_dict(self.actor.state_dict())      # 参数复制，把actor的参数复制到target_actor中
        self.target_critic.load_state_dict(self.critic.state_dict())    # 同上
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_learning_rate)
        self.gamma = gamma  # TD的衰减因子
        self.sigma = sigma  # 高斯噪声的标准差,均值直接设为0
        self.tau = tau  # 软更新的参数
        self.action_dim = action_dim

    def take_action(self, state):
        state = np.array(state, dtype=np.float32)
        state = torch.tensor(state).unsqueeze(0).to(device)  # 将状态转换为张量并添加批次维度
        action = self.actor(state).item()
        action = action + self.sigma * np.random.randn(self.action_dim)     # 给动作添加噪声，增加探索
        return action

    def soft_update(self, net, target_net):     # 软更新，把net参数的tau%更新给target_net，(1-tau)%不变
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_((1 - self.tau) * param_target.data + self.tau * param.data)

    def update(self, transition_dict):
        states = torch.tensor(transition_dict["states"], dtype=torch.float32).to(device)
        actions = torch.tensor(transition_dict["actions"], dtype=torch.int64).view(-1, 1).to(device)
        rewards = torch.tensor(transition_dict["rewards"], dtype=torch.float32).view(-1, 1).to(device)
        next_states = torch.tensor(transition_dict["next_states"], dtype=torch.float32).to(device)
        dones = torch.tensor(transition_dict["dones"], dtype=torch.float32).view(-1, 1).to(device)

        next_q_values = self.target_critic(next_states, self.target_actor(next_states))  # 下一状态执行该动作后的预期回报
        q_targets = rewards + self.gamma + next_q_values + (1 - dones)
        critic_loss = torch.mean(F.mse_loss(self.critic(states, actions), q_targets))   # 用q_targets更新critic
        # 用critic的打分更新actor，由于策略网络的目标是最大化Q值，所以通过最小化负Q值来实现这个目标
        actor_loss = -torch.mean(self.critic(states, self.actor(states)))
        self.critic_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()
        critic_loss.backward()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        self.soft_update(self.actor, self.target_actor)  # 软更新
        self.soft_update(self.critic, self.target_critic)










