import numpy as np
import torch
import torch.nn.functional as F
from utils import PolicyNet, ValueNet

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


class SAC:
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound, actor_learning_rate,
                 critic_learning_rate, alpha_learning_rate, target_entropy, tau, gamma):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.critic_1 = ValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic_2 = ValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic_1 = ValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic_2 = ValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer_1 = torch.optim.Adam(self.critic_1.parameters(), lr=critic_learning_rate)
        self.critic_optimizer_2 = torch.optim.Adam(self.critic_2.parameters(), lr=critic_learning_rate)
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float32)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_learning_rate)
        self.gamma = gamma  # 折扣系数
        self.target_entropy = target_entropy
        self.tau = tau

    def take_action(self, state):
        state = np.array(state, dtype=np.float32)
        state = torch.tensor(state).unsqueeze(0).to(device)  # 将状态转换为张量并添加批次维度
        action = self.actor(state)[0]
        return [action.item()]

    def calc_target(self, rewards, next_states, dones):
        next_actions, log_prob = self.actor(next_states)
        entropy = -log_prob
        q1_value = self.target_critic_1(next_states, next_actions)
        q2_value = self.target_critic_2(next_states, next_actions)
        next_value = torch.min(q1_value, q2_value) + self.log_alpha.exp() * entropy
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
        rewards = (rewards + 8.0) / 8.0     # 奖励重塑以便训练
        td_target = self.calc_target(rewards, next_states, dones)
        critic_loss_1 = torch.mean(F.mse_loss(self.critic_1(states, actions), td_target.detach()))
        critic_loss_2 = torch.mean(F.mse_loss(self.critic_2(states, actions), td_target.detach()))
        self.critic_optimizer_1.zero_grad()
        self.critic_optimizer_2.zero_grad()
        critic_loss_1.backward()
        critic_loss_2.backward()
        self.actor_optimizer.step()
        self.critic_optimizer_1.step()
        self.critic_optimizer_2.step()

        new_actions, log_prob = self.actor(states)
        entropy = -log_prob
        q1_value = self.critic_1(states, new_actions)
        q2_value = self.critic_2(states, new_actions)
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - torch.min(q1_value, q2_value))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = torch.mean((entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)


