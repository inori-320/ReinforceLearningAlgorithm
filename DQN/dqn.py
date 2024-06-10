import torch
import torch.nn.functional as F
import utils
import numpy as np
import random

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


class DQN:
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma, target_update):
        self.action_dim = action_dim
        self.q_net = utils.DQN_NETWORK(state_dim, hidden_dim, action_dim).to(device)
        self.target_net = utils.DQN_NETWORK(state_dim, hidden_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.target_update = target_update
        self.count = 0

    def take_action(self, state, episode_i, step, step_i, epsilon_decay, start, end):
        epsilon = np.interp(episode_i * step + step_i, [0, epsilon_decay], [start, end])
        random_sample = random.random()
        if random_sample <= epsilon:
            return np.random.randint(self.action_dim)
        else:
            if isinstance(state, tuple):
                state = state[0]  # 从 tuple 中提取第一个元素
            state = np.array(state, dtype=np.float32)
            state = torch.tensor(state).unsqueeze(0).to(device)  # 将状态转换为张量并添加批次维度
            with torch.no_grad():
                return self.q_net(state).argmax(dim=1).item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict["states"], dtype=torch.float32).to(device)
        actions = torch.tensor(transition_dict["actions"], dtype=torch.int64).view(-1, 1).to(device)
        rewards = torch.tensor(transition_dict["rewards"], dtype=torch.float32).view(-1, 1).to(device)
        next_states = torch.tensor(transition_dict["next_states"], dtype=torch.float32).to(device)
        dones = torch.tensor(transition_dict["dones"], dtype=torch.float32).view(-1, 1).to(device)

        q_values = self.q_net(states).gather(1, actions)
        max_next_q_values = self.target_net(next_states).max(1)[0].view(-1, 1)
        q_targets = rewards + max_next_q_values * (1 - dones)

        loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        self.count += 1
