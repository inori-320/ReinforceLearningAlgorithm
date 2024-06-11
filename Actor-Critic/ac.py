import numpy as np
import torch
import torch.nn.functional as F
import utils

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


class ActorCritic:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_learning_rate, critic_learning_rate, gamma):
        self.actor = utils.PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = utils.ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), actor_learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), critic_learning_rate)
        self.gamma = gamma

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float32).to(device)
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

        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)  # 计算TD误差，目标值（新）与当前critic网络（旧）的差距
        log_probs = torch.log(self.actor(states).gather(1, actions))
        actor_loss = torch.mean(-log_probs * td_delta.detach())     # .detach()表示将loss从计算图中分离出来，防止影响Critic参数
        critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()



