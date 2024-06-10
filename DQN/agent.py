import random
import torch
import numpy as np
import torch.nn as nn


class Replay_memory:
    def __init__(self, n_state, n_action):
        self.n_state = n_state
        self.n_action = n_action
        self.MEMORY_SIZE = 1000
        self.BATCH_SIZE = 64
        self.idx = 0
        self.max_idx = 0
        self.is_sample = False

        self.all_state = np.empty(shape=(self.MEMORY_SIZE, self.n_state), dtype=np.float32)
        self.all_action = np.random.randint(low=0, high=n_action, size=self.MEMORY_SIZE, dtype=np.uint8)
        self.all_reward = np.empty(shape=self.MEMORY_SIZE, dtype=np.float32)
        self.all_next_state = np.random.randint(low=0, high=2, size=self.MEMORY_SIZE, dtype=np.uint8)
        self.all_done = np.empty(shape=(self.MEMORY_SIZE, self.n_state), dtype=np.float32)

    def add_memo(self, state, action, reward, next_state, done):
        print(self.all_state)
        print(state)
        self.all_state[self.idx] = state[0]
        self.all_action[self.idx] = action
        self.all_reward[self.idx] = reward
        self.all_next_state[self.idx] = next_state[0]
        self.all_done[self.idx] = done
        self.idx = (self.idx + 1) % self.MEMORY_SIZE
        self.max_idx = max(self.max_idx, self.idx + 1)
        if self.idx >= self.BATCH_SIZE:
            self.is_sample = True

    def sample(self):
        batch_state = []
        batch_action = []
        batch_reward = []
        batch_next_state = []
        batch_done = []
        idxes = random.sample(range(0, self.max_idx), self.BATCH_SIZE)
        for i in idxes:
            batch_state.append(self.all_state[i])
            batch_action.append(self.all_action[i])
            batch_reward.append(self.all_reward[i])
            batch_next_state.append(self.all_next_state[i])
            batch_done.append(self.all_done[i])

        batch_state_tensor = torch.as_tensor(np.asarray(batch_state), dtype=torch.float32)
        batch_action_tensor = torch.as_tensor(np.asarray(batch_action), dtype=torch.int64).unsqueeze(-1)
        batch_reward_tensor = torch.as_tensor(np.asarray(batch_reward), dtype=torch.float32).unsqueeze(-1)
        batch_next_state_tensor = torch.as_tensor(np.asarray(batch_next_state), dtype=torch.uint8).unsqueeze(-1)
        batch_done_tensor = torch.as_tensor(np.asarray(batch_done), dtype=torch.float32)
        return batch_state_tensor, batch_action_tensor, batch_reward_tensor, batch_next_state_tensor, batch_done_tensor

    def can_sample(self):
        return self.is_sample


class DQN_NETWORK(nn.Module):
    def __init__(self, n_input, n_output):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_input, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_output)
        )

    def forward(self):
        return self.net

    def act(self, obs):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        q_values = self(obs_tensor.unsqueeze(0))
        max_q_idx = torch.argmax(input=q_values)
        action = max_q_idx.detach().item()
        return action


class Agent:
    def __init__(self, n_input, n_output):
        self.n_input = n_input
        self.n_output = n_output

        self.GAMMA = 0.99
        self.learning_rete = 1e-3
        self.memo = Replay_memory(n_input, n_output)
        self.q_net = DQN_NETWORK(n_input, n_output)
        self.target_net = DQN_NETWORK(n_input, n_output)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.learning_rete)



