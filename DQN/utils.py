import collections
import random
import torch
import numpy as np
import torch.nn as nn

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)    # 缓冲池使用双端队列，定义最大大小，如果满了还继续添加元素的话会自动弹出首元素

    def add_memo(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)    # 从buffer中随机采样batch_size个元素
        state, action, reward, next_state, done = zip(*transitions)     # *transitions是把这个列表展开，然后使用zip函数按列归类
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)


# dqn神经网络
class DQN_NETWORK(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DQN_NETWORK, self).__init__()
        self.net = nn.Sequential(   # 两个全连接层
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)
