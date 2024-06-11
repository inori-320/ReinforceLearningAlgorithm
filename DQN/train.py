import random
import gym
import numpy as np
import torch
import matplotlib.pyplot as plt
import utils
from dqn import DQN

n_episode = 500     # 玩500局
step = 1000     # 每局玩100步
target_update = 10  # 每10次更新一次target network，即复制q network的参数直接到target network中
buffer_size = 10000     # 缓冲区大小
mini_batch = 500    # 延迟采样，只有收集到这么多数据的时候才进行采样
batch_size = 64     # 一次采样64条数据
learning_rate = 2e-3    # 学习率，梯度下降时的步长
gamma = 0.98    # 衰减因子

env = gym.make("CartPole-v1")   # 推杆子环境
random.seed(0)  # 随机种子，表示在这个种子下，随机数取值固定，每次random都是这些数
np.random.seed(0)
torch.manual_seed(0)

epsilon_decay = 10000   # epsilon从1下降到0.01所需的步长
epsilon_start = 1.0
epsilon_end = 0.01
#device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")   # 定义设备

state_dim = env.observation_space.shape[0]  # 环境一开始的状态维度
hidden_dim = 128    # 隐藏层维度
action_dim = env.action_space.n     # 环境一开始的动作维度

replay_buffer = utils.ReplayBuffer(buffer_size)     # 初始化缓冲区
agent = DQN(state_dim, hidden_dim, action_dim, learning_rate, gamma, target_update)

reward_buffer = []  # 记录积累奖励

for episode_i in range(n_episode):
    state = env.reset()     # 重置状态
    episode_reward = 0
    for step_i in range(step):
        if isinstance(state, tuple):    # 如果不加这段代码，state后面会多一条{}
            state = state[0]  # 从 tuple 中提取第一个元素
        action = agent.take_action(state, episode_i, step, step_i, epsilon_decay, epsilon_start, epsilon_end)
        next_state, reward, done, done_, info = env.step(action)
        if isinstance(next_state, tuple):
            next_state = next_state[0]  # 从 tuple 中提取第一个元素
        replay_buffer.add_memo(state, action, reward, next_state, done)     # 把这次的信息添加到缓冲池中
        state = next_state
        episode_reward += reward

        if done:
            reward_buffer.append(episode_reward)
            break

        if replay_buffer.size() > mini_batch:
            batch_state, batch_action, batch_reward, batch_next_state, batch_done = replay_buffer.sample(batch_size)
            transition_dict = {
                'states': batch_state,
                'actions': batch_action,
                'next_states': batch_next_state,
                'rewards': batch_reward,
                'dones': batch_done
            }
            agent.update(transition_dict)

    print(f"Episode: {episode_i}, avg.Reward: {np.mean(reward_buffer[-100:])}")

# 画出奖励轨迹
episodes_list = list(range(len(reward_buffer)))
plt.plot(episodes_list, np.mean(reward_buffer[-100:]))
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN on {}'.format("CartPole"))
plt.show()
