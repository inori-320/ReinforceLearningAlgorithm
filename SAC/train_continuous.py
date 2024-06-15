import torch
import gym
import numpy as np
import matplotlib.pyplot as plt
from sac_continuous import SACContinuous
from utils import ReplayBuffer

env_name = 'Pendulum-v1'
env = gym.make(env_name)
torch.manual_seed(0)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high[0]  # env动作的最大值

n_episodes = 1000
step = 2000
actor_learning_rate = 3e-4
critic_learning_rate = 3e-3
alpha_learning_rate = 3e-4
hidden_dim = 128
gamma = 0.99
tau = 0.005
buffer_size = 100000
mini_batch = 1000
batch_size = 64
target_entropy = -env.action_space.shape[0]

agent = SACContinuous(state_dim, hidden_dim, action_dim, action_bound, actor_learning_rate, critic_learning_rate,
                      alpha_learning_rate, target_entropy, tau, gamma)

replay_buffer = ReplayBuffer(buffer_size)
reward_buffer = []  # 记录积累奖励
reward_mean_buffer = []

for episode_i in range(n_episodes):
    state = env.reset()
    episode_reward = 0
    for step_i in range(step):
        if isinstance(state, tuple):
            state = state[0]
        action = agent.take_action(state)
        next_state, reward, done, truncated, _ = env.step(action)
        if isinstance(next_state, tuple):
            next_state = next_state[0]
        replay_buffer.add_memp(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward
        if done or truncated:
            reward_buffer.append(episode_reward)
            break
        if replay_buffer.size() >= mini_batch:  # 只有当前收集到了足够多的数据，才开始采样
            b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
            transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
            agent.update(transition_dict)

    reward_mean_buffer.append(np.mean(reward_buffer[-100:]))
    print(f"Episode: {episode_i}, avg.Reward: {np.mean(reward_buffer[-100:])}, Reward: {episode_reward}")

# 画出奖励轨迹
episodes_list = list(range(len(reward_buffer)))
plt.plot(episodes_list, reward_mean_buffer)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('SAC on {}'.format('Pendulum-v1'))
plt.show()
