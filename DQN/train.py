import random
import gym
import numpy as np
import torch
import matplotlib.pyplot as plt
import utils
from dqn import DQN

n_episode = 500
step = 1000
target_update = 10
buffer_size = 10000
mini_batch = 500
batch_size = 64
learning_rate = 2e-3
gamma = 0.98

env = gym.make("CartPole-v1")
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

epsilon_decay = 10000
epsilon_start = 1.0
epsilon_end = 0.01
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

state_dim = env.observation_space.shape[0]
hidden_dim = 128
action_dim = env.action_space.n

replay_buffer = utils.ReplayBuffer(buffer_size)
agent = DQN(state_dim, hidden_dim, action_dim, learning_rate, gamma, target_update)

reward_buffer = []

for episode_i in range(n_episode):
    state = env.reset()
    episode_reward = 0
    for step_i in range(step):
        if isinstance(state, tuple):
            state = state[0]  # 从 tuple 中提取第一个元素
        action = agent.take_action(state, episode_i, step, step_i, epsilon_decay, epsilon_start, epsilon_end)
        next_state, reward, done, done_, info = env.step(action)
        if isinstance(next_state, tuple):
            next_state = next_state[0]  # 从 tuple 中提取第一个元素
        replay_buffer.add_memo(state, action, reward, next_state, done)
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

episodes_list = list(range(len(reward_buffer)))
plt.plot(episodes_list, np.mean(reward_buffer[-100:]))
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN on {}'.format("CartPole"))
plt.show()
