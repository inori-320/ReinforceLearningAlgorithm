import random
import gym
import numpy as np
import torch.nn as nn
import torch
from agent import Agent

env = gym.make("CartPole-v1")
state = env.reset()

EPSILON_DECAY = 10000
EPSILON_START = 1.0
EPSILON_END = 0.01
n_episode = 5000
n_time_step = 1000
n_state = len(state)
n_action = env.action_space.n

agent = Agent(n_input=n_state, n_output=n_action)

REWARD_BUFFER = np.empty(shape=n_episode)
TARGET_UPDATE_FREQUENCY = 10

for episode_i in range(n_episode):
    episode_reward = 0
    for step_i in range(n_time_step):
        epsilon = np.interp(episode_i * n_time_step + step_i, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])
        random_sample = random.random()
        if random_sample <= epsilon:
            action = env.action_space.sample()
        else:
            action = agent.q_net.act(state)
        next_state, reward, done, done_, info = env.step(action)
        agent.memo.add_memo(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward
        if done:
            state = env.reset()
            REWARD_BUFFER[episode_i] = episode_reward
            break

        if np.mean(REWARD_BUFFER[:episode_i]) >= 200:
            while True:
                action = agent.q_net.act(state)
                state, reward, done, done_, info = env.step(action)
                env.render()
                if done:
                    env.reset()

        if agent.memo.can_sample:
            batch_state, batch_action, batch_reward, batch_next_state, batch_done = agent.memo.sample()
            target_q_values = agent.target_net(batch_next_state)
            max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]
            targets = batch_reward + agent.GAMMA * (1 - batch_done) * max_target_q_values
            # 把batch个state放入qnet中，获取到batch*n个action
            q_values = agent.q_net(batch_state)
            # 把所有state对应的n个action中q值最高的拼成一个一维向量，匹配到对应的state下
            max_q_values = torch.gather(input=q_values, dim=1, index=batch_action)

            loss = nn.functional.smooth_l1_loss(targets, max_q_values)

            agent.optimizer.zero_grad()
            loss.backward()
            agent.optimizer.step()

    if episode_i % TARGET_UPDATE_FREQUENCY == 0:
        agent.target_net.load_state_dict(agent.q_net.state_dict())

        print("Episode:{}".format(episode_i))
        print("avg.Reward:{}".format(np.mean(REWARD_BUFFER[:episode_i])))
