import numpy as np
import gym
from gym import spaces
import random

class GridWorldEnv(gym.Env):
    def __init__(self):
        super(GridWorldEnv, self).__init__()
        self.grid_size = 10
        self.obstacles = [(1, 1), (2, 1), (2, 2)]
        self.goal = (3, 3)
        self.start = (0, 0)
        self.agent_pos = list(self.start)

        self.action_space = spaces.Discrete(4)  # 0:UP, 1:DOWN, 2:LEFT, 3:RIGHT
        self.observation_space = spaces.Discrete(self.grid_size * self.grid_size)

    def reset(self):
        self.agent_pos = list(self.start)
        return self._get_state()

    def step(self, action):
        x, y = self.agent_pos

        if action == 0:  # UP
            x = max(x - 1, 0)
        elif action == 1:  # DOWN
            x = min(x + 1, self.grid_size - 1)
        elif action == 2:  # LEFT
            y = max(y - 1, 0)
        elif action == 3:  # RIGHT
            y = min(y + 1, self.grid_size - 1)

        if (x, y) not in self.obstacles:
            self.agent_pos = [x, y]

        done = (tuple(self.agent_pos) == self.goal)
        reward = 1 if done else -0.1

        return self._get_state(), reward, done, {}

    def render(self, mode='human'):
        grid = np.full((self.grid_size, self.grid_size), '_')
        for ox, oy in self.obstacles:
            grid[ox, oy] = 'X'
        gx, gy = self.goal
        grid[gx, gy] = 'G'
        ax, ay = self.agent_pos
        grid[ax, ay] = 'A'
        print("\n".join([" ".join(row) for row in grid]))
        print()

    def _get_state(self):
        return self.agent_pos[0] * self.grid_size + self.agent_pos[1]

import matplotlib.pyplot as plt

env = GridWorldEnv()

q_table = np.zeros((env.observation_space.n, env.action_space.n))
alpha = 0.1       # 학습률
gamma = 0.99      # 할인율
epsilon = 0.2     # 탐험률
episodes = 500

reward_list = []

for ep in range(episodes):
    state = env.reset()
    total_reward = 0

    for _ in range(100):  # 한 에피소드당 최대 100 스텝
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        next_state, reward, done, _ = env.step(action)

        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])

        new_value = old_value + alpha * (reward + gamma * next_max - old_value)
        q_table[state, action] = new_value

        state = next_state
        total_reward += reward

        if done:
            break

    reward_list.append(total_reward)

print("학습 완료")
env.render()

# 보상 시각화
plt.plot(reward_list)
plt.title("Episode reward")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid(True)
plt.show()
