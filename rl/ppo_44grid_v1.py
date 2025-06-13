import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import matplotlib.pyplot as plt


class GridWorldEnv(gym.Env):
    def __init__(self):
        super(GridWorldEnv, self).__init__()
        self.grid_size = 10
        self.obstacles = [(1, 1), (2, 1), (2, 2)]
        self.goal = (3, 3)
        self.start = (0, 0)
        self.agent_pos = list(self.start)

        self.action_space = spaces.Discrete(4)  # 0:UP, 1:DOWN, 2:LEFT, 3:RIGHT
        self.observation_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = list(self.start)
        return self._get_state(), {}  # obs, info

    def step(self, action):
        x, y = self.agent_pos

        if action == 0:
            x = max(x - 1, 0)
        elif action == 1:
            x = min(x + 1, self.grid_size - 1)
        elif action == 2:
            y = max(y - 1, 0)
        elif action == 3:
            y = min(y + 1, self.grid_size - 1)

        if (x, y) not in self.obstacles:
            self.agent_pos = [x, y]

        terminated = (tuple(self.agent_pos) == self.goal)
        truncated = False
        reward = 1.0 if terminated else -0.1

        return self._get_state(), reward, terminated, truncated, {}  # obs, reward, terminated, truncated, info

    def render(self):
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
        return np.array(self.agent_pos, dtype=np.float32) / (self.grid_size - 1)


# 학습 및 테스트
env = GridWorldEnv()
check_env(env, warn=True)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)

# 테스트
total_rewards = []
for _ in range(10):
    obs, _ = env.reset()
    total_reward = 0
    for _ in range(100):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    total_rewards.append(total_reward)
    env.render()

# 보상 시각화
plt.plot(total_rewards)
plt.title("Evaluation Episode Rewards")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid(True)
plt.show()
