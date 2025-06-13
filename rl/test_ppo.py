import numpy as np
import gym
import time
from gym import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
import pygame

class CooperativeEscapeEnv(AECEnv):
    metadata = {"render_modes": ["human"], "name": "cooperative_escape_v0"}

    def __init__(self, grid_size=8):
        super().__init__()
        self.grid_size = grid_size
        # self.total_agents = 3
        

        
        self.agent_names = [f"agent_{i}" for i in range(3)]
        self.possible_agents = self.agent_names[:]
        self.agent_order = list(self.possible_agents)
        self.agent_selector = agent_selector(self.agent_order)
        self.observation_spaces = {
            agent: spaces.Box(low=0, high=1, shape=(grid_size, grid_size, 3), dtype=np.float32)
            for agent in self.agent_names
        }
        self.action_spaces = {
            agent: spaces.Discrete(5)  # 0: stay, 1: up, 2: down, 3: left, 4: right
            for agent in self.agent_names
        }

        self.goal = (grid_size // 2, grid_size // 8)  # top middle
        self.hazards = [(2, 2), (5, 2), (3, 4), (4, 5)]
        self.enemies = [(2, 5), (4, 5), (6, 5)]
        self.agent_positions = [(2, 7), (4, 7), (6, 7)]

    def reset(self, seed=None, options=None):
        self.agents = self.agent_names[:]
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.agent_selector = agent_selector(self.agent_order)
        self.agent_positions = [(2, 7), (4, 7), (6, 7)]
        self.enemies = [(2, 5), (4, 5), (6, 5)]
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()
        
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        
        return self._observe()

    def _observe(self):
        return {
            agent: self._get_obs(agent_idx)
            for agent_idx, agent in enumerate(self.agents)
        }

    def _get_obs(self, idx):
        obs = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.float32)

        # Add hazards
        for x, y in self.hazards:
            obs[y][x] = [0.5, 0.5, 0.5]  # gray

        # Add enemies
        for x, y in self.enemies:
            obs[y][x] = [1.0, 0.0, 0.0]  # red

        # Add goal
        gx, gy = self.goal
        obs[gy][gx] = [1.0, 0.5, 0.0]  # orange

        # Add agent
        ax, ay = self.agent_positions[idx]
        obs[ay][ax] = [0.0, 0.5, 1.0]  # blue

        return obs

    def step(self, action):
        agent = self.agent_selection
        idx = self.agents.index(agent)

        if self.dones[agent]:
            self._was_dead_step(action)
            return
        
        # if self.terminations[agent] or self.truncations[agent]:
        #     self._was_dead_step(None)
        #     continue

        x, y = self.agent_positions[idx]
        dx, dy = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)][action]
        nx, ny = np.clip(x + dx, 0, self.grid_size - 1), np.clip(y + dy, 0, self.grid_size - 1)
        self.agent_positions[idx] = (nx, ny)

        reward = 0.0

        if (nx, ny) in self.hazards:
            self.dones[agent] = True
            reward = -5.0
        elif (nx, ny) in self.enemies:
            self.enemies.remove((nx, ny))
            reward = 2.0
        elif (nx, ny) == self.goal:
            reward = 10.0
            self.dones[agent] = True

        self.rewards[agent] = reward

        # 다음 에이전트로 이동
        self.agent_selection = self._agent_selector.next()

    def render(self, mode="human"):
        scale = 60
        pygame.init()
        screen = pygame.display.set_mode((self.grid_size * scale, self.grid_size * scale))
        screen.fill((255, 255, 255))

        def draw_cell(x, y, color):
            rect = pygame.Rect(x * scale, y * scale, scale, scale)
            pygame.draw.rect(screen, color, rect)

        for x, y in self.hazards:
            draw_cell(x, y, (160, 160, 160))
        for x, y in self.enemies:
            draw_cell(x, y, (255, 0, 0))
        for idx, (x, y) in enumerate(self.agent_positions):
            pygame.draw.polygon(screen, (0, 120, 255), [
                (x * scale + scale//2, y * scale + 5),
                (x * scale + 5, y * scale + scale - 5),
                (x * scale + scale - 5, y * scale + scale - 5)
            ])
        gx, gy = self.goal
        pygame.draw.circle(screen, (255, 165, 0), (gx * scale + scale//2, gy * scale + scale//2), scale//3)

        pygame.display.flip()


env = CooperativeEscapeEnv()
obs = env.reset()
done = False

for _ in range(100):
    if all(env.dones.values()):
        break

    agent = env.agent_selection
    if env.dones[agent]:
        env.step(None)
        continue

    action = env.action_spaces[agent].sample()  # 또는 정책 출력
    env.step(action)
    env.render()
    time.sleep(1.0)