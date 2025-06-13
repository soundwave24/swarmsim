import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from pettingzoo.utils import parallel_to_aec
from gym import spaces
import random
from torch.distributions import Categorical

# --- CooperativeEscapeEnv (기존 환경 정의) ---
class CooperativeEscapeEnv(wrappers.BaseWrapper):
    def __init__(self, grid_size=8):
        class _InnerEnv(AECEnv):
            metadata = {"render_modes": ["human"], "name": "cooperative_escape_v0"}

            def __init__(self, grid_size=8):
                super().__init__()
                self.grid_size = grid_size
                self.agent_iter = [f"agent_{i}" for i in range(3)]
                self.possible_agents = self.agent_iter[:]
                self.agent_order = list(self.possible_agents)
                self.agent_selector = agent_selector(self.agent_order)
                self.observation_space = {
                    agent: spaces.Box(low=0, high=1, shape=(grid_size, grid_size, 3), dtype=np.float32)
                    for agent in self.agent_iter
                }
                self.action_spaces = {
                    agent: spaces.Discrete(5) for agent in self.agent_iter
                }

                self.goal = (grid_size // 2, grid_size // 8)
                self.hazards = [(2, 2), (5, 2), (3, 4), (4, 5)]
                self.enemies = [(2, 5), (4, 5), (6, 5)]
                self.agent_positions = [(2, 7), (4, 7), (6, 7)]

            def reset(self, seed=None, options=None):
                self.agents = self.agent_iter[:]
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
                for x, y in self.hazards:
                    obs[y][x] = [0.5, 0.5, 0.5]
                for x, y in self.enemies:
                    obs[y][x] = [1.0, 0.0, 0.0]
                gx, gy = self.goal
                obs[gy][gx] = [1.0, 0.5, 0.0]
                ax, ay = self.agent_positions[idx]
                obs[ay][ax] = [0.0, 0.5, 1.0]
                return obs

            def step(self, action):
                agent = self.agent_selection
                idx = self.agents.index(agent)
                if self.dones[agent]:
                    self._was_dead_step(action)
                    return

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
                self.agent_selection = self._agent_selector.next()

        env = _InnerEnv(grid_size=grid_size)
        super().__init__(env)


# --- Shared Policy Network ---
class ActorCritic(nn.Module):
    def __init__(self, input_shape, action_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(input_shape), 256),
            nn.ReLU(),
        )
        self.policy = nn.Linear(256, action_dim)
        self.value = nn.Linear(256, 1)

    def forward(self, x):
        x = self.shared(x)
        return self.policy(x), self.value(x)

    def get_action_and_value(self, x):
        logits, value = self.forward(x)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value


# --- MAPPO 학습 루프 ---
def train_mappo(env_fn, num_episodes=1000, gamma=0.99, clip_eps=0.2, learning_rate=3e-4):
    env = parallel_to_aec(env_fn())
    obs_space = env.observation_space[env.agent_iter[0]]
    act_space = env.action_spaces[env.agent_iter[0]]

    model = ActorCritic(obs_space.shape, act_space.n)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for episode in range(num_episodes):
        obs_dict = env.reset()
        log_probs = {agent: [] for agent in env.agents}
        values = {agent: [] for agent in env.agents}
        rewards = {agent: [] for agent in env.agents}
        entropies = {agent: [] for agent in env.agents}
        actions = {agent: [] for agent in env.agents}
        observations = {agent: [] for agent in env.agents}

        done = {agent: False for agent in env.agents}

        while not all(done.values()):
            for agent in env.agent_iter:
                if done[agent]:
                    continue

                obs = torch.tensor(obs_dict[agent], dtype=torch.float32).unsqueeze(0)
                action, log_prob, entropy, value = model.get_action_and_value(obs)

                obs_dict, reward, termination, truncation, _ = env.last()
                env.step(action.item())

                observations[agent].append(obs)
                actions[agent].append(action)
                log_probs[agent].append(log_prob)
                values[agent].append(value)
                rewards[agent].append(torch.tensor([reward], dtype=torch.float32))
                entropies[agent].append(entropy)

                done[agent] = termination or truncation

        # 학습 - 에이전트별 Advantage 계산
        for agent in env.agent_iter:
            R = 0
            returns = []
            for r in reversed(rewards[agent]):
                R = r + gamma * R
                returns.insert(0, R)

            returns = torch.cat(returns)
            values_tensor = torch.cat(values[agent]).squeeze()
            log_probs_tensor = torch.cat(log_probs[agent])
            advantage = returns - values_tensor

            policy_loss = -(log_probs_tensor * advantage.detach()).mean()
            value_loss = advantage.pow(2).mean()
            entropy_loss = -torch.cat(entropies[agent]).mean()

            loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if episode % 10 == 0:
            print(f"[Episode {episode}] Avg reward: {np.mean([r.sum().item() for r in rewards.values()]):.2f}")

# 실행
train_mappo(lambda: CooperativeEscapeEnv(), num_episodes=500)
# --- 테스트 ---
def test_mappo(env_fn, num_episodes=10):
    env = parallel_to_aec(env_fn())
    obs_dict = env.reset()

    for episode in range(num_episodes):
        done = {agent: False for agent in env.agents}
        total_reward = 0

        while not all(done.values()):
            for agent in env.agent_iter:
                if done[agent]:
                    continue

                obs = torch.tensor(obs_dict[agent], dtype=torch.float32).unsqueeze(0)
                action, _, _, _ = model.get_action_and_value(obs)
                obs_dict, reward, termination, truncation, _ = env.last()
                env.step(action.item())

                total_reward += reward
                done[agent] = termination or truncation

        print(f"[Episode {episode}] Total reward: {total_reward:.2f}")
# 실행
# test_mappo(lambda: CooperativeEscapeEnv(), num_episodes=5)
