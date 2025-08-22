import itertools

import gym
import numpy as np
import torch

from envs.base_env import BaseEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NavigationEnv(BaseEnv):
    def __init__(self, radius, goal, horizon, dense_reward):
        self.radius = radius
        self.goal = np.array(goal)
        self.horizon = horizon
        self.dt = 0.1
        self.goal_tolerance = 0.2
        self.dense_reward = dense_reward
        self.state_dim = 2
        self.action_dim = 2
        self.observation_space = gym.spaces.Box(
            low=-self.radius, high=self.radius, shape=(self.state_dim,)
        )
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.action_dim,))

    def sample_state(self):
        return self.observation_space.sample()

    def sample_action(self):
        return self.action_space.sample()

    def reset(self):
        self.current_step = 0
        self.state = np.array([0, 0])
        return self.state

    def transit(self, state, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        next_state = state + action * self.dt
        next_state = np.clip(next_state, -self.radius, self.radius)
        dist = np.linalg.norm(next_state - self.goal)
        if self.dense_reward:
            reward = np.exp(-dist)
        else:
            reward = dist < self.goal_tolerance
        return next_state, float(reward)

    def step(self, action):
        if self.current_step >= self.horizon:
            raise ValueError("Episode has already ended")

        self.state, r = self.transit(self.state, action)
        self.current_step += 1
        done = self.current_step >= self.horizon
        return self.state.copy(), r, done, {}

    def get_obs(self):
        return self.state.copy()

    def opt_action(self, state):
        diff = self.goal - state
        action = np.clip(diff / self.dt, self.action_space.low, self.action_space.high)
        return action


class NavigationVecEnv(BaseEnv):
    """
    Vectorized Darkroom environment.
    """

    def __init__(self, envs):
        self._goals = np.array([env.goal for env in envs])
        self._num_envs = len(envs)
        self._envs = envs
        self.dt = envs[0].dt
        self.goal_tolerance = envs[0].goal_tolerance
        self.radius = envs[0].radius
        self.horizon = envs[0].horizon
        self.state_dim = envs[0].state_dim
        self.action_dim = envs[0].action_dim
        self.observation_space = gym.spaces.Box(
            low=-self.radius, high=self.radius, shape=(self.state_dim,)
        )
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.action_dim,))

    def reset(self):
        self.current_step = np.zeros(self._num_envs, dtype=int)
        self.states = np.zeros((self._num_envs, 2), dtype=float)
        return self.states.copy()

    def step(self, actions):
        if np.any(self.current_step >= self._envs[0].horizon):
            raise ValueError("Episode has already ended for some environments")

        self.states, r = self.transit(self.states, actions)
        self.current_step += 1
        dones = self.current_step >= self._envs[0].horizon
        return self.states.copy(), r, dones, {}

    @property
    def num_envs(self):
        return self._num_envs

    @property
    def envs(self):
        return self._envs

    @property
    def state_dim(self):
        return self._envs[0].state_dim

    @property
    def action_dim(self):
        return self._envs[0].action_dim

    def opt_action(self, states):
        actions = []
        for env, state in zip(self._envs, states):
            action = env.opt_action(state)
            actions.append(action)
        return np.array(actions)

    def transit(self, states, actions):
        actions = np.clip(actions, self.action_space.low, self.action_space.high)
        next_states = states + actions * self.dt
        next_states = np.clip(next_states, -self.radius, self.radius)
        dists = np.linalg.norm(next_states - self._goals, axis=1)
        if self.dense_reward:
            rewards = np.exp(-dists)
        else:
            rewards = dists < self.goal_tolerance
        return next_states, rewards.astype(float)

    def deploy(self, ctrl):
        ob = self.reset()
        obs = []
        acts = []
        next_obs = []
        rews = []
        done = False

        while not done:
            act = ctrl.act(ob)

            obs.append(ob)
            acts.append(act)

            ob, rew, done, _ = self.step(act)
            done = all(done)

            rews.append(rew)
            next_obs.append(ob)

        obs = np.stack(obs, axis=1)
        acts = np.stack(acts, axis=1)
        next_obs = np.stack(next_obs, axis=1)
        rews = np.stack(rews, axis=1)
        return obs, acts, next_obs, rews
