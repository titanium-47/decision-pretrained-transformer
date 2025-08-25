import itertools

import gym
import numpy as np
import torch

from envs.base_env import BaseEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DarkroomEnv(BaseEnv):
    def __init__(self, dim, goal, horizon):
        self.dim = dim
        self.goal = np.array(goal)
        self.horizon = horizon
        self.state_dim = 2
        self.action_dim = 5
        self.observation_space = gym.spaces.Box(
            low=0, high=dim - 1, shape=(self.state_dim,)
        )
        self.action_space = gym.spaces.Discrete(self.action_dim)

    def sample_state(self):
        return np.random.randint(0, self.dim, 2).astype(float)

    def sample_action(self):
        i = np.random.randint(0, 5)
        a = np.zeros(self.action_space.n)
        a[i] = 1
        return a

    def reset(self):
        self.current_step = 0
        self.state = np.array([0, 0])
        return self.state

    def transit(self, state, action):
        action = np.argmax(action)
        assert action in np.arange(self.action_space.n)
        state = np.array(state)
        if action == 0:
            state[0] += 1
        elif action == 1:
            state[0] -= 1
        elif action == 2:
            state[1] += 1
        elif action == 3:
            state[1] -= 1
        state = np.clip(state, 0, self.dim - 1)

        if np.all(state == self.goal):
            reward = 1
        else:
            reward = 0
        return state, float(reward)

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
        if state[0] < self.goal[0]:
            action = 0
        elif state[0] > self.goal[0]:
            action = 1
        elif state[1] < self.goal[1]:
            action = 2
        elif state[1] > self.goal[1]:
            action = 3
        else:
            action = 4
        zeros = np.zeros(self.action_space.n)
        zeros[action] = 1
        return zeros


class DarkroomEnvVec(BaseEnv):
    """
    Vectorized Darkroom environment.
    """

    def __init__(self, envs):
        # self._envs = envs
        # self._num_envs = len(envs)
        self._goals = np.array([env.goal for env in envs])
        self._num_envs = len(envs)
        self._envs = envs
        self.horizon = envs[0].horizon

    def reset(self):
        # return [env.reset() for env in self._envs]
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
    
    def sample_state(self):
        return np.random.randint(0, self._envs[0].dim, (self._num_envs, 2)).astype(float)
    
    def sample_action(self):
        actions = np.zeros((self._num_envs, self.action_dim))
        actions[np.arange(self._num_envs), np.random.randint(0, self.action_dim, self._num_envs)] = 1
        return actions

    def opt_action(self, states):
        actions = []
        for env, state in zip(self._envs, states):
            action = env.opt_action(state)
            actions.append(action)
        return np.array(actions)

    def transit(self, states, actions):
        actions = np.argmax(actions, axis=1)
        assert actions.shape == (self._num_envs,)

        states[:, 0] += (actions == 0).astype(float)  # move right
        states[:, 0] -= (actions == 1).astype(float)  # move
        states[:, 1] += (actions == 2).astype(float)  # move down
        states[:, 1] -= (actions == 3).astype(float)  # move
        states = np.clip(states, 0, self._envs[0].dim - 1)
        rewards = np.linalg.norm(states - self._goals, axis=1) < 1e-5
        return states, rewards.astype(float)

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
