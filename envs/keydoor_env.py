import itertools

import gym
import numpy as np
import torch

from envs.base_env import BaseEnv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class KeyDoorEnv(BaseEnv):
    def __init__(self, dim, key, door, horizon, markovian):
        self.dim = dim
        self.key = np.array(key)
        self.door = np.array(door)
        self.horizon = horizon
        self.state_dim = 2
        self.action_dim = 5
        self.observation_space = gym.spaces.Box(
            low=0, high=dim - 1, shape=(self.state_dim,))
        self.action_space = gym.spaces.Discrete(self.action_dim)
        self.markovian = markovian

    def sample_state(self):
        return np.random.randint(0, self.dim, 2)

    def sample_action(self):
        i = np.random.randint(0, 5)
        a = np.zeros(self.action_space.n)
        a[i] = 1
        return a

    def reset(self):
        self.current_step = 0
        self.state = np.array([0, 0])
        self.have_key = False
        self.have_door = False
        return self.state
    
    def sample_flag(self, state):
        at_key = np.all(state == self.key)
        have_key = at_key or np.random.rand() < 0.5
        at_door = np.all(state == self.door)
        have_door = have_key and at_door
        return have_key, have_door
    
    def transit(self, state, action, have_key, have_door):
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

        reward = 0
        at_key = np.all(state == self.key) and not have_key
        have_key = at_key or have_key
        at_door = have_key and np.all(state == self.door) and not have_door
        have_door = at_door or have_door
        if self.markovian:
            reward = float(have_door) + float(have_key)
        else:
            reward = float(at_door) + float(at_key)

        return state, reward, have_key, have_door

    def step(self, action):
        if self.current_step >= self.horizon:
            raise ValueError("Episode has already ended")

        self.state, r, self.have_key, self.have_door = self.transit(self.state, action, self.have_key, self.have_door)
        self.current_step += 1
        done = (self.current_step >= self.horizon)
        return self.state.copy(), r, done, {}

    def get_obs(self):
        return self.state.copy()

    def opt_action(self, state, have_key):
        goal = self.door if have_key else self.key
        if state[0] < goal[0]:
            action = 0
        elif state[0] > goal[0]:
            action = 1
        elif state[1] < goal[1]:
            action = 2
        elif state[1] > goal[1]:
            action = 3
        else:
            action = 4
        zeros = np.zeros(self.action_space.n)
        zeros[action] = 1
        return zeros

class KeyDoorVecEnv(BaseEnv):
    """
    Vectorized KeyDoor environment.
    """

    def __init__(self, envs):
        self._doors = np.array([env.door for env in envs])
        self._keys = np.array([env.key for env in envs])
        self._horizon = envs[0].horizon
        self._num_envs = len(envs)
        self._envs = envs
        self.markovian = envs[0].markovian

    def reset(self):
        self.current_step = np.zeros(self._num_envs, dtype=int)
        self.have_keys = np.zeros(self._num_envs, dtype=bool)
        self.have_doors = np.zeros(self._num_envs, dtype=bool)
        self.states = np.zeros((self._num_envs, 2), dtype=float)
        return self.states.copy()

    def step(self, actions):
        if np.any(self.current_step >= self._horizon):
            raise ValueError("Episode has already ended for some environments")
        
        self.states, r, self.have_keys, self.have_doors = self.transit(self.states, actions, self.have_keys, self.have_doors)
        self.current_step += 1
        dones = (self.current_step >= self._horizon)
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

    def opt_action(self, states, have_keys):
        actions = []
        for env, state, have_key in zip(self._envs, states, have_keys):
            action = env.opt_action(state, have_key)
            actions.append(action)
        return np.array(actions)

    def sample_flags(self, states):
        at_keys = np.all(states == self._keys, axis=1)
        have_keys = at_keys | (np.random.rand(self._num_envs) < 0.5)
        have_doors = have_keys & np.all(states == self._doors, axis=1)
        return have_keys, have_doors

    def transit(self, states, actions, have_keys, have_doors):
        actions = np.argmax(actions, axis=1)
        assert actions.shape == (self._num_envs,)

        states[:, 0] += (actions == 0).astype(float)  # move right
        states[:, 0] -= (actions == 1).astype(float)  # move
        states[:, 1] += (actions == 2).astype(float)  # move down
        states[:, 1] -= (actions == 3).astype(float)  # move
        states = np.clip(states, 0, self._envs[0].dim - 1)

        at_keys = np.all(states == self._keys, axis=1) & ~have_keys
        have_keys = at_keys | have_keys
        at_doors = (have_keys & np.all(states == self._doors, axis=1)) & ~have_doors
        have_doors = at_doors | have_doors

        if self.markovian:
            rewards = have_keys.astype(float) + have_doors.astype(float)
        else:
            rewards = at_keys.astype(float) + at_doors.astype(float)
        
        return states, rewards.astype(float), have_keys.astype(bool), have_doors.astype(bool)

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
