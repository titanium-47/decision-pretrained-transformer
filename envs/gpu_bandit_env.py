import gym
import torch

from envs.base_env import BaseEnv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GPUBanditEnv(BaseEnv):
    """
    GPU-parallelized bandit environment. Matches BanditEnv behavior per env.
    """
    def __init__(self, dims, n_envs, H, var=0.0, type='uniform', device=None):
        self.dims = dims
        self.dim = dims  # API compatibility with BanditEnv (e.g. ctrl_bandit uses env.dim)
        self.n_envs = n_envs
        self._device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if type == 'uniform':
            self.means = torch.rand((n_envs, dims), device=self._device)
        elif type == 'bernoulli':
            self.means = torch.distributions.Beta(1, 1).sample((n_envs, dims)).to(self._device)
        else:
            raise NotImplementedError

        opt_a_index = torch.argmax(self.means, dim=1)
        self.opt_a_index = opt_a_index  # (n_envs,) for API compatibility
        self.opt_a = torch.zeros((n_envs, dims), device=self._device)
        self.opt_a[torch.arange(n_envs, device=self._device), opt_a_index] = 1.0

        self.H_context = H
        self.H = H  # episode length: allow K steps per env (matches train_interactive loop)

        self.var = var
        self.dx = 1
        self.du = dims
        self.topk = False
        self.type = type
        self.observation_space = gym.spaces.Box(low=1, high=1, shape=(1,))
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(self.dims,))
        self.state = torch.ones((n_envs, 1), device=self._device)
        self.current_step = torch.zeros(n_envs, device=self._device)
    
    # get expected reward for a simplex of actions (only use for evaluation)
    def get_arm_value(self, actions):
        return torch.sum(self.means * actions, dim=1)
    
    # reset the environment and current step to 0
    def reset(self):
        self.current_step = torch.zeros(self.n_envs, device=self._device)
        return self.state.detach()

    # compute next state and reward for a batch of actions (matches BanditEnv.transit per env)
    def transit(self, x, us):
        us = us.to(self._device) if us.device != self._device else us
        a = torch.argmax(us, dim=1)
        mean_rewards = self.means[torch.arange(self.n_envs, device=self._device), a]
        if self.type == 'uniform':
            r = mean_rewards + torch.randn(self.n_envs, device=self._device) * self.var
        elif self.type == 'bernoulli':
            r = torch.bernoulli(mean_rewards).float()
        else:
            raise NotImplementedError
        return self.state.detach(), r.detach()
    
    # compute next state, reward, and done for a batch of actions
    def step(self, actions):
        if self.current_step.max() >= self.H:
            raise ValueError("Episode has already ended")

        _, r = self.transit(self.state, actions)
        self.current_step += 1
        done = (self.current_step >= self.H)

        return self.state.detach(), r.detach(), done, {}

    def deploy_eval(self, ctrl):
        # No variance during evaluation (matches BanditEnv)
        tmp = self.var
        self.var = 0.0
        res = self.deploy(ctrl)
        self.var = tmp
        return res