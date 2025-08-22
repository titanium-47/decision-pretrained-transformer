import numpy as np
import torch
import scipy
from collections import deque


class BaseRolloutPolicy:
    def __init__(self, *args, **kwargs):
        pass

    def get_action(self, states):
        raise NotImplementedError

    def reset(self):
        pass

    def update_context(self, states, actions, rewards, dones):
        pass


class ExpertRolloutPolicy(BaseRolloutPolicy):
    def __init__(self, envs):
        super().__init__()
        self.envs = envs

    def get_action(self, states):
        if hasattr(self.envs, "sample_flags"):
            expert_actions = [
                env.opt_action(state, have_key)
                for env, state, have_key in zip(
                    self.envs._envs, states, self.envs.have_keys
                )
            ]
        else:
            expert_actions = [
                env.opt_action(state) for env, state in zip(self.envs._envs, states)
            ]
        return np.array(expert_actions)


class RandomRolloutPolicy(BaseRolloutPolicy):
    def __init__(self, envs):
        super().__init__()
        self.envs = envs

    def get_action(self, states):
        return np.array([env.sample_action() for env in self.envs._envs])


class MLPRolloutPolicy(BaseRolloutPolicy):
    def __init__(self, envs, model, temp, device):
        super().__init__(envs)
        self.model = model
        self.temp = temp
        self.device = device

    def get_action(self, states):
        state = np.array(state)
        x = {"query_states": torch.from_numpy(state).float().to(self.device)}
        with torch.no_grad():
            logits = self.model(x).cpu().numpy()  # shape: (batch_size, num_actions)
            probs = scipy.special.softmax(
                logits / self.temp, axis=1
            )  # shape: (batch_size, num_actions)
            # Sample an action for each batch element
            batch_size, num_actions = probs.shape
            action_ids = np.array(
                [np.random.choice(num_actions, p=probs[i]) for i in range(batch_size)]
            )
            # Create one-hot encoded actions for the batch
            action = np.zeros((batch_size, num_actions))
            action[np.arange(batch_size), action_ids] = 1.0
        return action


class TransformerRolloutPolicy(BaseRolloutPolicy):
    def __init__(self, envs, model, temp, device, context_horizon):
        super().__init__(envs)
        self.model = model
        self.temp = temp
        self.device = device
        self.context_horizon = context_horizon
        self.env_horizon = envs.horizon

    def reset(self):
        self.context_states = []
        self.context_actions = []
        self.context_rewards = []
        self.context_dones = []

    def get_context_input(self):
        input_states = (
            torch.from_numpy(np.stack(self.context_states, axis=1))
            .float()
            .to(self.device)
        )
        input_actions = (
            torch.from_numpy(np.stack(self.context_actions, axis=1))
            .float()
            .to(self.device)
        )
        input_rewards = (
            torch.from_numpy(np.stack(self.context_rewards, axis=1))
            .float()
            .to(self.device)
        )
        input_dones = (
            torch.from_numpy(np.stack(self.context_dones, axis=1))
            .float()
            .to(self.device)
        )
        if input_states.shape[1] - 1 > self.model.horizon:
            trimmed_dones = input_dones[:, -self.context_horizon :]
            # find the first done index in the last horizon steps
            first_done_index = (
                torch.argmax(trimmed_dones, dim=1)
                + 1
                + (input_states.shape[1] - self.context_horizon)
            )
            # now get all states from the first done index to the end
            input_states = input_states[:, first_done_index.min() :]
            input_actions = input_actions[:, first_done_index.min() :]
            input_rewards = input_rewards[:, first_done_index.min() :]
            input_dones = input_dones[:, first_done_index.min() :]
            assert input_states.shape[1] < self.context_horizon - 1
        return input_states, input_actions, input_rewards, input_dones

    def get_action(self, states):
        current_states = torch.from_numpy(states).float().to(self.device)
        if len(self.context_states) < 1:
            action = (
                self.model.get_action(current_states, None, None, None, None)
                .cpu()
                .numpy()
            )
        else:
            context_states, context_actions, context_rewards, context_dones = (
                self.get_context_input()
            )
            action = (
                self.model.get_action(
                    current_states,
                    context_states,
                    context_actions,
                    context_rewards,
                    context_dones,
                )
                .cpu()
                .numpy()
            )
        probs = scipy.special.softmax(
            action / self.temp, axis=1
        )  # shape: (batch_size, num_actions)
        batch_size, num_actions = probs.shape
        action_ids = np.array(
            [np.random.choice(num_actions, p=probs[i]) for i in range(batch_size)]
        )
        action = np.zeros((batch_size, num_actions))
        action[np.arange(batch_size), action_ids] = 1.0
        return action

    def update_context(self, states, actions, rewards, dones):
        self.context_states.append(states)
        self.context_actions.append(actions)
        self.context_rewards.append(rewards)
        self.context_dones.append(dones)
