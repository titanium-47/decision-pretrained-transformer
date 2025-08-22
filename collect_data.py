import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_rollout(env, horizon):
    states = []
    actions = []
    next_states = []
    rewards = []
    n_envs = len(env._envs)
    state = env.reset()
    for t in range(horizon):
        state = env.sample_state()
        action = env.sample_action()
        if hasattr(env, "sample_flags"):  # for KeyDoorEnv
            have_keys, have_doors = env.sample_flags()
            next_state, reward = env.transit(state, action, have_keys, have_doors)
            action = np.array(
                [e.opt_action(s, have_keys) for e, s in zip(env._envs, state)]
            )
        else:
            next_state, reward = env.transit(state, action)
            action = np.array([e.opt_action(s) for e, s in zip(env._envs, state)])
        next_state, reward, done, _ = env.step(action)

        states.append(state)
        actions.append(action)
        next_states.append(next_state)
        rewards.append(reward)
        state = next_state

    states = np.stack(states, axis=1)
    actions = np.stack(actions, axis=1)
    next_states = np.stack(next_states, axis=1)
    rewards = np.stack(rewards, axis=1)
    assert states.shape == (n_envs, horizon, env.state_dim)
    assert actions.shape == (n_envs, horizon, env.action_dim)
    assert next_states.shape == (n_envs, horizon, env.state_dim)
    assert rewards.shape == (n_envs, horizon)
    return states, actions, next_states, rewards


def get_dpt_data(envs, horizon):
    trajs = []
    for env in envs:
        (
            context_states,
            context_actions,
            context_next_states,
            context_rewards,
        ) = get_rollout(env, horizon)
        query_state = env.sample_state()
        if hasattr(env, "sample_flags"):
            have_keys, _ = env.sample_flags()
            optimal_action = np.array(
                [e.opt_action(query_state, have_keys) for e in env._envs]
            )
        else:
            optimal_action = env.opt_action(query_state)
        n_envs = len(env._envs)
        assert query_state.shape == (n_envs, env.state_dim)
        assert optimal_action.shape == (n_envs, env.action_dim)
        for k in range(n_envs):
            context_states_k = context_states[k]
            context_actions_k = context_actions[k]
            context_next_states_k = context_next_states[k]
            context_rewards_k = context_rewards[k]

            traj = {
                "query_state": query_state[k],
                "optimal_action": optimal_action[k],
                "context_states": context_states_k,
                "context_actions": context_actions_k,
                "context_next_states": context_next_states_k,
                "context_rewards": context_rewards_k,
                "goal": env._envs[k].goal,
            }
            trajs.append(traj)
    return trajs


def dagger_rollout(env, rollout_policy, horizon):
    state = env.reset()
    rollout_policy.reset()
    n_envs = len(env._envs)
    dones = np.zeros(n_envs, dtype=bool)
    states = []
    actions = []
    expert_actions = []
    rewards = []
    dones = []

    for t in range(horizon):
        action = rollout_policy.get_action(state)
        if hasattr(env, "sample_flags"):
            expert_action = env.opt_action(state, env.have_keys)
        else:
            expert_action = env.opt_action(state)
        next_state, reward, done, _ = env.step(action)
        states.append(state)
        actions.append(action)
        expert_actions.append(expert_action)
        rewards.append(reward)
        dones.append(done)
        state = next_state

        if np.any(done):
            state = env.reset()
            dones = np.zeros(n_envs, dtype=bool)

    states = np.stack(states, axis=1)
    actions = np.stack(actions, axis=1)
    expert_actions = np.stack(expert_actions, axis=1)
    rewards = np.stack(rewards, axis=1)
    dones = np.stack(dones, axis=1)
    assert states.shape == (n_envs, horizon, env.state_dim)
    assert actions.shape == (n_envs, horizon, env.action_dim)
    assert expert_actions.shape == (n_envs, horizon, env.action_dim)
    assert rewards.shape == (n_envs, horizon)
    assert dones.shape == (n_envs, horizon)

    return {
        "states": states,
        "actions": actions,
        "expert_actions": expert_actions,
        "rewards": rewards,
        "dones": dones,
    }


def get_dagger_data(envs, rollout_policy, horizon):
    trajs = []
    for env in envs:
        traj = dagger_rollout(env, rollout_policy, horizon)
        n_envs = len(env._envs)
        for k in range(n_envs):
            traj_k = {
                "states": traj["states"][k],
                "actions": traj["actions"][k],
                "expert_actions": traj["expert_actions"][k],
                "rewards": traj["rewards"][k],
                "dones": traj["dones"][k],
                "goal": env._envs[k].goal,
            }
            trajs.append(traj_k)
    return trajs


def get_dpt_dataset(train_envs, test_envs, horizon):
    from dataset import Dataset

    train_trajs = get_dpt_data(train_envs, horizon)
    test_trajs = get_dpt_data(test_envs, horizon)
    train_dataset = Dataset(train_trajs)
    test_dataset = Dataset(test_trajs)
    return train_dataset, test_dataset


def dagger_dataset(train_envs, test_envs, rollout_policy, horizon):
    from dataset import SequenceDataset

    train_trajs = get_dagger_data(train_envs, rollout_policy, horizon)
    test_trajs = get_dagger_data(test_envs, rollout_policy, horizon)
    train_dataset = SequenceDataset(train_trajs)
    test_dataset = SequenceDataset(test_trajs)
    return train_dataset, test_dataset
