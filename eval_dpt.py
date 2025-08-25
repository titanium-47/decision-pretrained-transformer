import torch
from ctrls.ctrl_darkroom import (
    DarkroomTransformerController
)
from utils import convert_to_tensor
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def deploy_online_vec(vec_env, controller, Heps, H, horizon):
    assert H % horizon == 0
    
    ctx_rollouts = H // horizon
    trajectories = []
    num_envs = vec_env.num_envs
    context_states = torch.zeros(
        (num_envs, ctx_rollouts, horizon, vec_env.state_dim)).float().to(device)
    context_actions = torch.zeros(
        (num_envs, ctx_rollouts, horizon, vec_env.action_dim)).float().to(device)
    context_next_states = torch.zeros(
        (num_envs, ctx_rollouts, horizon, vec_env.state_dim)).float().to(device)
    context_rewards = torch.zeros(
        (num_envs, ctx_rollouts, horizon, 1)).float().to(device)

    trajectories = {
        "states": [],
        "actions": [],
        "next_states": [],
        "rewards": []
    }
    for i in range(ctx_rollouts):
        batch = {
            'context_states': context_states[:, :i, :, :].reshape(num_envs, -1, vec_env.state_dim),
            'context_actions': context_actions[:, :i, :].reshape(num_envs, -1, vec_env.action_dim),
            'context_next_states': context_next_states[:, :i, :, :].reshape(num_envs, -1, vec_env.state_dim),
            'context_rewards': context_rewards[:, :i, :, :].reshape(num_envs, -1, 1),
        }
        controller.set_batch(batch)
        states_lnr, actions_lnr, next_states_lnr, rewards_lnr = vec_env.deploy_eval(
            controller)
        context_states[:, i, :, :] = convert_to_tensor(states_lnr)
        context_actions[:, i, :, :] = convert_to_tensor(actions_lnr)
        context_next_states[:, i, :, :] = convert_to_tensor(next_states_lnr)
        context_rewards[:, i, :, :] = convert_to_tensor(rewards_lnr[:, :, None])

        trajectories["states"].append(states_lnr)
        trajectories["actions"].append(actions_lnr)
        trajectories["next_states"].append(next_states_lnr)
        trajectories["rewards"].append(rewards_lnr)

    for _ in range(ctx_rollouts, Heps):
        # Reshape the batch as a singular length H = ctx_rollouts * horizon sequence.
        batch = {
            'context_states': context_states.reshape(num_envs, -1, vec_env.state_dim),
            'context_actions': context_actions.reshape(num_envs, -1, vec_env.action_dim),
            'context_next_states': context_next_states.reshape(num_envs, -1, vec_env.state_dim),
            'context_rewards': context_rewards.reshape(num_envs, -1, 1),
        }
        controller.set_batch(batch)
        states_lnr, actions_lnr, next_states_lnr, rewards_lnr = vec_env.deploy_eval(
            controller)

        # Convert to torch
        states_lnr = convert_to_tensor(states_lnr)
        actions_lnr = convert_to_tensor(actions_lnr)
        next_states_lnr = convert_to_tensor(next_states_lnr)
        rewards_lnr = convert_to_tensor(rewards_lnr[:, :, None])

        trajectories["states"].append(states_lnr.cpu().numpy())
        trajectories["actions"].append(actions_lnr.cpu().numpy())
        trajectories["next_states"].append(next_states_lnr.cpu().numpy())
        trajectories["rewards"].append(rewards_lnr.cpu().numpy()[:, :, 0])

        # Roll in new data by shifting the batch and appending the new data.
        context_states = torch.cat(
            (context_states[:, 1:, :, :], states_lnr[:, None, :, :]), dim=1)
        context_actions = torch.cat(
            (context_actions[:, 1:, :, :], actions_lnr[:, None, :, :]), dim=1)
        context_next_states = torch.cat(
            (context_next_states[:, 1:, :, :], next_states_lnr[:, None, :, :]), dim=1)
        context_rewards = torch.cat(
            (context_rewards[:, 1:, :, :], rewards_lnr[:, None, :, :]), dim=1)
    
    trajectories["states"] = np.concatenate(trajectories["states"], axis=1)
    trajectories["actions"] = np.concatenate(trajectories["actions"], axis=1)
    trajectories["next_states"] = np.concatenate(trajectories["next_states"], axis=1)
    trajectories["rewards"] = np.concatenate(trajectories["rewards"], axis=1)
    return trajectories

def online(envs, model, Heps, H, n_eval, horizon):
    assert H % horizon == 0
    lnr_controller = DarkroomTransformerController(
        model, batch_size=n_eval, sample=True)
    trajectories = deploy_online_vec(envs, lnr_controller, Heps, H, horizon)
    return trajectories

def run_dpt_eval(eval_env, model, eval_horizon, context_horizon):
    config = {
        "horizon": eval_env.horizon,
        "H": context_horizon,
        "Heps": eval_horizon // eval_env.horizon,
        'n_eval': eval_env.num_envs
    }
    print("Running DPT eval with config:", config)
    model.test = True
    return online(eval_env, model, **config)

    