"""
Evaluate the interactive bandit model (trained with train_interactive.py).
Runs online evaluation: deploys the model on random bandit instances and
compares cumulative reward / regret vs Opt, Empirical Mean, UCB, Thompson Sampling.
"""
import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch

import common_args
from ctrls.ctrl_bandit import (
    BanditTransformerController,
    EmpMeanPolicy,
    OptPolicy,
    ThompsonSamplingPolicy,
    UCBPolicy,
)
from envs.bandit_env import BanditEnv, BanditEnvVec
from net import Transformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_eval_trajs(n_eval, dim, bandit_type="uniform"):
    """Generate eval bandit instances (means only; no rollout data)."""
    eval_trajs = []
    for _ in range(n_eval):
        if bandit_type == "uniform":
            means = np.random.uniform(0, 1, dim)
        elif bandit_type == "bernoulli":
            means = np.random.beta(1, 1, dim)
        else:
            raise ValueError(f"Unknown bandit_type: {bandit_type}")
        eval_trajs.append({"means": means})
    return eval_trajs


def deploy_online_vec(vec_env, controller, horizon):
    """Run controller on vec_env for horizon steps; return per-step mean rewards (n_envs,)."""
    num_envs = vec_env.num_envs
    context_states = np.zeros((num_envs, horizon, vec_env.dx))
    context_actions = np.zeros((num_envs, horizon, vec_env.du))
    context_next_states = np.zeros((num_envs, horizon, vec_env.dx))
    context_rewards = np.zeros((num_envs, horizon, 1))

    cum_means = []
    for h in range(horizon):
        batch = {
            "context_states": context_states[:, :h, :],
            "context_actions": context_actions[:, :h, :],
            "context_next_states": context_next_states[:, :h, :],
            "context_rewards": context_rewards[:, :h, :],
        }
        controller.set_batch_numpy_vec(batch)
        states_lnr, actions_lnr, next_states_lnr, rewards_lnr = vec_env.deploy(controller)

        context_states[:, h, :] = states_lnr
        context_actions[:, h, :] = actions_lnr
        context_next_states[:, h, :] = next_states_lnr
        context_rewards[:, h, :] = rewards_lnr[:, None]

        mean = vec_env.get_arm_value(actions_lnr)
        cum_means.append(mean)

    return np.array(cum_means)


def run_online_eval(eval_trajs, model, n_eval, horizon, var, bandit_type, sample_model=False):
    """Run online evaluation: Opt, Interactive (transformer), Emp, UCB, Thompson."""
    envs = []
    for i_eval in range(n_eval):
        traj = eval_trajs[i_eval]
        means = traj["means"]
        env = BanditEnv(means, horizon, var=var, type=bandit_type)
        envs.append(env)

    vec_env = BanditEnvVec(envs)
    all_means = {}

    # Optimal (oracle)
    controller = OptPolicy(envs, batch_size=len(envs))
    cum_means = deploy_online_vec(vec_env, controller, horizon).T
    assert cum_means.shape[0] == n_eval
    all_means["opt"] = cum_means

    # Interactive transformer
    controller = BanditTransformerController(
        model, sample=sample_model, batch_size=len(envs)
    )
    cum_means = deploy_online_vec(vec_env, controller, horizon).T
    assert cum_means.shape[0] == n_eval
    all_means["Interactive"] = cum_means

    # Empirical mean
    controller = EmpMeanPolicy(envs[0], online=True, batch_size=len(envs))
    cum_means = deploy_online_vec(vec_env, controller, horizon).T
    assert cum_means.shape[0] == n_eval
    all_means["Emp"] = cum_means

    # UCB
    controller = UCBPolicy(envs[0], const=1.0, batch_size=len(envs))
    cum_means = deploy_online_vec(vec_env, controller, horizon).T
    assert cum_means.shape[0] == n_eval
    all_means["UCB1.0"] = cum_means

    # Thompson Sampling
    controller = ThompsonSamplingPolicy(
        envs[0],
        std=var if var > 0 else 0.3,
        sample=True,
        prior_mean=0.5,
        prior_var=1 / 12.0,
        warm_start=False,
        batch_size=len(envs),
    )
    cum_means = deploy_online_vec(vec_env, controller, horizon).T
    assert cum_means.shape[0] == n_eval
    all_means["Thomp"] = cum_means

    all_means = {k: np.array(v) for k, v in all_means.items()}
    all_means_diff = {k: all_means["opt"] - v for k, v in all_means.items()}

    means = {k: np.mean(v, axis=0) for k, v in all_means_diff.items()}
    sems = {k: scipy.stats.sem(v, axis=0) for k, v in all_means_diff.items()}

    cumulative_regret = {k: np.cumsum(v, axis=1) for k, v in all_means_diff.items()}
    regret_means = {k: np.mean(v, axis=0) for k, v in cumulative_regret.items()}
    regret_sems = {k: scipy.stats.sem(v, axis=0) for k, v in cumulative_regret.items()}

    return {
        "means": means,
        "sems": sems,
        "regret_means": regret_means,
        "regret_sems": regret_sems,
        "all_means": all_means,
        "all_means_diff": all_means_diff,
    }


def plot_results(results, horizon, save_path=None):
    """Plot suboptimality and cumulative regret."""
    means = results["means"]
    sems = results["sems"]
    regret_means = results["regret_means"]
    regret_sems = results["regret_sems"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for key in means.keys():
        if key == "opt":
            ax1.plot(
                means[key], label=key, linestyle="--", color="black", linewidth=2
            )
            ax1.fill_between(
                np.arange(horizon),
                means[key] - sems[key],
                means[key] + sems[key],
                alpha=0.2,
                color="black",
            )
        else:
            ax1.plot(means[key], label=key)
            ax1.fill_between(
                np.arange(horizon),
                means[key] - sems[key],
                means[key] + sems[key],
                alpha=0.2,
            )

    ax1.set_yscale("log")
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Suboptimality")
    ax1.set_title("Online Evaluation (Interactive Bandit)")
    ax1.legend()

    for key in regret_means.keys():
        if key != "opt":
            ax2.plot(regret_means[key], label=key)
            ax2.fill_between(
                np.arange(horizon),
                regret_means[key] - regret_sems[key],
                regret_means[key] + regret_sems[key],
                alpha=0.2,
            )

    ax2.set_xlabel("Steps")
    ax2.set_ylabel("Cumulative Regret")
    ax2.set_title("Cumulative Regret Over Time")
    ax2.legend()

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Evaluate interactive bandit model")
    common_args.add_dataset_args(parser)
    common_args.add_model_args(parser)
    parser.add_argument("--n_eval", type=int, default=100, help="Number of eval envs")
    parser.add_argument("--var", type=float, default=0.3, help="Reward variance")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--model_path",
        type=str,
        default="trained_models/interactive_bandit.pt",
        help="Path to saved model state dict",
    )
    parser.add_argument(
        "--bandit_type",
        type=str,
        default="uniform",
        choices=["uniform", "bernoulli"],
        help="Bandit arm means distribution",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Sample actions from model policy (default: greedy)",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="figs/eval_interactive_bandit",
        help="Directory or path prefix for saving plots",
    )
    parser.add_argument("--env", type=str, default="bandit", help="Unused, for CLI compat")

    args = vars(parser.parse_args())

    n_eval = args["n_eval"]
    dim = args["dim"]
    horizon = args["H"]
    var = args["var"]
    seed = args["seed"]
    model_path = args["model_path"]
    bandit_type = args["bandit_type"]
    sample_model = args["sample"]
    save = args["save"]

    # Model arch (must match train_interactive)
    state_dim = 1
    action_dim = dim
    config = {
        "horizon": horizon,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "n_layer": args["layer"],
        "n_embd": args["embd"],
        "n_head": args["head"],
        "dropout": args["dropout"],
        "test": True,
    }

    if seed >= 0:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    print(f"Loading model from {model_path}")
    model = Transformer(config).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    print(f"Generating {n_eval} eval bandits (dim={dim}, horizon={horizon}, type={bandit_type})")
    eval_trajs = generate_eval_trajs(n_eval, dim, bandit_type)

    print("Running online evaluation...")
    results = run_online_eval(
        eval_trajs, model, n_eval, horizon, var, bandit_type, sample_model=sample_model
    )

    save_path = save if save.endswith(".png") else os.path.join(save, "online.png")
    plot_results(results, horizon, save_path=save_path)
    print(f"Plots saved to {save_path}")


if __name__ == "__main__":
    main()
