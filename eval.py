"""
DPT Evaluation Script.

Evaluates a trained DPT model on test environments and saves results
in a structure matching train_context_accumulator.py.
"""

import argparse
import os
import pickle
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch

import common_args
from create_envs import create_env
from evals import eval_darkroom
from models import Transformer, ImageTransformer
from utils import (
    build_darkroom_data_filename,
    build_darkroom_model_filename,
    build_miniworld_data_filename,
    build_miniworld_model_filename,
    wandb_init,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute_episode_returns(rewards, env_horizon):
    """
    Compute per-episode returns from trajectory rewards.
    
    Args:
        rewards: (B, T) array of rewards
        env_horizon: Number of steps per episode
    
    Returns:
        episode_returns: (B, num_episodes) array of returns
        mean_returns: (num_episodes,) mean return per episode
        std_returns: (num_episodes,) standard error per episode
    """
    episode_returns = []
    for reward_seq in rewards:
        done_indices = np.arange(0, len(reward_seq) + 1, env_horizon)
        seq_returns = [
            np.sum(reward_seq[done_indices[i]:done_indices[i + 1]]) 
            for i in range(len(done_indices) - 1)
        ]
        episode_returns.append(np.array(seq_returns))
    
    episode_returns = np.stack(episode_returns)  # (B, num_episodes)
    mean_returns = np.mean(episode_returns, axis=0)
    std_returns = np.std(episode_returns, axis=0) / np.sqrt(episode_returns.shape[0])
    
    return episode_returns, mean_returns, std_returns


def plot_returns(mean_returns, std_returns, env_name, save_path, title_suffix=""):
    """
    Plot episode returns with confidence bands.
    """
    plt.figure(figsize=(10, 6))
    episodes = np.arange(len(mean_returns))
    plt.plot(episodes, mean_returns, label='Mean Return', linewidth=2)
    plt.fill_between(
        episodes, 
        mean_returns - std_returns, 
        mean_returns + std_returns, 
        alpha=0.2, 
        label='Standard Error'
    )
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title(f'DPT Evaluation Returns on {env_name}{title_suffix}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_dagger_data(data, filepath):
    """Save trajectory data as pickle file."""
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved trajectories to {filepath}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DPT Evaluation")
    common_args.add_dataset_args(parser)
    common_args.add_model_args(parser)
    common_args.add_eval_args(parser)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--log_wandb', action='store_true', help='Log results to wandb')
    parser.add_argument('--save_dir', type=str, default='./results', help='Base directory for saving results')

    args = vars(parser.parse_args())
    print("Args: ", args)

    # Extract args
    n_envs = args['envs']
    n_hists = args['hists']
    n_samples = args['samples']
    H = args['H']
    dim = args['dim']
    n_embd = args['embd']
    n_head = args['head']
    n_layer = args['layer']
    lr = args['lr']
    epoch = args['epoch']
    shuffle = args['shuffle']
    dropout = args['dropout']
    var = args['var']
    cov = args['cov']
    test_cov = args['test_cov']
    envname = args['env']
    horizon = args['hor']
    n_eval = args['n_eval']
    seed = args['seed']
    data_ratio = args['data_ratio']
    
    # Set seeds
    tmp_seed = seed if seed != -1 else 0
    torch.manual_seed(tmp_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(tmp_seed)
    np.random.seed(tmp_seed)
    random.seed(tmp_seed)

    if test_cov < 0:
        test_cov = cov
    if horizon < 0:
        horizon = H

    # Build model filename
    model_config = {
        'shuffle': shuffle,
        'lr': lr,
        'dropout': dropout,
        'n_embd': n_embd,
        'n_layer': n_layer,
        'n_head': n_head,
        'n_envs': n_envs,
        'n_hists': n_hists,
        'n_samples': n_samples,
        'horizon': horizon,
        'dim': dim,
        'seed': seed,
        'data_ratio': data_ratio,
        'rollin_type': args['rollin_type'],
    }

    if envname == 'miniworld':
        filename = build_miniworld_model_filename(envname, model_config)
    else:
        filename = build_darkroom_model_filename(envname, model_config)

    # Create environments
    from create_envs import create_env
    _, _, eval_envs = create_env(envname, n_envs, 1000)
    state_dim = eval_envs[0].state_dim
    action_dim = eval_envs[0].action_dim
    env_horizon = eval_envs[0].horizon

    # Model config
    config = {
        'horizon': H,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'n_layer': n_layer,
        'n_embd': n_embd,
        'n_head': n_head,
        'dropout': dropout,
        'test': True,
        'continuous_action': False
    }

    # Load model
    if envname == 'miniworld':
        config.update({'image_size': 25})
        model = ImageTransformer(config).to(device)
    else:
        model = Transformer(config).to(device)
    
    if epoch < 0:
        model_path = f'models/{filename}.pt'
    else:
        model_path = f'models/{filename}_epoch{epoch}.pt'
    
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
    model.eval()

    # Setup save directory (matching train_context_accumulator structure)
    exp_name = f"dpt-{envname}-seed{seed}"
    save_dir = os.path.join(args['save_dir'], exp_name)
    eval_save_dir = os.path.join(save_dir, f"eval_epoch{epoch}")
    os.makedirs(eval_save_dir, exist_ok=True)
    
    print(f"Saving results to {eval_save_dir}")

    # Save model args for reproducibility
    with open(os.path.join(save_dir, 'model_args.pkl'), 'wb') as f:
        pickle.dump(config, f)

    # Initialize wandb if requested
    wandb = None
    if args['log_wandb']:
        wandb = wandb_init(args, name=exp_name, tag='dpt-eval')

    # Run evaluation
    Heps = args['n_eval']  # Number of evaluation episodes
    eval_config = {
        'Heps': Heps,
        'H': H,
        'n_eval': eval_envs[0].num_envs,
        'dim': dim,
        'permuted': False,
        'continuous_action': False,
    }
    
    print(f"\nRunning DPT evaluation:")
    print(f"  Environment: {envname}")
    print(f"  Context horizon (H): {H}")
    print(f"  Env horizon: {env_horizon}")
    print(f"  Eval episodes: {Heps}")
    print(f"  Num eval envs: {eval_envs[0].num_envs}")
    
    start_time = time.time()
    trajectories = eval_darkroom.online(eval_envs, model, **eval_config)
    eval_time = time.time() - start_time
    print(f"Evaluation took {eval_time:.2f} seconds")

    # Save trajectories
    save_dagger_data(trajectories, os.path.join(eval_save_dir, 'eval_trajs.pkl'))

    # Compute episode returns
    rewards = trajectories['rewards']  # (B, T)
    episode_returns, mean_returns, std_returns = compute_episode_returns(rewards, env_horizon)

    # Save returns (matching eval_policy.py format)
    np.savez(
        os.path.join(eval_save_dir, 'eval_returns.npz'),
        episode_returns=episode_returns,
        mean_returns=mean_returns,
        std_returns=std_returns,
    )
    print(f"Saved evaluation returns to {os.path.join(eval_save_dir, 'eval_returns.npz')}")

    # Save plot
    plot_path = os.path.join(eval_save_dir, 'eval_returns.png')
    plot_returns(mean_returns, std_returns, envname, plot_path, f" (epoch {epoch})")
    print(f"Saved evaluation plot to {plot_path}")

    # Log to wandb
    if wandb is not None:
        # Log summary stats
        final_mean = mean_returns[-1]
        final_std = std_returns[-1]
        wandb.log({
            "eval/final_return": final_mean,
            "eval/final_return_std": final_std,
            "eval/mean_return": np.mean(mean_returns),
            "eval/eval_time": eval_time,
        })
        
        # Log returns plot
        fig, ax = plt.subplots(figsize=(10, 6))
        episodes = np.arange(len(mean_returns))
        ax.plot(episodes, mean_returns, label='Mean Return', linewidth=2)
        ax.fill_between(
            episodes,
            mean_returns - std_returns,
            mean_returns + std_returns,
            alpha=0.2,
        )
        ax.set_xlabel('Episode')
        ax.set_ylabel('Return')
        ax.set_title(f'DPT Eval Returns - {envname} (epoch {epoch})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        wandb.log({"eval/returns_plot": wandb.Image(fig)})
        plt.close(fig)
        
        # Log per-episode returns
        for ep_idx, (mean_ret, std_ret) in enumerate(zip(mean_returns, std_returns)):
            wandb.log({
                f"eval_episodes/episode_{ep_idx}_return": mean_ret,
            })
        
        wandb.finish()

    # Print summary
    print(f"\n{'='*60}")
    print(f"Evaluation Summary")
    print(f"{'='*60}")
    print(f"Environment: {envname}")
    print(f"Model epoch: {epoch}")
    print(f"Num episodes: {len(mean_returns)}")
    print(f"Final return: {mean_returns[-1]:.4f} ± {std_returns[-1]:.4f}")
    print(f"Mean return (all eps): {np.mean(mean_returns):.4f}")
    print(f"Max return: {np.max(mean_returns):.4f}")
    print(f"Results saved to: {eval_save_dir}")
    print(f"{'='*60}")
