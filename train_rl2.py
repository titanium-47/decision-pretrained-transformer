"""
RL^2 Baseline using Recurrent PPO from sb3_contrib.

RL^2 learns to adapt within a meta-episode by conditioning on reward history.
The observation is augmented with (prev_reward, prev_done) to provide adaptation signal.
"""

import argparse
import os
import pickle
import random

import numpy as np
import torch
import wandb
import matplotlib.pyplot as plt

from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib import RecurrentPPO

from create_envs import create_env
from envs.meta_env import MetaEnv, MetaVecEnv


class WandbCallback(BaseCallback):
    """Callback for logging to wandb."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        
    def _on_step(self):
        if self.n_calls % 1000 == 0:
            if len(self.model.ep_info_buffer) > 0:
                ep_rewards = [ep['r'] for ep in self.model.ep_info_buffer]
                wandb.log({
                    "train/mean_reward": np.mean(ep_rewards),
                    "train/std_reward": np.std(ep_rewards),
                    "train/timesteps": self.num_timesteps,
                })
        return True


def extract_env_pool(vec_envs):
    """Extract individual environments from list of VecEnvs."""
    env_pool = []
    for vec_env in vec_envs:
        # VecEnvs have ._envs attribute containing individual envs
        if hasattr(vec_env, '_envs'):
            env_pool.extend(vec_env._envs)
        else:
            env_pool.append(vec_env)
    return env_pool


def evaluate(model, env_pool, num_meta_episodes, env_horizon, n_eval=10):
    """Evaluate the model on test environments."""
    all_returns = []
    
    for env in env_pool[:n_eval]:
        # Create meta-env with single env in pool (fixed goal for this eval)
        meta_env = MetaEnv([env], num_meta_episodes, env_horizon)
        
        # Rollout
        obs = meta_env.reset()
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)
        
        episode_rewards = []
        current_ep_reward = 0
        
        for _ in range(meta_env.meta_horizon):
            action, lstm_states = model.predict(
                obs[None], state=lstm_states, episode_start=episode_starts, deterministic=True
            )
            obs, reward, done, _ = meta_env.step(action[0])
            current_ep_reward += reward
            episode_starts = np.array([done])
            
            # Track per-episode returns using internal done signal
            if obs[-1] > 0.5:  # prev_done
                episode_rewards.append(current_ep_reward)
                current_ep_reward = 0
            
            if done:
                break
        
        all_returns.append(episode_rewards)
    
    # Compute mean/std per episode across goals
    max_eps = max(len(r) for r in all_returns)
    padded = np.zeros((len(all_returns), max_eps))
    for i, r in enumerate(all_returns):
        padded[i, :len(r)] = r
    
    mean_returns = padded.mean(axis=0)
    std_returns = padded.std(axis=0)
    
    return mean_returns, std_returns, all_returns


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL^2 Baseline")
    
    # Experiment
    parser.add_argument("--exp_name", type=str, default="rl2")
    parser.add_argument("--env_name", type=str, default="darkroom-easy")
    parser.add_argument("--seed", type=int, default=42)
    
    # Meta-learning
    parser.add_argument("--num_meta_episodes", type=int, default=4, help="Episodes per meta-episode")
    parser.add_argument("--dataset_size", type=int, default=10000, help="Number of envs to create")
    
    # Training
    parser.add_argument("--total_timesteps", type=int, default=1000000)
    parser.add_argument("--n_envs", type=int, default=256, help="Parallel meta-envs for PPO")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n_steps", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=10)
    
    # Evaluation
    parser.add_argument("--eval_freq", type=int, default=50000)
    parser.add_argument("--n_eval", type=int, default=10)
    
    # Logging
    parser.add_argument("--log_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="dpt-sweep")
    parser.add_argument("--save_dir", type=str, default="./rl2_results")
    
    args = parser.parse_args()
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Save directory
    save_dir = os.path.join(args.save_dir, f"{args.exp_name}-{args.env_name}-seed{args.seed}")
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize wandb
    if args.log_wandb:
        wandb.init(
            project=args.wandb_project,
            config=vars(args),
            name=f"{args.exp_name}-{args.env_name}-seed{args.seed}",
        )
    
    # Create environments using create_env
    print(f"Creating environments: {args.env_name}")
    train_vec_envs, test_vec_envs, eval_vec_envs = create_env(
        args.env_name, args.dataset_size, n_envs=100  # batch size for creation
    )
    
    # Extract individual envs from VecEnvs
    train_env_pool = extract_env_pool(train_vec_envs)
    test_env_pool = extract_env_pool(test_vec_envs)
    eval_env_pool = extract_env_pool(eval_vec_envs)
    
    env_horizon = train_env_pool[0].horizon
    
    print(f"Train env pool: {len(train_env_pool)}, Test: {len(test_env_pool)}, Eval: {len(eval_env_pool)}")
    print(f"Env horizon: {env_horizon}, Meta-episodes: {args.num_meta_episodes}")
    
    # Create vectorized meta-environment for training (no subprocess overhead)
    vec_env = MetaVecEnv(train_env_pool, args.n_envs, args.num_meta_episodes, env_horizon)
    
    # Create model
    model = RecurrentPPO(
        "MlpLstmPolicy",
        vec_env,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        verbose=1,
        seed=args.seed,
    )
    
    print(f"Model created. Training for {args.total_timesteps} timesteps...")
    
    # Training loop with periodic evaluation
    timesteps_done = 0
    eval_results = []
    
    while timesteps_done < args.total_timesteps:
        # Train for eval_freq steps
        steps_to_train = min(args.eval_freq, args.total_timesteps - timesteps_done)
        
        callbacks = [WandbCallback()] if args.log_wandb else []
        model.learn(total_timesteps=steps_to_train, reset_num_timesteps=False, callback=callbacks)
        timesteps_done += steps_to_train
        
        # Evaluate on test envs
        print(f"\nEvaluating at {timesteps_done} timesteps...")
        mean_returns, std_returns, _ = evaluate(
            model, eval_env_pool, args.num_meta_episodes, env_horizon, args.n_eval
        )
        
        eval_results.append({
            "timesteps": timesteps_done,
            "mean_returns": mean_returns,
            "std_returns": std_returns,
        })
        
        print(f"  Final episode return: {mean_returns[-1]:.3f} ± {std_returns[-1]:.3f}")
        
        if args.log_wandb:
            wandb.log({
                "eval/final_return": mean_returns[-1],
                "eval/mean_return": np.mean(mean_returns),
                "timesteps": timesteps_done,
            })
            
            # Log per-episode returns
            for ep_idx, (mean_ret, std_ret) in enumerate(zip(mean_returns, std_returns)):
                wandb.log({f"eval/episode_{ep_idx}_return": mean_ret, "timesteps": timesteps_done})
    
    # Save final model and results
    model.save(os.path.join(save_dir, "final_model"))
    
    with open(os.path.join(save_dir, "eval_results.pkl"), "wb") as f:
        pickle.dump(eval_results, f)
    
    # Final evaluation
    mean_returns, std_returns, all_returns = evaluate(
        model, eval_env_pool, 40, env_horizon, len(eval_env_pool)
    )
    
    np.savez(
        os.path.join(save_dir, "eval_returns.npz"),
        mean_returns=mean_returns,
        std_returns=std_returns,
    )
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    episodes = np.arange(len(mean_returns))
    ax.plot(episodes, mean_returns, label='Mean Return', linewidth=2)
    ax.fill_between(episodes, mean_returns - std_returns, mean_returns + std_returns, alpha=0.2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Return')
    ax.set_title(f'RL^2 Eval Returns - {args.env_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, "eval_returns.png"), dpi=150, bbox_inches='tight')
    
    if args.log_wandb:
        wandb.log({"eval/returns_plot": wandb.Image(fig)})
    
    plt.close(fig)
    
    print(f"\nTraining complete! Results saved to {save_dir}")
    
    if args.log_wandb:
        wandb.finish()
