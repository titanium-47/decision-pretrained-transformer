"""
SPOC Training Algorithm.

Baseline for context_accumulator - trains for one iteration with sample_time=True,
then evaluates with sliding_window=True.
"""

import torch.multiprocessing as mp

if mp.get_start_method(allow_none=True) is None:
    mp.set_start_method("spawn", force=True)

import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
import wandb
import random
import tqdm
import gym
import pickle
from collections import defaultdict

import matplotlib.pyplot as plt

from create_envs import create_env
from collect_data import get_dagger_dataset
from dataset import collate_fn
from eval_policy import evaluate_policy_on_envs
from models import DecisionTransformer
from get_rollout_policy import get_rollout_policy


def get_loss_mask(attention_mask, horizon):
    """
    Get a mask for the loss only on the last `horizon` tokens.
    
    Args:
        attention_mask: (batch_size, seq_len) tensor
        horizon: Number of tokens from the end to include in loss
    
    Returns:
        loss_mask: (batch_size, seq_len) tensor with 1s for tokens to include
    """
    loss_mask = torch.zeros_like(attention_mask)
    for i in range(loss_mask.size(0)):
        non_zero_indices = torch.nonzero(attention_mask[i], as_tuple=False).squeeze()
        if len(non_zero_indices.shape) == 0:
            non_zero_indices = non_zero_indices.unsqueeze(0)
        if len(non_zero_indices) >= horizon:
            loss_mask[i, non_zero_indices[-horizon:]] = 1
        else:
            loss_mask[i, non_zero_indices] = 1
    return loss_mask


def get_optimizer_scheduler(model, total_steps, lr, warmup_ratio):
    """Create optimizer with warmup + cosine decay schedule."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    warmup_steps = int(total_steps * warmup_ratio)

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_steps - warmup_steps)
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup, cosine], milestones=[warmup_steps]
    )
    
    return optimizer, scheduler


def train_step(
    model,
    optimizer,
    scheduler,
    train_loader,
    test_loader,
    save_dir,
    args,
    device,
    action_dim,
    env_horizon,
    sample_time,
):
    """
    Train model for SPOC.
    
    Args:
        model: Model to train
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        train_loader: Training data loader
        test_loader: Test data loader
        save_dir: Directory to save checkpoints
        args: Training arguments
        device: Device to train on
        action_dim: Action dimension
        env_horizon: Environment horizon (steps per episode)
        sample_time: Whether to sample time for training
    
    Returns:
        Trained model
    """
    os.makedirs(save_dir, exist_ok=True)
    
    eval_freq = max(1, int(args.eval_interval * args.num_epochs))
    save_freq = max(1, int(args.save_interval * args.num_epochs))

    def forward(batch):
        """Compute loss for a batch (discrete actions only)."""
        batch = {k: v.to(device) for k, v in batch.items()}
        true_actions = batch["expert_actions"]
        pred_actions, _ = model(batch, sample_time=sample_time)
        
        # Cross entropy loss for discrete actions
        action_loss = F.cross_entropy(
            pred_actions.reshape(-1, action_dim),
            true_actions.reshape(-1, action_dim),
            reduction='none'
        )
        
        # Apply loss mask to only compute loss on last env_horizon tokens
        loss_mask = get_loss_mask(batch['attention_mask'], env_horizon)
        loss = (action_loss.reshape(batch['attention_mask'].shape) * loss_mask).sum() / loss_mask.sum()
        
        return loss, {"loss": loss.item()}
    
    for epoch in tqdm.tqdm(range(args.num_epochs), desc="Training SPOC"):
        # Evaluation
        if epoch % eval_freq == 0 or epoch == args.num_epochs - 1:
            model.eval()
            eval_stats = defaultdict(list)
            
            with torch.no_grad():
                for batch in test_loader:
                    loss, stats = forward(batch)
                    for k, v in stats.items():
                        eval_stats[k].append(v)
            
            for k, v in eval_stats.items():
                eval_stats[k] = np.mean(v)
            
            if args.log_wandb:
                for k, v in eval_stats.items():
                    wandb.log({f"spoc/test_{k}": v})
            
            print(f"Epoch {epoch} - Test Loss: {eval_stats['loss']:.4f}")

        # Training
        model.train()
        train_stats = defaultdict(list)
        
        for batch in train_loader:
            loss, stats = forward(batch)
            
            optimizer.zero_grad()
            loss.backward()
            
            # Compute gradient norm
            if args.log_wandb:
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                wandb.log({
                    "spoc/grad_norm": total_norm,
                    "spoc/lr": optimizer.param_groups[0]['lr']
                })
            
            # Clip gradients
            if args.gradient_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            for k, v in stats.items():
                train_stats[k].append(v)
        
        if args.log_wandb:
            for k, v in train_stats.items():
                wandb.log({f"spoc/{k}": np.mean(v)})

        # Save checkpoint
        if epoch % save_freq == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f"model_epoch_{epoch}.pth"))
    
    return model


def data_step(save_dir, train_envs, test_envs, rollout_policy, horizon):
    """
    Collect data for SPOC training.
    
    Args:
        save_dir: Directory to save/load data
        train_envs: Training environments
        test_envs: Test environments
        rollout_policy: Policy to use for data collection
        horizon: Horizon for data collection
    
    Returns:
        train_dataset, test_dataset
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Check if data already exists
    train_path = os.path.join(save_dir, "train_dataset.pkl")
    test_path = os.path.join(save_dir, "test_dataset.pkl")
    
    if os.path.exists(train_path) and os.path.exists(test_path):
        print(f"Loading existing data from {save_dir}")
        with open(train_path, "rb") as f:
            train_dataset = pickle.load(f)
        with open(test_path, "rb") as f:
            test_dataset = pickle.load(f)
    else:
        print(f"Collecting new data with horizon={horizon}")
        train_dataset, test_dataset = get_dagger_dataset(
            train_envs, test_envs, rollout_policy, horizon
        )
        with open(train_path, "wb") as f:
            pickle.dump(train_dataset, f)
        with open(test_path, "wb") as f:
            pickle.dump(test_dataset, f)

    return train_dataset, test_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SPOC Training")
    
    # Experiment
    parser.add_argument("--exp_name", type=str, default="spoc")
    parser.add_argument("--env_name", type=str, default="darkroom-easy")
    parser.add_argument("--seed", type=int, default=42)
    
    # Data
    parser.add_argument("--dataset_size", type=int, default=10000)
    parser.add_argument("--n_envs", type=int, default=10000)
    parser.add_argument("--horizon", type=int, default=1000, help="Model/context horizon")
    
    # Evaluation
    parser.add_argument("--eval_episodes", type=int, default=40, help="Number of episodes for evaluation")
    
    # Model
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    
    # Training
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--gradient_clip", action="store_true")
    parser.add_argument("--eval_interval", type=float, default=0.1)
    parser.add_argument("--save_interval", type=float, default=0.1)
    
    # Logging
    parser.add_argument("--log_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="dpt-sweep")
    parser.add_argument("--wandb_entity", type=str, default=None)
    
    # Paths
    parser.add_argument("--save_dir", type=str, default="./spoc_results")

    args = parser.parse_args()

    # Initialize wandb
    if args.log_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
            name=f"{args.exp_name}-{args.env_name}-seed{args.seed}",
        )

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Save directory
    save_dir = os.path.join(args.save_dir, f"{args.exp_name}-{args.env_name}-seed{args.seed}")
    os.makedirs(save_dir, exist_ok=True)

    # Create environments
    print(f"Creating environments: {args.env_name}")
    train_envs, test_envs, eval_envs = create_env(args.env_name, args.dataset_size, args.n_envs)
    
    state_dim = train_envs[0]._envs[0].state_dim
    action_dim = train_envs[0]._envs[0].action_dim
    env_horizon = train_envs[0]._envs[0].horizon
    
    print(f"State dim: {state_dim}, Action dim: {action_dim}, Env horizon: {env_horizon}")
    print(f"Model horizon: {args.horizon}")

    # Model configuration (discrete actions only)
    continuous_action = isinstance(train_envs[0]._envs[0].action_space, gym.spaces.Box)
    model_args = {
        "horizon": 4000,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "n_layer": args.num_layers,
        "n_head": args.num_heads,
        "n_embd": 128,
        "dropout": args.dropout,
        "shuffle": True,
        "test": False,
        "continuous_action": continuous_action,
        "gmm_heads": 1,
    }
    
    with open(os.path.join(save_dir, "model_args.pkl"), "wb") as f:
        pickle.dump(model_args, f)
    
    # Create model
    model = DecisionTransformer(model_args).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Collect data with expert policy
    print("\n" + "="*60)
    print("SPOC: Collecting data with expert policy")
    print("="*60)
    
    init_rollout_policy = get_rollout_policy("expert")
    train_dataset, test_dataset = data_step(
        os.path.join(save_dir, "data"),
        train_envs, test_envs, init_rollout_policy, args.horizon
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Create data loaders with collate_fn
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    
    # Setup optimizer and scheduler
    total_steps = len(train_dataset) // args.batch_size * args.num_epochs
    optimizer, scheduler = get_optimizer_scheduler(model, total_steps, args.lr, args.warmup_ratio)
    
    # Train with sample_time=True
    print("\n" + "="*60)
    print("SPOC: Training with sample_time=True")
    print("="*60)
    
    model = train_step(
        model,
        optimizer,
        scheduler,
        train_loader,
        test_loader,
        os.path.join(save_dir, "checkpoints"),
        args,
        device,
        action_dim,
        env_horizon,
        sample_time=True,
    )
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(save_dir, "final_model.pth"))
    
    # Evaluate with sliding_window=True
    print("\n" + "="*60)
    print("SPOC: Evaluating with sliding_window=True")
    print("="*60)
    
    eval_policy = get_rollout_policy(
        "decision_transformer",
        model=model,
        context_horizon=4000,
        env_horizon=env_horizon,
        context_accumulation=False,
        sliding_window=True,
    )
    
    eval_save_dir = os.path.join(save_dir, "eval")
    eval_horizon = args.eval_episodes * env_horizon
    
    eval_results = evaluate_policy_on_envs(
        eval_envs=eval_envs,
        policy=eval_policy,
        eval_horizon=eval_horizon,
        env_horizon=env_horizon,
        save_dir=eval_save_dir,
        env_name=args.env_name,
        plot=True,
    )
    
    # Log to wandb
    if args.log_wandb:
        # Log summary stats
        final_mean = eval_results['mean_returns'][-1]
        final_std = eval_results['std_returns'][-1]
        wandb.log({
            "eval/final_return": final_mean,
            "eval/final_return_std": final_std,
            "eval/mean_return": np.mean(eval_results['mean_returns']),
        })
        
        # Log returns plot
        fig, ax = plt.subplots(figsize=(10, 6))
        episodes = np.arange(len(eval_results['mean_returns']))
        ax.plot(episodes, eval_results['mean_returns'], label='Mean Return', linewidth=2)
        ax.fill_between(
            episodes,
            eval_results['mean_returns'] - eval_results['std_returns'],
            eval_results['mean_returns'] + eval_results['std_returns'],
            alpha=0.2,
        )
        ax.set_xlabel('Episode')
        ax.set_ylabel('Return')
        ax.set_title(f'SPOC Eval Returns - {args.env_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        wandb.log({"eval/returns_plot": wandb.Image(fig)})
        plt.close(fig)
        
        # Log episode returns
        for ep_idx, (mean_ret, std_ret) in enumerate(zip(eval_results['mean_returns'], eval_results['std_returns'])):
            wandb.log({f"eval/episode_{ep_idx}_return": mean_ret})
    
    print(f"\nEvaluation complete - Final return: {eval_results['mean_returns'][-1]:.2f} ± {eval_results['std_returns'][-1]:.2f}")
    print(f"\nTraining complete! Results saved to {save_dir}")
    
    if args.log_wandb:
        wandb.finish()
