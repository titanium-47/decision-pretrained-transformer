import torch.multiprocessing as mp

if mp.get_start_method(allow_none=True) is None:
    mp.set_start_method("spawn", force=True)  # or 'forkserver'

import argparse
import os
import numpy as np
import torch
import wandb
import random
import tqdm
import gym
import pickle

from create_envs import create_env
from collect_data import get_dagger_dataset, merge_sequence_datasets
from models import get_model
from get_rollout_policy import get_rollout_policy

def train_step(
    step_id,
    model,
    train_loader,
    test_loader,
    save_dir,
    args,
    device,
    continuous_action,
    action_dim,
):
    save_dir = os.path.join(save_dir, f"dagger_step_{step_id}")
    os.makedirs(save_dir, exist_ok=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    def get_loss(pred_actions, true_actions):
        if continuous_action:
            log_probs = pred_actions.log_prob(true_actions).sum(-1)
            loss = -log_probs.mean()
        else:
            loss = torch.nn.functional.cross_entropy(
                pred_actions.reshape(-1, action_dim),
                true_actions.reshape(-1, action_dim),
            )
        return loss

    for epoch in tqdm.tqdm(
        range(1, args.num_epochs + 1), desc=f"Training Dagger Step {step_id}"
    ):
        if epoch % args.eval_interval == 0:
            model.eval()
            with torch.no_grad():
                test_loss = []
                for _, batch in tqdm.tqdm(
                    enumerate(test_loader), desc=f"Evaluating Epoch {epoch}",
                    total=len(test_loader)
                ):
                    batch = {k: v.to(device) for k, v in batch.items()}
                    true_actions = batch["expert_actions"]
                    pred_actions = model(batch)
                    loss = get_loss(pred_actions, true_actions)
                    test_loss.append(loss.item())
                test_loss = np.mean(test_loss)
            if args.log_wandb:
                wandb.log({f"dagger-{step_id}/test_loss": test_loss})

        model.train()
        train_loss = []
        for _, batch in tqdm.tqdm(
            enumerate(train_loader), desc=f"Training Epoch {epoch}",
            total=len(train_loader)
        ):
            batch = {k: v.to(device) for k, v in batch.items()}
            true_actions = batch["expert_actions"]
            pred_actions = model(batch)
            loss = get_loss(pred_actions, true_actions)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        train_loss = np.mean(train_loss)
        if args.log_wandb:
            wandb.log({f"dagger-{step_id}/train_loss": train_loss})

        if epoch % args.save_interval == 0:
            torch.save(
                model.state_dict(), os.path.join(save_dir, f"model_epoch_{epoch}.pth")
            )

    return model

def data_step(save_dir, step_id, train_envs, test_envs, args, rollout_policy):
    save_dir = os.path.join(save_dir, f"dagger_step_{step_id}")
    os.makedirs(save_dir, exist_ok=True)
    # Collect dataset
    if os.path.exists(os.path.join(save_dir, "train_dataset.pkl")) and os.path.exists(
        os.path.join(save_dir, "test_dataset.pkl")
    ):
        with open(os.path.join(save_dir, "train_dataset.pkl"), "rb") as f:
            train_dataset = pickle.load(f)
        with open(os.path.join(save_dir, "test_dataset.pkl"), "rb") as f:
            test_dataset = pickle.load(f)
    else:
        train_dataset, test_dataset = get_dagger_dataset(
            train_envs, test_envs, rollout_policy, args.horizon
        )
        with open(os.path.join(save_dir, "train_dataset.pkl"), "wb") as f:
            pickle.dump(train_dataset, f)
        with open(os.path.join(save_dir, "test_dataset.pkl"), "wb") as f:
            pickle.dump(test_dataset, f)

    return train_dataset, test_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="dpt")
    parser.add_argument("--env_name", type=str, default="keydoor-markovian")
    parser.add_argument("--dataset_size", type=int, default=100000)
    parser.add_argument("--dagger_steps", type=int, default=10)
    # parser.add_argument("--initial_policy_type", type=str, choices=["expert", "random", "bc", "decisiiontransformer"], default="expert")
    # parser.add_argument("--init_checkpoint_path", type=str, default=None)
    parser.add_argument("--n_envs", type=int, default=10000)
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument(
        "--model_type", type=str, choices=["decision_transformer", "mlp"], default="transformer"
    )

    parser.add_argument("--log_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="dpt")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--eval_interval", type=int, default=5)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")

    args = parser.parse_args()

    if args.log_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
            name=f"{args.exp_name}-{args.env_name}-seed{args.seed}",
        )

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = os.path.join(
        args.save_dir, f"{args.exp_name}-{args.env_name}-seed{args.seed}"
    )
    os.makedirs(save_dir, exist_ok=True)

    # Create environments
    train_envs, test_envs, eval_envs = create_env(
        args.env_name, args.dataset_size // args.dagger_steps, args.n_envs
    )
    state_dim = train_envs[0]._envs[0].state_dim
    action_dim = train_envs[0]._envs[0].action_dim

    # Model
    continuous_action = isinstance(train_envs[0]._envs[0].action_space, gym.spaces.Box)
    model_args = {
        "model_type": "decision_transformer",
        "horizon": args.horizon,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "continuous_action": continuous_action,
    }
    with open(os.path.join(save_dir, "model_args.pkl"), "wb") as f:
        pickle.dump(model_args, f)
    model = get_model(**model_args).to(device)

    params = {
        "batch_size": args.batch_size,
        "shuffle": True,
    }

    ### Load initial policy if specified
    #TODO: add masking and rollout policy
    init_rollout_policy = get_rollout_policy("expert")
    ###
    train_dataset, test_dataset = data_step(
        save_dir, 0, train_envs, test_envs, args, init_rollout_policy
    )
    for step_idx in range(args.dagger_steps):
        print(f"Starting DAgger step {step_idx}")
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Test dataset size: {len(test_dataset)}")
        model = train_step(
            step_idx,
            model,
            torch.utils.data.DataLoader(train_dataset, **params),
            torch.utils.data.DataLoader(test_dataset, **params),
            save_dir,
            args,
            device,
            continuous_action,
            action_dim,
        )
        step_policy = get_rollout_policy(
            "decision_transformer",
            model=model,
            temp=1.0,
            context_horizon=args.horizon,
            env_horizon=train_envs[0]._envs[0].horizon
        )
        if step_idx < args.dagger_steps - 1:
            step_train_dataset, step_test_dataset = data_step(
                save_dir, step_idx + 1, train_envs, test_envs, args, step_policy
            )
            train_dataset = merge_sequence_datasets(train_dataset, step_train_dataset)
            test_dataset = merge_sequence_datasets(test_dataset, step_test_dataset)
            # Combine datasets
