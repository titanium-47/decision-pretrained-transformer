import argparse
import os
import numpy as np
import random
import torch
import pickle
import pathlib
from create_envs import create_env
from models import get_model
from collect_data import get_dagger_data, save_dagger_data
from get_rollout_policy import get_rollout_policy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='eval_policy')
    parser.add_argument('--env_name', type=str, default='darkroom-easy')
    parser.add_argument('--n_envs', type=int, default=1000)
    # parser.add_argument('--env_horizon', type=int, default=100)
    parser.add_argument('--eval_episodes', type=int, default=40)
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--policy_type', type=str, choices=['', 'random', 'expert'], default='')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_path', type=str, default='./results')
    parser.add_argument('--plot_returns', action='store_true')
    args = parser.parse_args()

    _, _, eval_envs = create_env(args.env_name, 10000, args.n_envs)
    assert len(eval_envs) == 1
    # print(f"Created {eval_envs.num_envs} environments of type {args.env_name} with horizon {args.eval_horizon}")
    env_horizon = eval_envs[0]._envs[0].horizon
    eval_horizon = args.eval_episodes * env_horizon
    

    np.random.seed(args.seed)
    random.seed(args.seed)
    save_dir = os.path.join(args.save_path, f"{args.exp_name}-{args.env_name}-seed{args.seed}")
    os.makedirs(save_dir, exist_ok=True)

    # Load the trained model
    try:
        model_args_pkl = os.path.join(args.checkpoint_path, 'model_args.pkl')
        with open(model_args_pkl, 'rb') as f:
            model_args = pickle.load(f)
    except Exception as e: #HACK
        model_args_pkl = os.path.join(args.checkpoint_path, '../model_args.pkl')
        with open(model_args_pkl, 'rb') as f:
            model_args = pickle.load(f)
    model = get_model(**model_args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoints = pathlib.Path(args.checkpoint_path).glob('model_epoch_*.pth')
    checkpoint = sorted(checkpoints, key=lambda x: int(x.stem.split('_')[-1]))[-1]
    model.load_state_dict(torch.load(checkpoint, map_location=device), strict=False) #HACK

    model.eval()

    policy_type = args.policy_type if len(args.policy_type) > 0 else model_args["model_type"]
    if policy_type == "transformer":
        from eval_dpt import run_dpt_eval
        eval_trajs = run_dpt_eval(eval_envs[0], model, eval_horizon, model_args['horizon'])
    else:
        policy = get_rollout_policy(policy_type, model=model, temp=1.0, context_horizon=model_args['horizon'], env_horizon=env_horizon)
        eval_trajs = get_dagger_data(eval_envs, policy, eval_horizon)
    eval_trajs = save_dagger_data(eval_trajs, os.path.join(save_dir, 'eval_trajs.pkl'))

    if args.plot_returns:
        returns = eval_trajs['rewards'] # B x T
        episode_returns = []
        for re in returns:
            done_indices = np.arange(0, len(re)+1, env_horizon)
            seq_returns = [np.sum(re[done_indices[i]:done_indices[i + 1]]) for i in range(len(done_indices) - 1)]
            seq_returns = np.array(seq_returns)
            episode_returns.append(seq_returns)
        episode_returns = np.stack(episode_returns) # B x num_episodes
        mean_returns = np.mean(episode_returns, axis=0)
        std_returns = np.std(episode_returns, axis=0) / np.sqrt(episode_returns.shape[0])  # Standard Error
        np.savez(os.path.join(save_dir, 'eval_returns.npz'), mean_returns=mean_returns, std_returns=std_returns)
        print(f"Saved evaluation returns to {os.path.join(save_dir, 'eval_returns.npz')}")

        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(mean_returns, label='Mean Return')
        plt.fill_between(np.arange(len(mean_returns)), mean_returns - std_returns, mean_returns +
                            std_returns, alpha=0.2, label='Std Dev')
        plt.xlabel('Episode')
        plt.ylabel('Return')
        plt.title(f'Evaluation Returns on {args.env_name}')
        plt.legend()
        plt.savefig(os.path.join(save_dir, 'eval_returns.png'))
        print(f"Saved evaluation returns plot to {os.path.join(save_dir, 'eval_returns.png')}")

