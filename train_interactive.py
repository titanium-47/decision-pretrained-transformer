import argparse
import os
import random
import time

import numpy as np
import torch

import common_args
from envs.gpu_bandit_env import GPUBanditEnv
from net import Transformer

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


def set_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def build_bandit_env(dim, n_envs, horizon, var, env_name):
    if env_name == 'bandit':
        bandit_type = 'uniform'
    elif env_name == 'bandit_thompson':
        bandit_type = 'bernoulli'
    else:
        raise ValueError(f"Unsupported env: {env_name}")
    env = GPUBanditEnv(dim, n_envs, horizon, var=var, type=bandit_type, device=device)
    return env


if __name__ == '__main__':
    if not os.path.exists('models'):
        os.makedirs('models', exist_ok=True)

    parser = argparse.ArgumentParser()
    common_args.add_dataset_args(parser)
    common_args.add_model_args(parser)
    common_args.add_train_args(parser)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--K', type=int, default=-1, help='Interactive rollout steps')
    parser.add_argument('--episodes_per_epoch', type=int, default=1000)

    args = vars(parser.parse_args())
    print("Args: ", args)

    env_name = args['env']
    if env_name not in ['bandit', 'bandit_thompson']:
        raise ValueError("train_interactive.py currently supports bandit envs only.")

    horizon = args['H']
    K = args['K'] if args['K'] > 0 else horizon
    dim = args['dim']
    state_dim = 1
    action_dim = dim
    n_embd = args['embd']
    n_layer = args['layer']
    n_head = args['head']
    lr = args['lr']
    dropout = args['dropout']
    var = args['var']
    num_epochs = args['num_epochs']
    seed = args['seed']
    episodes_per_epoch = args['episodes_per_epoch']
    n_envs = args['envs']

    tmp_seed = 0 if seed == -1 else seed
    set_seeds(tmp_seed)

    config = {
        'horizon': K,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'n_layer': n_layer,
        'n_embd': n_embd,
        'n_head': n_head,
        'dropout': dropout,
        'test': True,
    }
    model = Transformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    zeros = torch.zeros((n_envs, state_dim ** 2 + action_dim + 1), device=device)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()

        for _ in range(episodes_per_epoch):
            env = build_bandit_env(dim, n_envs, K, var, env_name)
            state = env.reset()

            context_states = torch.zeros((n_envs, K, state_dim), device=device)
            context_actions = torch.zeros((n_envs, K, action_dim), device=device)
            context_next_states = torch.zeros((n_envs, K, state_dim), device=device)
            context_rewards = torch.zeros((n_envs, K, 1), device=device)

            last_logits = None
            for t in range(K):
                query_states = state.float().view(n_envs, state_dim)
                batch = {
                    'context_states': context_states[:, :t, :],
                    'context_actions': context_actions[:, :t, :],
                    'context_next_states': context_next_states[:, :t, :],
                    'context_rewards': context_rewards[:, :t, :],
                    'query_states': query_states,
                    'zeros': zeros,
                }

                logits = model(batch)
                last_logits = logits
                action_dist = torch.distributions.Categorical(logits=logits)
                action_idx = action_dist.sample()
                action = torch.nn.functional.one_hot(
                    action_idx, num_classes=action_dim).float().to(device)

                next_state, reward, _, _ = env.step(action)

                context_states[:, t, :] = state.float()
                context_actions[:, t, :] = action
                context_next_states[:, t, :] = next_state.float()
                context_rewards[:, t, 0] = reward.float()

                state = next_state

            target = env.opt_a_index.to(device)
            loss = loss_fn(last_logits, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(1, episodes_per_epoch)
        elapsed = time.time() - start_time
        print(f"Epoch {epoch + 1}/{num_epochs} - loss: {avg_loss:.6f} - time: {elapsed:.2f}s")

    torch.save(model.state_dict(), 'models/interactive_bandit.pt')
    print("Done.")
