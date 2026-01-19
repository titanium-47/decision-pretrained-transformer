## data generation


from create_envs import create_navigation_env
import numpy as np
import pickle
import torch
from utils import convert_to_tensor
import tqdm
from models import DecisionTransformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_envs, _, _ = create_navigation_env("sparse-2dnav-reset_free-random_start-long-hard", 100000, 10000)


def get_rollout(env, horizon):
    steps = horizon // env.horizon
    envs = env._envs

    # sample a grid of envs of shape (len(envs), steps)
    sampled_envs = []
    for i in range(len(envs)):
        # create a temp list without current env
        other_envs = envs[:i] + envs[i+1:]
        sampled = np.random.choice(other_envs, size=steps, replace=False)
        sampled_envs.append(sampled)
    sampled_envs = np.array(sampled_envs)
    sample_termination_step = np.random.randint(1, steps + 1, size=len(envs))

    states = []
    actions = []
    dones = []
    rewards = []
    n_envs = len(env._envs)
    state = env.reset()

    current_step = 0
    
    for t in range(horizon):
        true_expert_action = env.opt_action(state)
        sampled_env_expert_action = np.array([
            sampled_envs[i, current_step].opt_action(state[i])
            for i in range(n_envs)
        ])
        terminal_flag = current_step >= sample_termination_step
        action = np.where(terminal_flag.reshape(-1,1), true_expert_action, sampled_env_expert_action)

        next_state, reward, done, _ = env.step(action)

        states.append(state)
        actions.append(action)
        dones.append(done)
        rewards.append(reward)
        state = next_state

        if np.any(done):
            state = env.reset(reset_free=True)

    states = np.stack(states, axis=1)
    actions = np.stack(actions, axis=1)
    # next_states = np.stack(next_states, axis=1)
    rewards = np.stack(rewards, axis=1)
    dones = np.stack(dones, axis=1)
    assert states.shape == (n_envs, horizon, env.state_dim)
    assert actions.shape == (n_envs, horizon, env.action_dim)
    # assert next_states.shape == (n_envs, horizon, env.state_dim)
    assert rewards.shape == (n_envs, horizon)
    return states, actions, rewards, dones

trajs = []
import tqdm
for env in tqdm.tqdm(train_envs, desc="Collecting dagger data"):
    states, actions, rewards, dones = get_rollout(env, 20*10)
    n_envs = len(env._envs)
    for k in range(n_envs):
        traj_k = {
            "states": states[k],
            "actions": actions[k],
            "rewards": rewards[k],
            "dones": dones[k],
        }
        trajs.append(traj_k)

class SequenceDataset(torch.utils.data.Dataset):
    """Dataset class for sequence data."""

    def __init__(self, trajs, config):
        self.shuffle = config['shuffle']
        self.horizon = config['horizon']
        self.store_gpu = config['store_gpu']
        self.config = config
        self.trajs = trajs
    
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.trajs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # breakpoint()
        res = {
            'states': convert_to_tensor(self.trajs[index]['states'], store_gpu=self.store_gpu).reshape(200, 2),
            'actions': convert_to_tensor(self.trajs[index]['actions'], store_gpu=self.store_gpu).reshape(200, 21),
            'rewards': convert_to_tensor(self.trajs[index]['rewards'], store_gpu=self.store_gpu).reshape(200),
            'dones': convert_to_tensor(self.trajs[index]['dones'], store_gpu=self.store_gpu).reshape(200),
        }
        return res
        # if self.shuffle:
        #     perm = torch.randperm(self.horizon)
        #     res['states'] = res['states'][perm]
        #     res['actions'] = res['actions'][perm]
        #     res['rewards'] = res['rewards'][perm]
        #     res['dones'] = res['dones'][perm]
    
config = {"horizon": 200, "store_gpu": False, "state_dim": train_envs[0].state_dim, "action_dim": train_envs[0].action_dim}
train_dataset = SequenceDataset(trajs, {**config, "shuffle": True})

with open('ordered_dagger_data_sample_term2.pkl', 'wb') as f:
    pickle.dump(train_dataset, f)

# ---
with open('ordered_dagger_data_sample_term2.pkl', 'rb') as f:
    train_dataset = pickle.load(f)
# breakpoint()
train_dataset = SequenceDataset(train_dataset.trajs, train_dataset.config)
import os
from models import get_model

save_dir = "ordered_data_sample_term2"
model_args = {
    'horizon': 200,
    'state_dim': 2,
    'action_dim': 21,
    'n_layer': 4,
    'n_embd': 128,
    'n_head': 4,
    'shuffle': True,
    'dropout': 0.1,
    'test': False,
    'store_gpu': True,
    'continuous_action': False,
    'gmm_heads': 1,
    'tanh_action': False,
    'low_noise_eval': False
}
os.makedirs(save_dir, exist_ok=True)
with open(os.path.join(save_dir, "model_args.pkl"), "wb") as f:
    pickle.dump(model_args, f)

# from models import DecisionTransformer
# model = get_model(**model_args).to(device)
model = DecisionTransformer(model_args).to(device)

params = {
        "batch_size": 256,
        "shuffle": True,
        "prefetch_factor": 32,
        "num_workers": 10,
    }


step_id=0
    # model,
train_loader=torch.utils.data.DataLoader(train_dataset, **params)
    # test_loader,
    # save_dir,
    # args,
    # device,
continuous_action=False
action_dim=21

os.makedirs(save_dir, exist_ok=True)

optimizer = torch.optim.AdamW(
    model.parameters(), lr=1e-3, weight_decay=1e-4
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
    range(1, 1000 + 1), desc=f"Training Dagger Step {step_id}"
):

    model.train()
    train_loss = []
    for _, batch in enumerate(train_loader):
    # tqdm.tqdm(
    #     enumerate(train_loader), desc=f"Training Epoch {epoch}",
    #     total=len(train_loader)
    # ):
        batch = {k: v.to(device) for k, v in batch.items()}
        true_actions = batch["actions"]
        pred_actions, pred_values = model(batch)
        action_loss = get_loss(pred_actions, true_actions)
        loss = action_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
    train_loss = np.mean(train_loss)
    # if args.log_wandb:
    #     wandb.log({f"dagger-{step_id}/train_loss": train_loss})
    print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}")

    if epoch % 50 == 0:
        torch.save(
            model.state_dict(), os.path.join(save_dir, f"model_epoch_{epoch}.pth")
        )
# # Save final model
torch.save(
    model.state_dict(), os.path.join(save_dir, "model_final.pth")
)

    # return model