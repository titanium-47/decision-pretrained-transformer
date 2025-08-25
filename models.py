import torch
import torch.nn as nn
import transformers
transformers.set_seed(0)
from transformers import GPT2Config, GPT2Model
from IPython import embed
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_model(model_type, horizon, state_dim, action_dim, continuous_action):
    n_embd = 32
    n_head = 4
    n_layer = 4
    dropout = 0
    shuffle = True
    config = {
        'horizon': horizon,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'n_layer': n_layer,
        'n_embd': n_embd,
        'n_head': n_head,
        'shuffle': shuffle,
        'dropout': dropout,
        'test': False,
        'store_gpu': True,
        'continuous_action': continuous_action,
    }
    if model_type == "decision_transformer":
        model = DecisionTransformer(config).to(device)
    elif model_type == "transformer":
        model = Transformer(config).to(device)
    elif model_type == "mlp":
        model = MLP(config).to(device)
    else:
        raise ValueError(f"Unknown model name: {model_type}")
    return model

class Transformer(nn.Module):
    """Transformer class."""

    def __init__(self, config):
        super(Transformer, self).__init__()

        self.config = config
        self.test = config['test']
        self.horizon = self.config['horizon']
        self.n_embd = self.config['n_embd']
        self.n_layer = self.config['n_layer']
        self.n_head = self.config['n_head']
        self.state_dim = self.config['state_dim']
        self.action_dim = self.config['action_dim']
        self.dropout = self.config['dropout']

        config = GPT2Config(
            n_positions=4 * (1 + self.horizon),
            n_embd=self.n_embd,
            n_layer=self.n_layer,
            n_head=1,
            resid_pdrop=self.dropout,
            embd_pdrop=self.dropout,
            attn_pdrop=self.dropout,
            use_cache=False,
        )
        self.transformer = GPT2Model(config)

        self.embed_transition = nn.Linear(
            2 * self.state_dim + self.action_dim + 1, self.n_embd)
        self.continuous_action = self.config['continuous_action']
        if self.continuous_action:
            self.pred_action_means = nn.Linear(self.n_embd, self.action_dim)
            self.pred_action_log_stds = nn.Linear(self.n_embd, self.action_dim)
        else:
            self.pred_actions = nn.Linear(self.n_embd, self.action_dim)

    def forward(self, x):
        query_states = x['query_states'][:, None, :]
        zeros = x['zeros'][:, None, :]

        state_seq = torch.cat([query_states, x['context_states']], dim=1)
        action_seq = torch.cat(
            [zeros[:, :, :self.action_dim], x['context_actions']], dim=1)
        next_state_seq = torch.cat(
            [zeros[:, :, :self.state_dim], x['context_next_states']], dim=1)
        reward_seq = torch.cat([zeros[:, :, :1], x['context_rewards']], dim=1)

        seq = torch.cat(
            [state_seq, action_seq, next_state_seq, reward_seq], dim=2)
        stacked_inputs = self.embed_transition(seq)
        transformer_outputs = self.transformer(inputs_embeds=stacked_inputs)
        if self.continuous_action:
            action_means = self.pred_action_means(
                transformer_outputs['last_hidden_state'])
            action_log_stds = self.pred_action_log_stds(
                transformer_outputs['last_hidden_state'])
            if self.test:
                action_means = action_means[:, -1, :]
                action_log_stds = action_log_stds[:, -1, :]
            dist = torch.distributions.Normal(action_means, action_log_stds.exp())
            return dist
        else:
            preds = self.pred_actions(transformer_outputs['last_hidden_state'])

            if self.test:
                return preds[:, -1, :]
            return preds[:, 1:, :]


class ImageTransformer(Transformer):
    """Transformer class for image-based data."""

    def __init__(self, config):
        super().__init__(config)
        self.im_embd = 8

        size = self.config['image_size']
        size = (size - 3) // 2 + 1
        size = (size - 3) // 1 + 1

        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Flatten(start_dim=1),
            nn.Linear(int(16 * size * size), self.im_embd),
            nn.ReLU(),
        )

        new_dim = self.im_embd + self.state_dim + self.action_dim + 1
        self.embed_transition = torch.nn.Linear(new_dim, self.n_embd)
        self.embed_ln = nn.LayerNorm(self.n_embd)

    def forward(self, x):
        query_images = x['query_images'][:, None, :]
        query_states = x['query_states'][:, None, :]
        context_images = x['context_images']
        context_states = x['context_states']
        context_actions = x['context_actions']
        context_rewards = x['context_rewards']

        if len(context_rewards.shape) == 2:
            context_rewards = context_rewards[:, :, None]

        batch_size = query_states.shape[0]

        image_seq = torch.cat([query_images, context_images], dim=1)
        image_seq = image_seq.view(-1, *image_seq.size()[2:])

        image_enc_seq = self.image_encoder(image_seq)
        image_enc_seq = image_enc_seq.view(batch_size, -1, self.im_embd)

        context_states = torch.cat([query_states, context_states], dim=1)
        context_actions = torch.cat([
            torch.zeros(batch_size, 1, self.action_dim).to(device),
            context_actions,
        ], dim=1)
        context_rewards = torch.cat([
            torch.zeros(batch_size, 1, 1).to(device),
            context_rewards,
        ], dim=1)

        stacked_inputs = torch.cat([
            image_enc_seq,
            context_states,
            context_actions,
            context_rewards,
        ], dim=2)
        stacked_inputs = self.embed_transition(stacked_inputs)
        stacked_inputs = self.embed_ln(stacked_inputs)

        transformer_outputs = self.transformer(inputs_embeds=stacked_inputs)
        preds = self.pred_actions(transformer_outputs['last_hidden_state'])

        if self.test:
            return preds[:, -1, :]
        return preds[:, 1:, :]


class MLP(nn.Module):
    """MLP class."""

    def __init__(self, config):
        super(MLP, self).__init__()

        self.config = config
        self.horizon = config['horizon']
        self.state_dim = config['state_dim']
        self.action_dim = config['action_dim']
        self.model = nn.Sequential(
            nn.Linear(self.state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        if self.config['continuous_action']:
            self.pred_action_means = nn.Linear(256, self.action_dim)
            self.pred_action_log_stds = nn.Linear(256, self.action_dim)
        else:
            self.pred_actions = nn.Linear(256, self.action_dim)
    
    def forward(self, x, sample_time=False):
        assert not sample_time, "MLP does not support sampling time."
        query_states = x
        query_states = query_states.view(-1, self.state_dim)
        query_states = query_states.to(device)
        if self.config['continuous_action']:
            action_means = self.pred_action_means(self.model(query_states))
            action_log_stds = self.pred_action_log_stds(self.model(query_states))
            dist = torch.distributions.Normal(action_means, action_log_stds.exp())
            return dist
        else:
            pred_actions = self.model(query_states)
            return pred_actions

class DecisionTransformer(nn.Module):
    """Decision Transformer class."""

    def __init__(self, config):
        super(DecisionTransformer, self).__init__()

        self.config = config
        self.test = config['test']
        self.horizon = self.config['horizon']
        self.n_embd = self.config['n_embd']
        self.n_layer = self.config['n_layer']
        self.n_head = self.config['n_head']
        self.state_dim = self.config['state_dim']
        self.action_dim = self.config['action_dim']
        self.dropout = self.config['dropout']

        config = GPT2Config(
            n_positions=4 * (1 + self.horizon),
            n_embd=self.n_embd,
            n_layer=self.n_layer,
            n_head=1,
            resid_pdrop=self.dropout,
            embd_pdrop=self.dropout,
            attn_pdrop=self.dropout,
            use_cache=False,
        )
        self.transformer = GPT2Model(config)

        self.embed_timestep = nn.Embedding(self.horizon, self.n_embd)
        self.embed_transition = nn.Linear(
            self.state_dim + self.action_dim + 2, self.n_embd)
        self.continuous_action = self.config['continuous_action']
        if self.continuous_action:
            self.pred_action_means = nn.Linear(self.n_embd, self.action_dim)
            self.pred_action_log_stds = nn.Linear(self.n_embd, self.action_dim)
        else:
            self.pred_actions = nn.Linear(self.n_embd, self.action_dim)

    def forward(self, x, sample_time=False):
        states = x['states']
        actions = x['actions']
        rewards = x['rewards']
        dones = x['dones']
        # print("Shapes:")
        # print(f"States: {states.shape}, Actions: {actions.shape}, Rewards: {rewards.shape}, Dones: {dones.shape}")
        # print("Pad:", torch.zeros(states.shape[0], 1, self.action_dim).shape, actions[:, :-1, :].shape)
        input_actions = torch.cat([
            torch.zeros(states.shape[0], 1, self.action_dim).to(device),
            actions[:, :-1, :],
        ], dim=1)
        input_rewards = torch.cat([
            torch.zeros(states.shape[0], 1).to(device),
            rewards[:, :-1],
        ], dim=1)
        input_dones = torch.cat([
            torch.zeros(states.shape[0], 1).to(device),
            dones[:, :-1],
        ], dim=1)
        # print(f"Shapes: states: {states.shape}, input_actions: {input_actions.shape}, rewards: {rewards.shape}, dones: {dones.shape}")
        inputs = torch.cat([states, input_actions, input_rewards.unsqueeze(-1), input_dones.unsqueeze(-1)], dim=2)
        # inputs = torch.cat([states, input_actions, rewards.unsqueeze(-1), dones.unsqueeze(-1)], dim=2)
        timesteps = torch.arange(
            inputs.shape[1], device=inputs.device).unsqueeze(0).expand(
                inputs.shape[0], -1)
        if sample_time:
            assert self.horizon > inputs.shape[1], "Horizon must be greater than the input sequence length."
            initial_timestep = torch.randint(0, self.horizon - inputs.shape[1], (inputs.shape[0],), device=inputs.device)
            timesteps = timesteps + initial_timestep.unsqueeze(1)

        timestep_embeds = self.embed_timestep(timesteps)
        inputs = self.embed_transition(inputs)
        inputs = inputs + timestep_embeds

        transformer_outputs = self.transformer(inputs_embeds=inputs)
        if self.continuous_action:
            action_means = self.pred_action_means(
                transformer_outputs['last_hidden_state'])
            action_log_stds = self.pred_action_log_stds(
                transformer_outputs['last_hidden_state'])
            dist = torch.distributions.Normal(action_means, action_log_stds.exp())
            return dist
        preds = self.pred_actions(transformer_outputs['last_hidden_state']) # B x T x A
        return preds
    
    def get_action(self, current_state, states, actions, rewards, dones):
        # return self.debug_mlp(current_state)  # B x D -> B x A
        if states is None: # current_state is B x D
            input_states = current_state.unsqueeze(1)
            input_actions = torch.zeros(current_state.shape[0], 1, self.action_dim).to(device)
            input_rewards = torch.zeros(current_state.shape[0], 1).to(device)
            input_dones = torch.zeros(current_state.shape[0], 1).to(device)
        else: # states is B x T x D, actions is B x T x A, rewards is B x T, dones is B x T, current_state is B x D
            input_states = torch.cat([
                states, current_state.unsqueeze(1)
            ], dim=1) # B x (T+1) x D
            input_actions = torch.cat([
                torch.zeros(states.shape[0], 1, self.action_dim).to(device),
                actions,
            ], dim=1) # B x (T+1) x A
            input_rewards = torch.cat([
                torch.zeros(states.shape[0], 1).to(device),
                rewards,
            ], dim=1) # B x (T+1)
            input_dones = torch.cat([
                torch.zeros(states.shape[0], 1).to(device),
                dones,
            ], dim=1) # B x (T+1)

            input_states = input_states[:, -self.horizon:, :]  # Keep only the last horizon states
            input_actions = input_actions[:, -self.horizon:, :]
            input_rewards = input_rewards[:, -self.horizon:]
            input_dones = input_dones[:, -self.horizon:]
        
        timesteps = torch.arange(
            input_states.shape[1], device=input_states.device).unsqueeze(0).expand(
                input_states.shape[0], -1)
        timestep_embeds = self.embed_timestep(timesteps)
        # print(f"Shapes: input_states: {input_states.shape}, input_actions: {input_actions.shape}, rewards: {input_rewards.shape}, dones: {input_dones.shape}")
        inputs = torch.cat([input_states, input_actions, input_rewards.unsqueeze(-1), input_dones.unsqueeze(-1)], dim=2)
        inputs = self.embed_transition(inputs)
        inputs = inputs + timestep_embeds
        transformer_outputs = self.transformer(inputs_embeds=inputs)
        if self.continuous_action:
            action_means = self.pred_action_means(
                transformer_outputs['last_hidden_state'])[:, -1, :]
            action_log_stds = self.pred_action_log_stds(
                transformer_outputs['last_hidden_state'])[:, -1, :]
            dist = torch.distributions.Normal(action_means, action_log_stds.exp())
            return dist.sample()
        preds = self.pred_actions(transformer_outputs['last_hidden_state']) # B x (T+1) x A
        return preds[:, -1, :]  # Return the last action
