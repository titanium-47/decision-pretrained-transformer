# varibad_recurrentppo_sb3.py

from __future__ import annotations

from copy import deepcopy
from typing import Any, ClassVar, NamedTuple, Optional, Tuple

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces

from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from sb3_contrib.common.recurrent.buffers import RecurrentRolloutBuffer
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.utils import explained_variance, obs_as_tensor
from stable_baselines3.common.torch_layers import FlattenExtractor


# -------------------------
# Helpers
# -------------------------

def kl_diag_gaussians(mu_q: th.Tensor, logvar_q: th.Tensor, mu_p: th.Tensor, logvar_p: th.Tensor) -> th.Tensor:
    """
    KL(q||p) for diagonal Gaussians.
    Shapes: (B, D) -> (B,)
    """
    var_q = th.exp(logvar_q)
    var_p = th.exp(logvar_p)
    return 0.5 * (
        (logvar_p - logvar_q) +
        (var_q + (mu_q - mu_p) ** 2) / (var_p + 1e-8) -
        1.0
    ).sum(dim=-1)


def one_hot(actions: th.Tensor, n: int) -> th.Tensor:
    if actions.ndim == 2 and actions.shape[1] == 1:
        actions = actions.squeeze(1)
    return F.one_hot(actions.long(), num_classes=n).float()


# -------------------------
# Buffer samples
# -------------------------

class VariBADRolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    lstm_states: Any
    episode_starts: th.Tensor
    mask: th.Tensor
    next_observations: th.Tensor

    latent_z: th.Tensor              # sampled z used for PPO step
    prior_mean: th.Tensor            # prior params for KL (stored at rollout)
    prior_logvar: th.Tensor
    post_mean: th.Tensor             # posterior params from rollout (optional logging / debugging)
    post_logvar: th.Tensor
    unnorm_rewards: th.Tensor        # unnormalized rewards for VAE decoder training


# -------------------------
# Rollout buffer
# -------------------------

class VariBADRolloutBuffer(RecurrentRolloutBuffer):
    """
    Extends SB3 RecurrentRolloutBuffer with:
      - next_observations
      - unnorm_rewards (for VAE decoder - original environment rewards)
      - latent_z (sampled)
      - prior params (mean/logvar) and posterior params (mean/logvar)
    
    Note: The parent class stores normalized rewards for PPO advantage computation.
          We store unnormalized rewards separately for VAE reward decoder training.
    """
    def __init__(self, *args, latent_dim: int, **kwargs):
        self.latent_dim = latent_dim
        super().__init__(*args, **kwargs)

    def reset(self) -> None:
        super().reset()
        self.next_observations = np.zeros_like(self.observations)
        # Unnormalized rewards for VAE decoder (predicts original env rewards)
        self.unnorm_rewards_ = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        self.latent_z = np.zeros((self.buffer_size, self.n_envs, self.latent_dim), dtype=np.float32)
        self.prior_mean = np.zeros((self.buffer_size, self.n_envs, self.latent_dim), dtype=np.float32)
        self.prior_logvar = np.zeros((self.buffer_size, self.n_envs, self.latent_dim), dtype=np.float32)
        self.post_mean = np.zeros((self.buffer_size, self.n_envs, self.latent_dim), dtype=np.float32)
        self.post_logvar = np.zeros((self.buffer_size, self.n_envs, self.latent_dim), dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
        *,
        lstm_states=None,
        next_obs: np.ndarray,
        unnorm_reward: np.ndarray,
        latent_z: np.ndarray,
        prior_mean: np.ndarray,
        prior_logvar: np.ndarray,
        post_mean: np.ndarray,
        post_logvar: np.ndarray,
    ) -> None:
        self.next_observations[self.pos] = next_obs
        # Store unnormalized rewards for VAE decoder
        self.unnorm_rewards_[self.pos] = unnorm_reward.astype(np.float32)

        self.latent_z[self.pos] = latent_z.astype(np.float32)
        self.prior_mean[self.pos] = prior_mean.astype(np.float32)
        self.prior_logvar[self.pos] = prior_logvar.astype(np.float32)
        self.post_mean[self.pos] = post_mean.astype(np.float32)
        self.post_logvar[self.pos] = post_logvar.astype(np.float32)

        # Parent stores normalized reward for PPO advantage computation
        super().add(obs, action, reward, episode_start, value, log_prob, lstm_states=lstm_states)

    def _get_samples(self, batch_inds, env_change, env=None):
        base = super()._get_samples(batch_inds, env_change, env=env)
        padded_batch_size = base.observations.shape[0]

        next_obs_flat = self.next_observations.reshape(-1, *self.obs_shape)
        next_obs = self.pad(next_obs_flat[batch_inds]).reshape((padded_batch_size, *self.obs_shape))

        # Unnormalized rewards for VAE decoder
        unnorm_rewards_flat = self.unnorm_rewards_.reshape(-1)
        unnorm_rewards = self.pad(unnorm_rewards_flat[batch_inds]).reshape((padded_batch_size,))

        z_flat = self.latent_z.reshape(-1, self.latent_dim)
        z = self.pad(z_flat[batch_inds]).reshape((padded_batch_size, self.latent_dim))

        pm = self.prior_mean.reshape(-1, self.latent_dim)
        plv = self.prior_logvar.reshape(-1, self.latent_dim)
        prior_mean = self.pad(pm[batch_inds]).reshape((padded_batch_size, self.latent_dim))
        prior_logvar = self.pad(plv[batch_inds]).reshape((padded_batch_size, self.latent_dim))

        qm = self.post_mean.reshape(-1, self.latent_dim)
        qlv = self.post_logvar.reshape(-1, self.latent_dim)
        post_mean = self.pad(qm[batch_inds]).reshape((padded_batch_size, self.latent_dim))
        post_logvar = self.pad(qlv[batch_inds]).reshape((padded_batch_size, self.latent_dim))

        return VariBADRolloutBufferSamples(
            observations=base.observations,
            actions=base.actions,
            old_values=base.old_values,
            old_log_prob=base.old_log_prob,
            advantages=base.advantages,
            returns=base.returns,
            lstm_states=base.lstm_states,
            episode_starts=base.episode_starts,
            mask=base.mask,
            next_observations=next_obs,
            latent_z=z,
            prior_mean=prior_mean,
            prior_logvar=prior_logvar,
            post_mean=post_mean,
            post_logvar=post_logvar,
            unnorm_rewards=unnorm_rewards,  # unnormalized rewards for VAE
        )


# -------------------------
# Policy
# -------------------------

class VariBADPolicy(RecurrentActorCriticPolicy):
    """
    Encoder = recurrent module (SB3 lstm_actor / lstm_critic optional)
    Policy/value = MLP conditioned on [obs, z]
    
    Supports binary (sigmoid+BCE) or multiclass (softmax+CE) reward prediction.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        net_arch=None,
        activation_fn=th.nn.Tanh,
        ortho_init=True,
        use_sde=False,
        log_std_init=0.0,
        full_std=True,
        use_expln=False,
        squash_output=False,
        features_extractor_class=FlattenExtractor,
        features_extractor_kwargs=None,
        share_features_extractor=True,
        normalize_images=True,
        optimizer_class=th.optim.Adam,
        optimizer_kwargs=None,
        lstm_hidden_size: int = 64,       # VariBAD: encoder_gru_hidden_size=64
        n_lstm_layers: int = 1,
        shared_lstm: bool = False,
        enable_critic_lstm: bool = True,
        lstm_kwargs=None,
        latent_dim: int = 32,             # VariBAD: latent_dim=32
        decoder_hidden: int = 32,         # VariBAD: decoder layers [32, 32]
        reward_type: str = "binary",      # "binary" or "multiclass"
        num_reward_classes: int = 2,      # for multiclass (e.g., 3 for keydoor-markovian)
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
            use_sde=use_sde,
            log_std_init=log_std_init,
            full_std=full_std,
            use_expln=use_expln,
            squash_output=squash_output,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            share_features_extractor=share_features_extractor,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            lstm_hidden_size=lstm_hidden_size,
            n_lstm_layers=n_lstm_layers,
            shared_lstm=shared_lstm,
            enable_critic_lstm=enable_critic_lstm,
            lstm_kwargs=lstm_kwargs,
        )

        assert isinstance(self.action_space, spaces.Discrete), "This minimal implementation assumes discrete actions."
        assert len(self.observation_space.shape) == 1, "Use flattened obs for minimal baseline."

        self.latent_dim = latent_dim
        self.obs_dim = int(np.prod(self.observation_space.shape))
        self.n_actions = int(self.action_space.n)
        self.reward_type = reward_type
        self.num_reward_classes = num_reward_classes

        # Posterior params from encoder hidden (actor LSTM output)
        self.posterior_head = nn.Linear(lstm_hidden_size, 2 * self.latent_dim)

        # MLP policy/value conditioned on [obs, z]
        pi_latent_dim = self.mlp_extractor.latent_dim_pi
        vf_latent_dim = self.mlp_extractor.latent_dim_vf

        self.pi_cond = nn.Sequential(
            nn.Linear(self.obs_dim + self.latent_dim, decoder_hidden),
            activation_fn(),
            nn.Linear(decoder_hidden, pi_latent_dim),
            activation_fn(),
        )
        self.vf_cond = nn.Sequential(
            nn.Linear(self.obs_dim + self.latent_dim, decoder_hidden),
            activation_fn(),
            nn.Linear(decoder_hidden, vf_latent_dim),
            activation_fn(),
        )

        # Reward decoder: binary (sigmoid+BCE) or multiclass (softmax+CE)
        # Predicts UNNORMALIZED rewards from environment
        reward_decoder_input_dim = self.latent_dim + self.obs_dim + self.n_actions + self.obs_dim
        if reward_type == "binary":
            # Binary classification: outputs probability via sigmoid
            self.reward_decoder = nn.Sequential(
                nn.Linear(reward_decoder_input_dim, decoder_hidden),
                activation_fn(),
                nn.Linear(decoder_hidden, decoder_hidden),
                activation_fn(),
                nn.Linear(decoder_hidden, 1),
                nn.Sigmoid(),
            )
        else:
            # Multiclass classification: outputs logits for num_reward_classes
            self.reward_decoder = nn.Sequential(
                nn.Linear(reward_decoder_input_dim, decoder_hidden),
                activation_fn(),
                nn.Linear(decoder_hidden, decoder_hidden),
                activation_fn(),
                nn.Linear(decoder_hidden, num_reward_classes),
            )
        
        # State decoder (predicts next_obs, MSE loss)
        self.state_decoder = nn.Sequential(
            nn.Linear(self.latent_dim + self.obs_dim + self.n_actions, decoder_hidden),
            activation_fn(),
            nn.Linear(decoder_hidden, decoder_hidden),
            activation_fn(),
            nn.Linear(decoder_hidden, self.obs_dim),
        )

        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    @staticmethod
    def _sample_gaussian(mu: th.Tensor, logvar: th.Tensor) -> th.Tensor:
        std = th.exp(0.5 * logvar)
        eps = th.randn_like(std)
        return mu + eps * std

    def posterior_params(self, obs: th.Tensor, lstm_states, episode_starts: th.Tensor) -> Tuple[th.Tensor, th.Tensor, Any]:
        """
        Returns posterior params (mu, logvar) and updated actor LSTM state.
        """
        features = self.extract_features(obs)
        pi_features = features if self.share_features_extractor else features[0]
        pi_seq, new_pi_state = self._process_sequence(pi_features, lstm_states.pi, episode_starts, self.lstm_actor)
        stats = self.posterior_head(pi_seq)
        mu = stats[..., : self.latent_dim]
        logvar = stats[..., self.latent_dim :]
        return mu, logvar, new_pi_state

    def _dist_value_from_z(self, obs: th.Tensor, z: th.Tensor) -> Tuple[Any, th.Tensor]:
        pi_in = th.cat([obs, z], dim=-1)
        vf_in = th.cat([obs, z], dim=-1)
        latent_pi = self.pi_cond(pi_in)
        latent_vf = self.vf_cond(vf_in)
        dist = self._get_action_dist_from_latent(latent_pi)
        value = self.value_net(latent_vf)
        return dist, value

    # --- rollout forward: sample z, but do not backprop through PPO ---
    def forward(self, obs: th.Tensor, lstm_states, episode_starts: th.Tensor):
        mu, logvar, new_pi_state = self.posterior_params(obs, lstm_states, episode_starts)
        z = self._sample_gaussian(mu, logvar).detach()

        dist, value = self._dist_value_from_z(obs, z)
        actions = dist.get_actions(deterministic=False)
        log_prob = dist.log_prob(actions)

        # critic lstm state passthrough (we won't use critic-lstm here; keep SB3 state contract)
        new_states = type(lstm_states)(pi=new_pi_state, vf=lstm_states.vf)
        return actions, value, log_prob, new_states

    # --- PPO eval with fixed z (from buffer) ---
    def evaluate_actions_with_z(
        self,
        obs: th.Tensor,
        actions: th.Tensor,
        z: th.Tensor,
    ):
        dist, value = self._dist_value_from_z(obs, z)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return value, log_prob, entropy

    # keep SB3 API (unused in our algo's train, but required)
    def evaluate_actions(self, obs, actions, lstm_states, episode_starts):
        mu, logvar, _ = self.posterior_params(obs, lstm_states, episode_starts)
        z = mu.detach()
        return self.evaluate_actions_with_z(obs, actions, z)

    # --- VAE losses: grads flow to encoder+decoders ---
    def vae_losses(
        self,
        obs: th.Tensor,
        actions: th.Tensor,
        unnorm_rewards: th.Tensor,  # UNNORMALIZED rewards from environment
        next_obs: th.Tensor,
        lstm_states,
        episode_starts: th.Tensor,
        mask: th.Tensor,
        prior_mean: th.Tensor,
        prior_logvar: th.Tensor,
    ):
        mu, logvar, _ = self.posterior_params(obs, lstm_states, episode_starts)
        z = self._sample_gaussian(mu, logvar)  # reparam => encoder grads

        a_oh = one_hot(actions, self.n_actions)

        # Reward decoder - predicts UNNORMALIZED rewards
        r_in = th.cat([z, obs, a_oh, next_obs], dim=-1)
        
        if self.reward_type == "binary":
            # Binary classification: sigmoid already applied in decoder
            # BCE loss: rewards are 0 or 1
            r_pred = self.reward_decoder(r_in).squeeze(-1)
            reward_targets = unnorm_rewards.clamp(0, 1)
            reward_loss = F.binary_cross_entropy(r_pred, reward_targets, reduction='none')[mask].mean()
        else:
            # Multiclass classification: softmax + CE
            # Rewards are class indices (0, 1, 2, ...)
            r_logits = self.reward_decoder(r_in)  # (B, num_classes)
            reward_targets = unnorm_rewards.long().clamp(0, self.num_reward_classes - 1)
            reward_loss = F.cross_entropy(r_logits, reward_targets, reduction='none')[mask].mean()

        # State decoder (predicts next_obs, MSE loss)
        s_in = th.cat([z, obs, a_oh], dim=-1)
        s_pred = self.state_decoder(s_in)
        state_loss = ((s_pred - next_obs) ** 2).sum(dim=-1)[mask].mean()

        # KL(q||p) with priors from rollout buffer
        kl = kl_diag_gaussians(mu, logvar, prior_mean, prior_logvar)
        kl_loss = kl[mask].mean()

        return reward_loss, state_loss, kl_loss, mu, logvar


# -------------------------
# Algorithm
# -------------------------

class VariBADPPO(RecurrentPPO):
    """
    Minimal modifications on top of sb3_contrib.RecurrentPPO:
      - custom buffer with z + (s', r) + KL priors
      - rollout stores sampled z, posterior params, and prior params
      - train uses PPO conditioned on stored z, plus VAE losses updating encoder/decoders
    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "VariBADPolicy": VariBADPolicy,
    }

    def __init__(
        self,
        *args,
        latent_dim: int = 5,
        kl_coef: float = 1.0,
        reward_coef: float = 1.0,
        state_coef: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.latent_dim = latent_dim
        self.kl_coef = kl_coef
        self.reward_coef = reward_coef
        self.state_coef = state_coef

    def _setup_model(self) -> None:
        super()._setup_model()

        lstm = self.policy.lstm_actor
        hidden_state_buffer_shape = (self.n_steps, lstm.num_layers, self.n_envs, lstm.hidden_size)

        self.rollout_buffer = VariBADRolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            hidden_state_buffer_shape,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            latent_dim=self.policy.latent_dim,
        )

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        assert self._last_obs is not None
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        callback.on_rollout_start()

        lstm_states = deepcopy(self._last_lstm_states)

        # rolling prior for KL:
        # p(z_0)=N(0,1) at episode_start; else p(z_t)=q(z_{t-1}) from rollout-time posterior params
        prior_mean = np.zeros((env.num_envs, self.latent_dim), dtype=np.float32)
        prior_logvar = np.zeros((env.num_envs, self.latent_dim), dtype=np.float32)
        
        # Check if env is wrapped with VecNormalize (for getting unnormalized rewards)
        from stable_baselines3.common.vec_env import VecNormalize
        is_vec_normalize = isinstance(env, VecNormalize)

        while n_steps < n_rollout_steps:
            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                episode_starts = th.tensor(self._last_episode_starts, dtype=th.float32, device=self.device)

                # posterior (no_grad for rollout)
                mu, logvar, new_pi_state = self.policy.posterior_params(obs_tensor, lstm_states, episode_starts)
                z = self.policy._sample_gaussian(mu, logvar)

                # PPO uses z sampled at rollout time (store it)
                dist, value = self.policy._dist_value_from_z(obs_tensor, z)
                actions = dist.get_actions(deterministic=False)
                log_prob = dist.log_prob(actions)

                mu_np = mu.cpu().numpy()
                logvar_np = logvar.cpu().numpy()
                z_np = z.cpu().numpy()

                new_lstm_states = type(lstm_states)(pi=new_pi_state, vf=lstm_states.vf)

            actions_np = actions.cpu().numpy()

            # discrete => (n_env,1) for buffer
            actions_buf = actions_np.reshape(-1, 1)

            new_obs, rewards, dones, infos = env.step(actions_np)
            self.num_timesteps += env.num_envs
            
            # Get unnormalized rewards for VAE decoder training
            # VecNormalize stores original rewards in old_reward attribute after step
            if is_vec_normalize:
                unnorm_rewards = env.get_original_reward()
            else:
                unnorm_rewards = rewards

            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            rollout_buffer.add(
                self._last_obs,
                actions_buf,
                rewards,  # normalized rewards for PPO
                self._last_episode_starts,
                value,
                log_prob,
                lstm_states=self._last_lstm_states,
                next_obs=new_obs,
                unnorm_reward=unnorm_rewards,  # unnormalized rewards for VAE
                latent_z=z_np,
                prior_mean=prior_mean.copy(),
                prior_logvar=prior_logvar.copy(),
                post_mean=mu_np,
                post_logvar=logvar_np,
            )

            # update priors for next step: p_{t+1} := q_t
            prior_mean = mu_np
            prior_logvar = logvar_np

            # reset KL prior for new episodes
            if np.any(dones):
                prior_mean[dones] = 0.0
                prior_logvar[dones] = 0.0

            self._last_obs = new_obs
            self._last_episode_starts = dones
            self._last_lstm_states = new_lstm_states
            lstm_states = new_lstm_states

        with th.no_grad():
            episode_starts = th.tensor(dones, dtype=th.float32, device=self.device)
            # value baseline at final obs uses posterior mean (doesn't matter much; consistent)
            obs_tensor = obs_as_tensor(new_obs, self.device)
            mu, logvar, _ = self.policy.posterior_params(obs_tensor, lstm_states, episode_starts)
            z = mu
            _, values = self.policy._dist_value_from_z(obs_tensor, z)

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        callback.on_rollout_end()
        return True

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        clip_range = self.clip_range(self._current_progress_remaining)
        clip_range_vf = self.clip_range_vf(self._current_progress_remaining) if self.clip_range_vf is not None else None

        entropy_losses, pg_losses, value_losses, clip_fractions = [], [], [], []
        vae_rew_losses, vae_state_losses, vae_kl_losses = [], [], []

        continue_training = True

        for epoch in range(self.n_epochs):
            approx_kl_divs = []

            for rollout_data in self.rollout_buffer.get(self.batch_size):
                mask = rollout_data.mask > 1e-8

                actions = rollout_data.actions.long()  # (B,1)
                actions_flat = actions.flatten()       # (B,)

                # PPO terms with stored z (key fix)
                z = rollout_data.latent_z
                values, log_prob, entropy = self.policy.evaluate_actions_with_z(
                    rollout_data.observations,
                    actions_flat,
                    z,
                )
                values = values.flatten()

                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages[mask].mean()) / (advantages[mask].std() + 1e-8)

                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2)
                pg_loss = policy_loss[mask].mean()

                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()[mask]).item()
                clip_fractions.append(clip_fraction)

                if clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )

                value_loss = th.mean(((rollout_data.returns - values_pred) ** 2)[mask])

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob[mask])
                else:
                    entropy_loss = -th.mean(entropy[mask])

                # VAE losses (encoder+decoders)
                # Note: unnorm_rewards for VAE decoder, NOT normalized rewards
                rew_loss, state_loss, kl_loss, _, _ = self.policy.vae_losses(
                    obs=rollout_data.observations,
                    actions=rollout_data.actions,
                    unnorm_rewards=rollout_data.unnorm_rewards,  # UNNORMALIZED rewards for VAE
                    next_obs=rollout_data.next_observations,
                    lstm_states=rollout_data.lstm_states,
                    episode_starts=rollout_data.episode_starts,
                    mask=mask,
                    prior_mean=rollout_data.prior_mean,
                    prior_logvar=rollout_data.prior_logvar,
                )

                loss = (
                    pg_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                    + self.reward_coef * rew_loss
                    + self.state_coef * state_loss
                    + self.kl_coef * kl_loss
                )

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean(((th.exp(log_ratio) - 1) - log_ratio)[mask]).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

                pg_losses.append(pg_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                vae_rew_losses.append(rew_loss.item())
                vae_state_losses.append(state_loss.item())
                vae_kl_losses.append(kl_loss.item())

            if not continue_training:
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self.logger.record("train/entropy_loss", float(np.mean(entropy_losses)))
        self.logger.record("train/policy_gradient_loss", float(np.mean(pg_losses)))
        self.logger.record("train/value_loss", float(np.mean(value_losses)))
        self.logger.record("train/approx_kl", float(np.mean(approx_kl_divs)))
        self.logger.record("train/clip_fraction", float(np.mean(clip_fractions)))
        self.logger.record("train/explained_variance", float(explained_var))

        self.logger.record("vae/reward_recon_loss", float(np.mean(vae_rew_losses)))
        self.logger.record("vae/state_recon_loss", float(np.mean(vae_state_losses)))
        self.logger.record("vae/kl_loss", float(np.mean(vae_kl_losses)))

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", float(clip_range))
        if clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", float(clip_range_vf))
