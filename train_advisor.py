from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from copy import deepcopy
import numpy as np
from sb3_contrib.common.recurrent.buffers import RecurrentRolloutBuffer
import torch as th
from typing import Any, ClassVar, TypeVar, NamedTuple
from gymnasium import spaces
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.utils import explained_variance, obs_as_tensor
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.torch_layers import FlattenExtractor

class RecurrentRolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    lstm_states: Any
    episode_starts: th.Tensor
    mask: th.Tensor
    expert_actions: th.Tensor
    next_observations: th.Tensor


class AdvisorPolicy(RecurrentActorCriticPolicy):
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        net_arch = None,
        activation_fn = th.nn.Tanh,
        ortho_init = True,
        use_sde = False,
        log_std_init = 0.0,
        full_std = True,
        use_expln = False,
        squash_output = False,
        features_extractor_class = FlattenExtractor,
        features_extractor_kwargs = None,
        share_features_extractor = True,
        normalize_images = True,
        optimizer_class = th.optim.Adam,
        optimizer_kwargs = None,
        lstm_hidden_size = 256,
        n_lstm_layers = 1,
        shared_lstm = False,
        enable_critic_lstm = True,
        lstm_kwargs = None,
    ):

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            lstm_hidden_size=lstm_hidden_size,
            n_lstm_layers=n_lstm_layers,
            shared_lstm=shared_lstm,
            enable_critic_lstm=enable_critic_lstm,
            lstm_kwargs=lstm_kwargs,
        )
        self.aux_policy_net = deepcopy(self.mlp_extractor.policy_net)
        self.aux_action_net = deepcopy(self.action_net)
        # self.auxillary_log_std = deepcopy(self.log_std)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def forward_aux_expert(
        self,
        obs,
        lstm_states,
        episode_starts,
    ):
        features = self.extract_features(obs)
        aux_pi_features = features if self.share_features_extractor else features[0]
        latent_aux_pi, _ = self._process_sequence(aux_pi_features, lstm_states.pi, episode_starts, self.lstm_actor)
        
        latent_aux = self.aux_policy_net(latent_aux_pi)
        aux_logits = self.aux_action_net(latent_aux)
        return self.action_dist.proba_distribution(action_logits=aux_logits)

class MetaRecurrentRolloutBuffer(RecurrentRolloutBuffer):
    def reset(self):
        super().reset()
        self.next_observations = np.zeros_like(self.observations)
        self.expert_actions = np.zeros_like(self.actions)

    def add(self, *args, next_obs=None, expert_action=None, lstm_states=None, **kwargs):
        self.next_observations[self.pos] = next_obs
        self.expert_actions[self.pos] = expert_action.argmax(-1)[:, None]
        super().add(*args, lstm_states=lstm_states, **kwargs)

    def _get_samples(self, batch_inds, env_change, env=None):
        base = super()._get_samples(batch_inds, env_change, env=env)
        padded_batch_size = base.observations.shape[0]

        next_observations = self.next_observations.reshape(-1, *self.obs_shape)
        next_obs = self.pad(next_observations[batch_inds]).reshape(
            (padded_batch_size, *self.obs_shape)
        )
        expert_actions = self.expert_actions.reshape(-1, *self.actions.shape[1:])
        expert_action = self.pad(expert_actions[batch_inds]).reshape(
            (padded_batch_size, *self.actions.shape[1:])
        )
        return RecurrentRolloutBufferSamples(
            observations=base.observations,
            actions=base.actions,
            old_values=base.old_values,
            old_log_prob=base.old_log_prob,
            advantages=base.advantages,
            returns=base.returns,
            lstm_states=base.lstm_states,
            episode_starts=base.episode_starts,
            mask=base.mask,
            expert_actions=expert_action,
            next_observations=next_obs,
        )

class AdvisorPPO(RecurrentPPO):
    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "AdvisorPolicy": AdvisorPolicy,
    }

    def _setup_model(self) -> None:
        super()._setup_model()

        buffer_cls = MetaRecurrentRolloutBuffer
        lstm = self.policy.lstm_actor
        hidden_state_buffer_shape = (self.n_steps, lstm.num_layers, self.n_envs, lstm.hidden_size)

        self.rollout_buffer = buffer_cls(
            self.n_steps,
            self.observation_space,
            self.action_space,
            hidden_state_buffer_shape,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )

        self.alpha = 2.0


    def collect_rollouts(
        self,
        env,
        callback,
        rollout_buffer,
        n_rollout_steps,
    ):
        assert self._last_obs is not None, "No previous observation was provided"
        self.policy.set_training_mode(False)
        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        lstm_states = deepcopy(self._last_lstm_states)

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                episode_starts = th.tensor(self._last_episode_starts, dtype=th.float32, device=self.device)
                actions, values, log_probs, lstm_states = self.policy(obs_tensor, lstm_states, episode_starts)

            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)
            expert_action = np.array([info["expert_action"] for info in infos])

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done_ in enumerate(dones):
                if (
                    done_
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_lstm_state = (
                            lstm_states.vf[0][:, idx : idx + 1, :].contiguous(),
                            lstm_states.vf[1][:, idx : idx + 1, :].contiguous(),
                        )
                        # terminal_lstm_state = None
                        episode_starts = th.tensor([False], dtype=th.float32, device=self.device)
                        terminal_value = self.policy.predict_values(terminal_obs, terminal_lstm_state, episode_starts)[0]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
                lstm_states=self._last_lstm_states,
                expert_action=expert_action,
                next_obs=new_obs,
            )

            self._last_obs = new_obs
            self._last_episode_starts = dones
            self._last_lstm_states = lstm_states

        with th.no_grad():
            # Compute value for the last timestep
            episode_starts = th.tensor(dones, dtype=th.float32, device=self.device)
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device), lstm_states.vf, episode_starts)

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        il_losses, rl_losses, advisor_ws = [], [], []

        continue_training = True

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Convert mask from float to bool
                mask = rollout_data.mask > 1e-8

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations,
                    actions,
                    rollout_data.lstm_states,
                    rollout_data.episode_starts,
                )

                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages[mask].mean()) / (advantages[mask].std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2) # (batch_size, n_envs)
                expert_actions = rollout_data.expert_actions[:, 0]
                _, expert_log_prob, _ = self.policy.evaluate_actions(
                    rollout_data.observations,
                    expert_actions,
                    rollout_data.lstm_states,
                    rollout_data.episode_starts,
                )
                # print("expert_actions.shape, expert_log_prob.shape", expert_actions.shape, expert_log_prob.shape)
                # expert_actions
                aux_distribution = self.policy.forward_aux_expert(rollout_data.observations, rollout_data.lstm_states, rollout_data.episode_starts)
                aux_imitation_loss = -th.mean(aux_distribution.log_prob(expert_actions)[mask])

                w = th.exp(self.alpha * aux_distribution.log_prob(expert_actions).detach())  # = exp(-alpha*NLL_aux)
                # print("aux_distribution.sample().shape", aux_distribution.sample().shape)
                # print("expert_actions.shape", expert_actions.shape)
                # print("w.shape", w.shape)
                # print("policy_loss.shape", policy_loss.shape)

                il_loss = (-(w * expert_log_prob)[mask]).mean()
                rl_loss = (((1-w) * policy_loss)[mask]).mean()

                # Logging
                pg_losses.append(policy_loss.mean().item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()[mask]).item()
                clip_fractions.append(clip_fraction)
                il_losses.append(il_loss.item())
                rl_losses.append(rl_loss.item())
                advisor_ws.append(w.mean().item())

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the different between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                # Mask padded sequences
                value_loss = th.mean(((rollout_data.returns - values_pred) ** 2)[mask])

                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob[mask])
                else:
                    entropy_loss = -th.mean(entropy[mask])

                entropy_losses.append(entropy_loss.item())

                loss = rl_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + il_loss + aux_imitation_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean(((th.exp(log_ratio) - 1) - log_ratio)[mask]).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            if not continue_training:
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/il_loss", np.mean(il_losses))
        self.logger.record("train/rl_loss", np.mean(rl_losses))
        self.logger.record("train/advisor_w", np.mean(advisor_ws))
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)