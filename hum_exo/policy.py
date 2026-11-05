"""Separate human and sensor-only exoskeleton actors with a shared critic."""

from __future__ import annotations

from typing import Any

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from musclemimic.algorithms.common.networks import FullyConnectedNet
from musclemimic.algorithms.ppo.ppo import PPOJax


class ObservationNorm(nn.Module):
    """Per-channel normalization with frozen statistics during deployment."""

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        mean = self.variable("run_stats", "mean", lambda: jnp.zeros(x.shape[-1]))
        var = self.variable("run_stats", "var", lambda: jnp.ones(x.shape[-1]))
        count = self.variable("run_stats", "count", lambda: jnp.array(1e-6))
        if self.is_mutable_collection("run_stats"):
            batch = jnp.reshape(x, (-1, x.shape[-1]))
            n = batch.shape[0]
            total = count.value + n
            delta = batch.mean(axis=0) - mean.value
            new_var = (var.value * count.value + batch.var(axis=0) * n + delta**2 * count.value * n / total) / total
            mean.value = mean.value + delta * n / total
            var.value = new_var
            count.value = total
        return (x - mean.value) / jnp.sqrt(var.value + 1e-8)


class HumanExoActorCritic(nn.Module):
    """Factorized Gaussian action distribution, independently restricted actors."""

    action_dim: int
    human_indices: tuple[int, ...]
    exo_indices: tuple[int, ...]
    human_actions: tuple[int, ...]
    exo_actions: tuple[int, ...]
    actor_hidden: tuple[int, ...] = (256, 256)
    exo_hidden: tuple[int, ...] = (64, 64)
    critic_hidden: tuple[int, ...] = (256, 256)
    init_std: float = 0.3
    learnable_std: bool = True

    @nn.compact
    def __call__(self, x: jax.Array) -> tuple[Any, jax.Array]:
        x = ObservationNorm()(x)
        human = FullyConnectedNet(
            self.actor_hidden,
            len(self.human_actions),
            name="actor_human",
            use_running_mean_stand=False,
            squeeze_output=False,
        )(x[..., jnp.array(self.human_indices)])
        exo = FullyConnectedNet(
            self.exo_hidden, len(self.exo_actions), name="actor_exo", use_running_mean_stand=False, squeeze_output=False
        )(x[..., jnp.array(self.exo_indices)])
        mean = jnp.zeros((*x.shape[:-1], self.action_dim), dtype=x.dtype)
        mean = mean.at[..., jnp.array(self.human_actions)].set(human)
        mean = mean.at[..., jnp.array(self.exo_actions)].set(exo)
        log_std = self.param("log_std", nn.initializers.constant(jnp.log(self.init_std)), (self.action_dim,))
        if not self.learnable_std:
            log_std = jax.lax.stop_gradient(log_std)
        pi = distrax.MultivariateNormalDiag(mean, jnp.exp(log_std))
        value = FullyConnectedNet(
            self.critic_hidden, 1, name="critic", use_running_mean_stand=False, squeeze_output=False
        )(x)
        return pi, jnp.squeeze(value, axis=-1)


class HumanExoPPO(PPOJax):
    """Reuse MuscleMimic PPO updates with a human/device network constructor."""

    @classmethod
    def _create_network(cls, env: Any, config: Any) -> HumanExoActorCritic:
        exp = config.experiment
        if exp.get("len_obs_history", 1) != 1:
            raise ValueError(
                "The device actor requires len_obs_history=1; add history explicitly to its sensor contract"
            )
        exo_indices = tuple(int(i) for i in env.obs_container.get_obs_ind_by_group("exo"))
        if len(exo_indices) != len(env.device.spec.input_names):
            raise ValueError("Observation dimensions do not match the device manifest")
        size = env.mdp_info.observation_space.shape[0]
        human_indices = tuple(i for i in range(size) if i not in exo_indices)
        names = [env._model.actuator(int(i)).name for i in env._action_indices]
        exo_actions = tuple(names.index(name) for name in env.device.spec.actuators)
        human_actions = tuple(i for i in range(len(names)) if i not in exo_actions)
        return HumanExoActorCritic(
            action_dim=len(names),
            human_indices=human_indices,
            exo_indices=exo_indices,
            human_actions=human_actions,
            exo_actions=exo_actions,
            actor_hidden=tuple(exp.actor_hidden_layers),
            exo_hidden=tuple(exp.exo_hidden_layers),
            critic_hidden=tuple(exp.critic_hidden_layers),
            init_std=exp.init_std,
            learnable_std=exp.learnable_std,
        )
