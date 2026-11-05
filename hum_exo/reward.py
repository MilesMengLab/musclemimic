"""Task tracking and device effort costs for human-device cooperation."""

from typing import Any

import mujoco
import numpy as np

from musclemimic.core.reward.trajectory_based import MimicReward


class HumanExoReward(MimicReward):
    """Retain imitation reward and price muscle activation and motor effort."""

    def __init__(
        self, env: Any, muscle_effort_weight: float = 0.05, device_effort_weight: float = 0.005, **kwargs: Any
    ) -> None:
        super().__init__(env, **kwargs)
        self.muscle_weight, self.device_weight = muscle_effort_weight, device_effort_weight
        model = env._model
        self.activation_ids = np.array(
            [
                model.actuator_actadr[i]
                for i in range(model.nu)
                if model.actuator_dyntype[i] == mujoco.mjtDyn.mjDYN_MUSCLE
            ],
            dtype=int,
        )
        self.motor_ids = np.array([model.actuator(name).id for name in env.device.spec.actuators])

    def __call__(
        self,
        state: Any,
        action: Any,
        next_state: Any,
        absorbing: Any,
        info: Any,
        env: Any,
        model: Any,
        data: Any,
        carry: Any,
        backend: Any,
    ) -> Any:
        reward, carry, metrics = super().__call__(
            state, action, next_state, absorbing, info, env, model, data, carry, backend
        )
        muscle = backend.mean(backend.square(data.act[self.activation_ids]))
        device = backend.mean(backend.square(data.ctrl[self.motor_ids]))
        reward = reward - self.muscle_weight * muscle - self.device_weight * device
        metrics["reward_total"] = reward
        metrics["penalty_activation_energy"] = metrics["penalty_activation_energy"] - self.muscle_weight * muscle
        metrics["penalty_total"] = metrics["penalty_total"] - self.muscle_weight * muscle - self.device_weight * device
        return reward, carry, metrics


HumanExoReward.register()


class KneeExoReward(HumanExoReward):
    """Historical reward name for existing checkpoints."""


KneeExoReward.register()
