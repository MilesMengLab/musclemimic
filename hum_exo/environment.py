"""Human and device environments using MuscleMimic's public registries."""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from omegaconf import DictConfig, ListConfig, OmegaConf

from hum_exo.device import load_device
from hum_exo.paths import resolve_model
from loco_mujoco.core import ObservationType
from loco_mujoco.core.observations.base import SimpleObs
from loco_mujoco.task_factories import ImitationFactory
from loco_mujoco.task_factories.dataset_confs import (
    AMASSDatasetConf,
    expand_amass_dataset_group_spec,
    get_amass_dataset_groups,
)
from musclemimic.environments.humanoids.myofullbody import MjxMyoFullBody, MyoFullBody


class DeviceSensor(SimpleObs):
    """Read named MuJoCo sensor channels in local-frame SI units."""

    def _init_from_mj(self, env: Any, model: Any, data: Any, current_obs_size: int) -> None:
        sensor = model.sensor(self.xml_name)
        dim = int(sensor.dim[0])
        self.dim = dim
        self.data_type_ind = np.arange(int(sensor.adr[0]), int(sensor.adr[0]) + dim)
        self.obs_ind = np.arange(current_obs_size, current_obs_size + dim)
        self.min, self.max = np.full(dim, -np.inf), np.full(dim, np.inf)
        self._initialized_from_mj = True

    @classmethod
    def data_type(cls) -> str:
        return "sensordata"


DeviceSensor.register()


class MyoFullBodyHumanExo(MyoFullBody):
    """A MyoFullBody task composed with a manifest-selected device adapter."""

    # Upstream site-cache selection uses "MyoFullBody" in the class name.
    # Keep that host identity while selecting device mechanics by manifest.

    def __init__(self, device_id: str = "knee_exo", device_model: str | None = None, **kwargs: Any) -> None:
        self.device = load_device(device_id)
        if self.device.spec.human_model != "MyoFullBody":
            raise ValueError(
                "This environment currently supports MyoFullBody; other humans need a host environment adapter"
            )
        path = resolve_model(device_model) if device_model else self.device.build_model()
        kwargs["spec"] = str(path)
        reward = dict(kwargs.get("reward_params") or {})
        spec = mujoco.MjSpec.from_file(str(path))
        reward["joints_for_mimic"] = [j.name for j in spec.joints if j.name not in self.device.spec.joints]
        kwargs["reward_params"] = reward
        # The upstream constructor appends timestep/integrator defaults in
        # place. A Hydra struct DictConfig forbids inserting those keys.
        if isinstance(kwargs.get("model_option_conf"), DictConfig):
            kwargs["model_option_conf"] = OmegaConf.to_container(kwargs["model_option_conf"], resolve=True)
        super().__init__(**kwargs)
        for sensor in self.device.spec.sensors:
            if int(self._model.sensor(sensor["name"]).dim[0]) != len(sensor["channels"]):
                raise ValueError(f"Sensor dimension differs from manifest: {sensor['name']}")

    def _get_observation_specification(self, spec: mujoco.MjSpec) -> list:
        joints = [
            j.name
            for j in spec.joints
            if j.name != self.root_free_joint_xml_name and j.name not in self.device.spec.joints
        ]
        obs = [
            ObservationType.FreeJointPosNoXY("q_free_joint", self.root_free_joint_xml_name, group="human"),
            ObservationType.JointPosArray("q_human", joints, group="human"),
            ObservationType.FreeJointVel("dq_free_joint", self.root_free_joint_xml_name, group="human"),
            ObservationType.JointVelArray("dq_human", joints, group="human"),
        ]
        for actuator in spec.actuators:
            if actuator.dyntype != mujoco.mjtDyn.mjDYN_MUSCLE:
                continue
            for flag, cls in (
                ("length", "ActuatorLength"),
                ("velocity", "ActuatorVelocity"),
                ("force", "ActuatorForce"),
                ("excitation", "ActuatorExcitation"),
                ("activation", "ActuatorActivation"),
            ):
                if getattr(self, f"_enable_muscle_{flag}_observations"):
                    obs.append(
                        getattr(ObservationType, cls)(f"muscle_{flag}_{actuator.name}", actuator.name, group="human")
                    )
        if self._enable_touch_sensor_observations:
            for name in ("r_foot", "r_toes", "l_foot", "l_toes"):
                obs.append(ObservationType.TouchSensor("touch_" + name, name, group="human"))
        for sensor in self.device.spec.sensors:
            obs.append(DeviceSensor(sensor["name"], sensor["name"], group="exo"))
        return obs

    def load_trajectory(self, *args: Any, **kwargs: Any) -> None:
        super().load_trajectory(*args, **kwargs)
        self.th.to_numpy()
        self.reference_fit_report = self.device.extend_reference(self)


class MjxMyoFullBodyHumanExo(MyoFullBodyHumanExo, MjxMyoFullBody):
    """GPU environment sharing the CPU model, sensors and rewards."""

    mjx_enabled = True

    # Cooperative super() runs the upstream MJX constructor, including its
    # solver defaults (4/8 iterations and EULERDAMP disabled) and user overrides.


class MyoFullBodyKneeExo(MyoFullBodyHumanExo):
    """Historical knee checkpoint environment name."""


class MjxMyoFullBodyKneeExo(MjxMyoFullBodyHumanExo):
    """Historical knee GPU checkpoint environment name."""


class HumanExoImitationFactory(ImitationFactory):
    """Reuse human reference caches while composing the additional device state."""

    @staticmethod
    def get_amass_trajectory_set(env: Any, amass_dataset_conf: Any, visualize_goal: bool = False) -> Any:
        from loco_mujoco.smpl.retargeting import load_retargeted_amass_trajectory_set

        conf = (
            AMASSDatasetConf(**amass_dataset_conf)
            if isinstance(amass_dataset_conf, (dict, DictConfig))
            else amass_dataset_conf
        )
        paths = []
        groups = get_amass_dataset_groups()
        for group in expand_amass_dataset_group_spec(conf.dataset_group):
            if group not in groups:
                raise ValueError(f"Unknown AMASS group: {group}")
            paths.extend(groups[group])
        if conf.rel_dataset_path is not None:
            paths.extend(
                conf.rel_dataset_path
                if isinstance(conf.rel_dataset_path, (list, ListConfig))
                else [conf.rel_dataset_path]
            )
        paths = list(dict.fromkeys(paths))
        if conf.max_motions is not None and len(paths) > conf.max_motions:
            paths = random.Random(0).sample(paths, conf.max_motions)
        return load_retargeted_amass_trajectory_set(
            env.device.spec.human_model,
            paths,
            retargeting_method=conf.retargeting_method,
            gmr_config=conf.gmr_config,
            clear_cache=conf.clear_cache,
        )


class KneeExoImitationFactory(HumanExoImitationFactory):
    """Historical factory name for existing knee checkpoints."""


def register() -> None:
    """Register common environments and historical checkpoint aliases."""
    for cls in (MyoFullBodyHumanExo, MjxMyoFullBodyHumanExo, MyoFullBodyKneeExo, MjxMyoFullBodyKneeExo):
        cls.register()
    for cls in (HumanExoImitationFactory, KneeExoImitationFactory):
        if cls.get_name() not in cls.registered:
            cls.register()
