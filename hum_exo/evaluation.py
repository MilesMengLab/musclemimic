"""Reusable policy and reference-motion evaluation for manifest-selected devices."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

import hum_exo.reward
from hum_exo.device import load_device
from hum_exo.environment import register
from hum_exo.policy import HumanExoPPO
from musclemimic.runner.engine import instantiate_env


def evaluate(
    config: Any,
    state: Any = None,
    *,
    seconds: float = 30.0,
    seed: int = 0,
    preview: bool = False,
    motor_off: bool = False,
    show_viewer: bool = True,
    output: Path | None = None,
) -> dict:
    """Simulate a policy or reference for `seconds` [s], recording SI device metrics."""
    if seconds <= 0 or not np.isfinite(seconds):
        raise ValueError("seconds must be positive and finite")
    if state is None and not preview:
        raise ValueError("Policy evaluation requires a trained state")
    register()
    config.experiment.env_params.env_name = "MyoFullBodyHumanExo"
    device = load_device(config.experiment.env_params.device_id)
    if output is None:
        output = device.spec.generated_dir / "play_metrics.json"
    output = Path(output)
    env = instantiate_env(config)
    muscle_ids = np.array(
        [
            env._model.actuator_actadr[i]
            for i in range(env._model.nu)
            if env._model.actuator_dyntype[i] == mujoco.mjtDyn.mjDYN_MUSCLE
        ],
        dtype=int,
    )
    obs = env.reset(key=jax.random.PRNGKey(seed))
    viewer = None
    if show_viewer:
        from mujoco import viewer as mj_viewer

        viewer = mj_viewer.launch_passive(env._model, env._data)
        viewer.cam.trackbodyid = env._model.body("pelvis").id
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.distance = 3.0
    if state is not None:
        agent = HumanExoPPO.init_agent_conf(env, config)
        variables = {"params": state.train_state.params, "run_stats": state.train_state.run_stats}
        infer = jax.jit(lambda x: agent.network.apply(variables, x)[0].mean())
        infer(jnp.asarray(obs)).block_until_ready()
    records = []
    resets = 0
    try:
        for step in range(int(np.ceil(seconds / env.dt))):
            if viewer is not None and not viewer.is_running():
                break
            start = time.monotonic()
            if preview:
                index = step % env.th.traj.data.qpos.shape[0]
                env._data.qpos[:] = env.th.traj.data.qpos[index]
                env._data.qvel[:] = env.th.traj.data.qvel[index]
                mujoco.mj_forward(env._model, env._data)
                reward, terminated, truncated = 0.0, False, False
            else:
                action = np.array(infer(jnp.asarray(obs)))
                if motor_off:
                    action[list(agent.network.exo_actions)] = 0
                obs, reward, terminated, truncated, _ = env.step(action)
            data, model = env._data, env._model
            motors = [model.actuator(name).id for name in device.spec.actuators]
            torques = data.actuator_force[motors] * model.actuator_gear[motors, 0]
            motor_velocity = np.array([data.qvel[model.jnt_dofadr[model.actuator_trnid[i, 0]]] for i in motors])
            rows = {
                "time_s": step * env.dt,
                "reward": float(reward),
                "episode_end": bool(terminated or truncated),
                "torque_nm": torques.tolist(),
                "device_velocity_rad_s": motor_velocity.tolist(),
                "device_power_w": (torques * motor_velocity).tolist(),
                "mean_muscle_activation_squared": float(np.mean(data.act[muscle_ids] ** 2)),
            }
            rows.update(device.metrics(model, data))
            records.append(rows)
            if terminated or truncated:
                resets += 1
                obs = env.reset()
            if viewer is not None:
                viewer.sync()
                time.sleep(max(0, env.dt - (time.monotonic() - start)))
    finally:
        if viewer is not None:
            viewer.close()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "device_id": device.spec.id,
                    "preview": preview,
                    "motor_off": motor_off,
                    "seed": seed,
                    "episode_resets": resets,
                    "steps": records,
                },
                indent=2,
            )
            + "\n"
        )
    return {"output": output, "steps": len(records), "episode_resets": resets}
