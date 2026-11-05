"""Preview a mounted model or play a human-device checkpoint in MuJoCo."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from omegaconf import open_dict

import hum_exo.reward
from hum_exo.checkpoint import load_checkpoint
from hum_exo.configuration import compose_config
from hum_exo.device import load_device
from hum_exo.environment import register
from hum_exo.policy import HumanExoPPO
from musclemimic.runner.engine import instantiate_env


def main() -> None:
    """Inspect prescribed installation motion or simulate a trained policy."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=str)
    source.add_argument("--preview", action="store_true")
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--motor-off", action="store_true", help="Apply zero motor command for an evaluation comparison"
    )
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="knee_exo", help="Device asset folder for reference preview")
    args = parser.parse_args()
    if args.seconds <= 0 or not np.isfinite(args.seconds):
        parser.error("--seconds must be positive and finite")
    register()
    if args.preview:
        print(
            "Reference-motion preview includes fitted passive-joint motion. Use python -m hum_exo.preview for neutral installation and knee-axis measurements."
        )
    if args.checkpoint:
        config, state, _ = load_checkpoint(args.checkpoint)
    else:
        config = compose_config(args.device)
        device = load_device(args.device)
        with open_dict(config.experiment.env_params):
            config.experiment.env_params.device_model = str(device.build_model())
    config.experiment.env_params.env_name = "MyoFullBodyHumanExo"
    device = load_device(config.experiment.env_params.device_id)
    if args.output is None:
        args.output = device.spec.generated_dir / "play_metrics.json"
    env = instantiate_env(config)
    muscle_ids = np.array(
        [
            env._model.actuator_actadr[i]
            for i in range(env._model.nu)
            if env._model.actuator_dyntype[i] == mujoco.mjtDyn.mjDYN_MUSCLE
        ],
        dtype=int,
    )
    obs = env.reset(key=jax.random.PRNGKey(args.seed))
    viewer = None
    if not args.no_viewer:
        from mujoco import viewer as mj_viewer

        viewer = mj_viewer.launch_passive(env._model, env._data)
        viewer.cam.trackbodyid = env._model.body("pelvis").id
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.distance = 3.0
    if args.checkpoint:
        agent = HumanExoPPO.init_agent_conf(env, config)
        variables = {"params": state.train_state.params, "run_stats": state.train_state.run_stats}
        infer = jax.jit(lambda x: agent.network.apply(variables, x)[0].mean())
        infer(jnp.asarray(obs)).block_until_ready()
    records = []
    resets = 0
    try:
        for step in range(int(np.ceil(args.seconds / env.dt))):
            if viewer is not None and not viewer.is_running():
                break
            start = time.monotonic()
            if args.preview:
                index = step % env.th.traj.data.qpos.shape[0]
                env._data.qpos[:] = env.th.traj.data.qpos[index]
                env._data.qvel[:] = env.th.traj.data.qvel[index]
                mujoco.mj_forward(env._model, env._data)
                reward, terminated, truncated = 0.0, False, False
            else:
                action = np.array(infer(jnp.asarray(obs)))
                if args.motor_off:
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
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "device_id": device.spec.id,
                    "preview": args.preview,
                    "motor_off": args.motor_off,
                    "seed": args.seed,
                    "episode_resets": resets,
                    "steps": records,
                },
                indent=2,
            )
            + "\n"
        )
    print(f"Saved {len(records)} steps; {resets} episode resets: {args.output}")


if __name__ == "__main__":
    main()
