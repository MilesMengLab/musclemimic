"""Knee-specific mechanics, reference initialization and diagnostics."""

import logging
from pathlib import Path
from typing import Any

import numpy as np

from hum_exo.device import DeviceAdapter
from hum_exo.devices.knee_exo.model import build_model
from hum_exo.devices.knee_exo.reference import fit_device_reference

REFERENCE_LIMIT_TOLERANCE_RAD = np.deg2rad(0.5)
REFERENCE_FIT_WARNING_MM = 5.0


class KneeExoAdapter(DeviceAdapter):
    """Bilateral three-joint CAD device with 26 hardware sensor channels."""

    def build_model(self, output: Path | None = None) -> Path:
        return build_model(output or self.spec.generated_dir / "human_exo.xml")

    def preview_axes(self) -> tuple[tuple[str, str], ...]:
        return tuple(("knee_angle_" + side, "exo_knee_" + side) for side in ("r", "l"))

    def create_fitting_session(self) -> Any:
        from hum_exo.devices.knee_exo.fitting import KneeFittingSession

        return KneeFittingSession(self)

    def extend_reference(self, env: Any) -> dict:
        # Materialization maps names to the expanded model before assigning device state.
        q, v = np.array(env.th.traj.data.qpos), np.array(env.th.traj.data.qvel)
        for side in ("r", "l"):
            human = env._model.joint("knee_angle_" + side)
            motor = env._model.joint("exo_knee_" + side)
            q[:, motor.qposadr[0]] = q[:, human.qposadr[0]]
            v[:, motor.dofadr[0]] = v[:, human.dofadr[0]]
            angles = q[:, motor.qposadr[0]]
            if (
                angles.min() < motor.range[0] - REFERENCE_LIMIT_TOLERANCE_RAD
                or angles.max() > motor.range[1] + REFERENCE_LIMIT_TOLERANCE_RAD
            ):
                raise ValueError(
                    f"Reference knee range [{angles.min():.4f}, {angles.max():.4f}] rad exceeds "
                    f"{side} device range {motor.range.tolist()}; select compatible motions"
                )
            # Mocap interpolation can place the human just outside extension.
            # Keep the hardware joint at its stop and report each corrected frame.
            clipped = np.clip(angles, motor.range[0], motor.range[1])
            limited = clipped != angles
            if limited.any():
                logging.getLogger("hum_exo.devices.knee_exo.reference").warning(
                    "%s: clamped %d device reference frames within %.3f rad of a joint stop; human targets retained",
                    side,
                    int(limited.sum()),
                    REFERENCE_LIMIT_TOLERANCE_RAD,
                )
            q[:, motor.qposadr[0]] = clipped
            v[limited, motor.dofadr[0]] = 0
            for kind in ("upper_hinge", "lower_hinge"):
                joint = env._model.joint(f"exo_{kind}_{side}")
                q[:, joint.qposadr[0]] = env._model.qpos0[joint.qposadr[0]]
                v[:, joint.dofadr[0]] = 0
        q, v, env.reference_fit_report = fit_device_reference(
            env._model, q, v, np.asarray(env.th.traj.data.split_points), env.dt
        )
        logging.getLogger("hum_exo.devices.knee_exo.reference").info(
            "Device reference fit: %s", env.reference_fit_report
        )
        if max(item["max_position_error_mm"] for item in env.reference_fit_report.values()) > REFERENCE_FIT_WARNING_MM:
            logging.getLogger("hum_exo.devices.knee_exo.reference").warning(
                "Reference cuff position residual exceeds 5 mm: inspect reference_fit.json and verify attachment loads before interpreting assistance results"
            )
        env.th.traj.data = env.th.traj.data.replace(qpos=q, qvel=v)

        return env.reference_fit_report

    def metrics(self, model: Any, data: Any) -> dict:
        cuff_errors = []
        for side in ("r", "l"):
            weld = model.equality("knee_cuff_" + side).id
            tibia, shank = int(model.eq_obj1id[weld]), int(model.eq_obj2id[weld])
            relative = data.xmat[tibia].reshape(3, 3).T @ (data.xpos[shank] - data.xpos[tibia])
            cuff_errors.append(float(np.linalg.norm(relative - model.eq_data[weld, 3:6])))
        return {
            "cuff_position_error_m": cuff_errors,
            "knee_rad": [float(data.joint("knee_angle_" + side).qpos[0]) for side in ("r", "l")],
            "device_knee_rad": [float(data.joint("exo_knee_" + side).qpos[0]) for side in ("r", "l")],
        }
