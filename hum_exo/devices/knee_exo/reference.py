"""Fit device reference states to a fixed human trajectory for initialization."""

from __future__ import annotations

import hashlib
import json
from itertools import pairwise
from typing import Any

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from hum_exo.devices.knee_exo.model import KINDS, ROOT

REFERENCE_FIT_SCHEMA = b"knee-reference-fit-v1"
REFERENCE_FIT_MAX_EVALUATIONS = 30


def fit_device_reference(
    model: mujoco.MjModel, qpos: np.ndarray, qvel: np.ndarray, split_points: np.ndarray, dt: float
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Minimize cuff pose errors [m, torquescale*rad], keeping humans fixed.

    The three device coordinates cannot generally satisfy all six weld pose
    conditions along the coupled anatomical knee motion. Residuals are reported
    explicitly. At runtime, physics evolves the passive joints and motor freely.
    """
    signature = hashlib.sha256(REFERENCE_FIT_SCHEMA)
    for array in (
        qpos,
        qvel,
        split_points,
        model.body_pos,
        model.body_quat,
        model.body_parentid,
        model.jnt_type,
        model.jnt_bodyid,
        model.jnt_axis,
        model.jnt_pos,
        model.jnt_range,
        model.eq_data,
        model.eq_obj1id,
        model.eq_obj2id,
        np.array([dt]),
    ):
        array = np.ascontiguousarray(array)
        signature.update(str((array.dtype, array.shape)).encode())
        signature.update(array.tobytes())
    cache = ROOT / "generated/knee_exo/reference_fit" / (signature.hexdigest() + ".npz")
    if cache.is_file():
        with np.load(cache, allow_pickle=False) as stored:
            return stored["qpos"], stored["qvel"], json.loads(str(stored["report"]))
    qpos, qvel = qpos.copy(), qvel.copy()
    data = mujoco.MjData(model)
    report = {}
    for side in ("r", "l"):
        weld = model.equality("knee_cuff_" + side).id
        tibia, shank = int(model.eq_obj1id[weld]), int(model.eq_obj2id[weld])
        target_pos = model.eq_data[weld, 3:6]
        target_rot = Rotation.from_quat(model.eq_data[weld, 6:10], scalar_first=True).as_matrix()
        joint_ids = [model.joint(f"exo_{kind}_{side}").id for kind in KINDS]
        qids, vids = model.jnt_qposadr[joint_ids], model.jnt_dofadr[joint_ids]
        low, high = model.jnt_range[joint_ids].T
        scale = float(model.eq_data[weld, 10])
        positions, rotations = [], []
        for frame in range(len(qpos)):
            data.qpos[:] = qpos[frame]

            def residual(
                values: np.ndarray,
                qids=qids,
                tibia=tibia,
                shank=shank,
                target_pos=target_pos,
                target_rot=target_rot,
                scale=scale,
            ) -> np.ndarray:
                data.qpos[qids] = values
                mujoco.mj_kinematics(model, data)
                rt = data.xmat[tibia].reshape(3, 3)
                rs = data.xmat[shank].reshape(3, 3)
                return np.r_[
                    rt.T @ (data.xpos[shank] - data.xpos[tibia]) - target_pos,
                    scale * Rotation.from_matrix(target_rot.T @ rt.T @ rs).as_rotvec(),
                ]

            solved = least_squares(
                residual,
                np.clip(qpos[frame, qids], low, high),
                bounds=(low, high),
                max_nfev=REFERENCE_FIT_MAX_EVALUATIONS,
            )
            if not solved.success or not np.isfinite(solved.x).all():
                raise ValueError(f"Device reference fit failed at {side} frame {frame}")
            qpos[frame, qids] = solved.x
            positions.append(float(np.linalg.norm(solved.fun[:3])))
            rotations.append(float(np.linalg.norm(solved.fun[3:]) / scale))
        for start, stop in pairwise(split_points):
            segment = qpos[int(start) : int(stop), qids]
            if len(segment) > 1:
                qvel[int(start) : int(stop), vids] = np.gradient(segment, dt, axis=0)
            else:
                qvel[int(start) : int(stop), vids] = 0
        report[side] = {
            "frames": len(qpos),
            "max_position_error_mm": max(positions) * 1000,
            "p95_position_error_mm": float(np.percentile(positions, 95)) * 1000,
            "max_orientation_error_deg": float(np.rad2deg(max(rotations))),
            "rms_position_error_mm": float(np.sqrt(np.mean(np.square(positions)))) * 1000,
        }
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, qpos=qpos, qvel=qvel, report=np.array(json.dumps(report)))
    return qpos, qvel, report
