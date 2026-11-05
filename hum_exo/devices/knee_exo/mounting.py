"""Fit real cuff hinges to the rendered bones with a fixed motor housing."""

from __future__ import annotations

import hashlib
import json
import logging
from itertools import pairwise
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import brentq

from hum_exo.geometry import (
    capsule_surface_gaps,
    fitting_distance,
    mesh_geometry,
    rendered_tendon_capsules,
    surface_crossings,
    surface_distance,
)
from hum_exo.paths import project_root

ANGLE_SCAN_STEP_DEG = 2.0
CLEARANCE_SOLVER_TOLERANCE_DEG = 0.0001
CLEARANCE_TOLERANCE_M = 0.00005
EXACT_REFINEMENT_SPANS_DEG = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 20.0, 40.0)
VISUAL_SCENE_CAPACITY = 10000


def body_transform(data: mujoco.MjData, body: int) -> np.ndarray:
    """Return a body homogeneous transform [m]."""
    pose = np.eye(4)
    pose[:3, :3] = data.xmat[body].reshape(3, 3)
    pose[:3, 3] = data.xpos[body]
    return pose


def nearest_bone(surface: tuple, bones: list[tuple[str, tuple]], exact: bool = True) -> dict:
    """Measure triangle clearance [m], with crossing surfaces at zero."""
    vertices = surface[0]
    candidates = []
    for name, bone in bones:
        gap = np.maximum(0, np.maximum(vertices.min(0) - bone[0].max(0), bone[0].min(0) - vertices.max(0)))
        candidates.append((float(np.linalg.norm(gap)), name, bone))
    best = {"distance_m": np.inf, "bone": None, "crossing_edges": 0}
    for lower, name, bone in sorted(candidates, key=lambda item: item[0]):
        if lower > best["distance_m"]:
            break
        crossings = surface_crossings(bone, surface) if exact else 0
        distance = 0.0 if crossings else (surface_distance(bone, surface) if exact else fitting_distance(bone, surface))
        if distance < best["distance_m"]:
            best = {"distance_m": distance, "bone": name, "crossing_edges": int(crossings)}
    return best


def fit_hinges(
    human: mujoco.MjModel,
    human_data: mujoco.MjData,
    device: mujoco.MjModel,
    device_data: mujoco.MjData,
    femur_pose: np.ndarray,
    mount: np.ndarray,
    target_m: float,
    upper_hinge_deg: float | None = None,
    upper_visual_gap_m: float = 0.0,
) -> tuple[np.ndarray, dict]:
    """Fit upper/lower angles to bone clearance, preserving the motor pose.

    `mount` maps device base to human femur. Joint limits come from the URDF;
    every candidate uses full CAD kinematics without moving individual meshes.
    """
    bones = [
        (human.geom(i).name, mesh_geometry(human, human_data, i))
        for i in range(human.ngeom)
        if human.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH and human.geom_group[i] == 0
    ]
    housing = device.body("motor_housing_link").id
    housing_target = mount @ body_transform(device_data, housing)
    report = {}

    def current_mount() -> np.ndarray:
        return housing_target @ np.linalg.inv(body_transform(device_data, housing))

    segments, radii = rendered_tendon_capsules(human, human_data)
    hand_segments, hand_radii = rendered_tendon_capsules(human, human_data, body_roots=("ulna_r", "ulna_l"))
    signature = hashlib.sha256(b"bone-hinge-fit-v3-hand-clearance")
    for values in (
        human.mesh_vert,
        human.mesh_face,
        human_data.geom_xpos,
        human_data.geom_xmat,
        device.mesh_vert,
        device.mesh_face,
        device.body_pos,
        device.body_quat,
        device.jnt_axis,
        device.jnt_range,
        housing_target,
        femur_pose,
        device_data.qpos,
        np.array([target_m]),
        np.array([upper_hinge_deg if upper_hinge_deg is not None else np.nan, upper_visual_gap_m]),
        segments,
        radii,
        hand_segments,
        hand_radii,
    ):
        signature.update(np.ascontiguousarray(values).tobytes())
    cache = project_root() / "generated/knee_exo/mount_fits" / (signature.hexdigest() + ".json")
    if cache.is_file():
        report = json.loads(cache.read_text())
        for kind, result in report.items():
            device_data.qpos[device.joint(kind + "_joint").qposadr[0]] = np.deg2rad(result["angle_deg"])
        mujoco.mj_kinematics(device, device_data)
        return current_mount(), report

    for kind, link in (("upper_hinge", "base_link"), ("lower_hinge", "shank_link")):
        joint = device.joint(kind + "_joint")
        qid = joint.qposadr[0]
        initial = float(np.rad2deg(device_data.qpos[qid]))
        geom = next(
            i
            for i in range(device.ngeom)
            if device.geom_bodyid[i] == device.body(link).id and device.geom_group[i] == 1
        )
        evaluations = {}

        def measure(angle: float, exact=False, qid=qid, geom=geom, evaluations=evaluations, link=link) -> float:
            device_data.qpos[qid] = np.deg2rad(angle)
            mujoco.mj_kinematics(device, device_data)
            transform = femur_pose @ current_mount()
            vertices, faces = mesh_geometry(device, device_data, geom)
            surface = vertices @ transform[:3, :3].T + transform[:3, 3], faces
            result = nearest_bone(surface, bones, exact=exact)
            evaluations[float(angle)] = result
            logging.getLogger("hum_exo.devices.knee_exo.mounting").info(
                "%s angle %.5f gap %.5f mm", link, angle, result["distance_m"] * 1000
            )
            return result["distance_m"] - target_m

        limits = np.rad2deg(joint.range)
        if kind == "upper_hinge" and upper_hinge_deg is not None:
            if not limits[0] <= upper_hinge_deg <= limits[1]:
                raise ValueError("Configured upper hinge angle exceeds the CAD joint limit")
            error = measure(upper_hinge_deg, exact=True)
            if error < -CLEARANCE_TOLERANCE_M:
                raise ValueError("Configured upper hinge does not clear the human bone surfaces")
            transform = femur_pose @ current_mount()
            vertices, faces = mesh_geometry(device, device_data, geom)
            surface = vertices @ transform[:3, :3].T + transform[:3, 3], faces
            gaps = capsule_surface_gaps(surface, segments, radii, max(target_m, upper_visual_gap_m))
            minimum = float(gaps.min()) if len(gaps) else float("inf")
            if minimum < upper_visual_gap_m:
                raise ValueError(f"Upper cuff intersects the rendered muscle paths: {minimum * 1000:.3f} mm gap")
            hand_gaps = capsule_surface_gaps(surface, hand_segments, hand_radii, target_m)
            hand_minimum = float(hand_gaps.min()) if len(hand_gaps) else float("inf")
            if hand_minimum < 0:
                raise ValueError(f"Upper cuff intersects hand/forearm muscles: {hand_minimum * 1000:.3f} mm gap")
            report[kind] = {
                "angle_deg": upper_hinge_deg,
                **evaluations[upper_hinge_deg],
                "visual_muscle_gap_m": minimum,
                "muscle_surface_intersection_count": int((gaps < 0).sum()),
                "hand_visual_gap_m": hand_minimum,
                "hand_surface_intersection_count": int((hand_gaps < 0).sum()),
            }
            continue
        # Expand around the prior fit to select the closest reachable branch.
        samples = np.unique(np.r_[initial, np.arange(limits[0], limits[1], ANGLE_SCAN_STEP_DEG), limits[1]])
        ordered = sorted(samples, key=lambda x: abs(x - initial))
        measured = {}
        selected = None
        logging.getLogger("hum_exo.devices.knee_exo.mounting").info(
            "Fitting %s cuff to %.1f mm bone clearance", kind, target_m * 1000
        )
        for angle in ordered:
            measured[float(angle)] = measure(float(angle))
            by_angle = sorted(measured)
            brackets = [(a, b) for a, b in pairwise(by_angle) if measured[a] * measured[b] <= 0]
            if brackets:
                a, b = min(brackets, key=lambda pair: abs(np.mean(pair) - initial))
                selected = brentq(measure, a, b, xtol=CLEARANCE_SOLVER_TOLERANCE_DEG)
                break
        if selected is None:
            raise ValueError(f"Cannot achieve {target_m * 1000:.1f} mm {kind} gap within URDF limits: {measured}")
        exact_error = measure(selected, exact=True)
        if abs(exact_error) > CLEARANCE_TOLERANCE_M:
            # Approximate vertex/face distance may bracket another branch than
            # the full triangle surface. Expand from the candidate until exact
            # distances straddle the requested gap, retaining CAD joint limits.
            measured_exact = {selected: exact_error}
            bracket = None
            for span in EXACT_REFINEMENT_SPANS_DEG:
                for angle in (max(limits[0], selected - span), min(limits[1], selected + span)):
                    if angle not in measured_exact:
                        measured_exact[angle] = measure(angle, exact=True)
                brackets = [
                    (a, b) for a, b in pairwise(sorted(measured_exact)) if measured_exact[a] * measured_exact[b] <= 0
                ]
                if brackets:
                    bracket = min(brackets, key=lambda pair: abs(np.mean(pair) - selected))
                    break
            if bracket is None:
                raise ValueError(f"No exact {kind} clearance solution within CAD joint limits")
            selected = brentq(
                lambda angle: measure(angle, exact=True),
                *bracket,
                xtol=CLEARANCE_SOLVER_TOLERANCE_DEG,
            )
            measure(selected, exact=True)
        report[kind] = {"angle_deg": selected, **evaluations[selected]}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(report, indent=2) + "\n")
    return current_mount(), report
