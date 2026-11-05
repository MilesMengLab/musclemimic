"""Device-independent joint-axis measurements and MuJoCo overlays."""

import mujoco
import numpy as np

from hum_exo.device import DeviceAdapter

AXIS_HALF_LENGTH_M = 0.10


def alignment(model: mujoco.MjModel, data: mujoco.MjData, device: DeviceAdapter) -> dict:
    """Measure paired human/device axis angle [deg], line distance and axial offset [mm]."""
    result = {}
    for human_name, device_name in device.preview_axes():
        human = model.joint(human_name).id
        motor = model.joint(device_name).id
        delta = data.xanchor[motor] - data.xanchor[human]
        axis = data.xaxis[human]
        result[device_name] = {
            "axis_angle_deg": float(np.rad2deg(np.arccos(np.clip(np.dot(axis, data.xaxis[motor]), -1, 1)))),
            "axis_line_distance_mm": float(np.linalg.norm(np.cross(delta, axis)) * 1000),
            "motor_axial_offset_mm": float(np.dot(delta, axis) * 1000),
        }
    return result


def draw_axes(scene: mujoco.MjvScene, model: mujoco.MjModel, data: mujoco.MjData, device: DeviceAdapter) -> None:
    """Draw current measured human (green) and motor (yellow) joint axes."""
    for human_name, device_name in device.preview_axes():
        for name, color, length, radius in (
            (human_name, [0.1, 0.9, 0.2, 1], AXIS_HALF_LENGTH_M, 0.0025),
            (device_name, [1, 0.8, 0, 1], AXIS_HALF_LENGTH_M * 1.4, 0.0015),
        ):
            joint = model.joint(name).id
            start = data.xanchor[joint] - length * data.xaxis[joint]
            end = data.xanchor[joint] + length * data.xaxis[joint]
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).ravel(), np.array(color)
            )
            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, radius, start, end)
            scene.ngeom += 1
