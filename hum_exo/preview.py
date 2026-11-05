"""Inspect the calibrated neutral installation and measured knee axes."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np

from hum_exo.device import DeviceAdapter, load_device
from hum_exo.paths import resolve_model

AXIS_HALF_LENGTH_M = 0.10


def alignment(model: mujoco.MjModel, data: mujoco.MjData, device: DeviceAdapter) -> dict:
    """Measure knee axis angle [deg], line distance and axial offset [mm]."""
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


def main() -> None:
    """Build the current calibration, then open or render a static preview."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", action="store_true", help="Save static images without opening a window")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="knee_exo")
    parser.add_argument(
        "--model", type=Path, help="Inspect an existing candidate XML without changing the installation"
    )
    args = parser.parse_args()
    device = load_device(args.device)
    if args.output is None:
        args.output = args.model.parent / "preview" if args.model else device.spec.generated_dir / "mounting_preview"
    path = resolve_model(args.model) if args.model else device.build_model()
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    report = alignment(model, data, device)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "alignment.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Neutral mounting. Green: human knee axis. Yellow: motor axis. Orange: right device. Blue: left device.")
    print(json.dumps(report, indent=2))
    camera = mujoco.MjvCamera()
    camera.lookat[:] = (
        np.mean([data.xanchor[model.joint(pair[0]).id] for pair in device.preview_axes()], axis=0)
        if device.preview_axes()
        else data.subtree_com[0]
    )
    camera.distance, camera.azimuth, camera.elevation = 1.4, 90, -8
    if args.render:
        import imageio.v2 as imageio

        model.vis.global_.offwidth, model.vis.global_.offheight = 900, 900
        with mujoco.Renderer(model, height=900, width=900) as renderer:
            for name, azimuth in (("front", 0), ("side", 90), ("oblique", 45)):
                camera.azimuth = azimuth
                renderer.update_scene(data, camera=camera)
                draw_axes(renderer.scene, model, data, device)
                imageio.imwrite(args.output / (name + ".png"), renderer.render())
        print("Images:", args.output)
        return
    from mujoco import viewer as mj_viewer

    with mj_viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = camera.lookat
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = (
            camera.distance,
            camera.azimuth,
            camera.elevation,
        )
        while viewer.is_running():
            with viewer.lock():
                # Reset UI edits to the calibrated neutral posture.
                data.qpos[:] = model.qpos0
                mujoco.mj_forward(model, data)
                viewer.user_scn.ngeom = 0
                draw_axes(viewer.user_scn, model, data, device)
            viewer.sync()
            time.sleep(1 / 30)


if __name__ == "__main__":
    main()
