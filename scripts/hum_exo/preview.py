"""Inspect the calibrated neutral installation and measured joint axes."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np

from hum_exo.device import load_device
from hum_exo.paths import resolve_model
from hum_exo.visualization import alignment, draw_axes


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
    print("Neutral mounting. Green: human joint axis. Yellow: device joint axis.")
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
