"""Internal renderer process for the live training grid."""

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np

from hum_exo.training_view import GridScene, StateSnapshots


def run_window(args) -> None:
    """Own GLFW on this process's main thread and display latest available poses."""
    import glfw

    model = mujoco.MjModel.from_binary_path(str(args.model))
    snapshots = StateSnapshots(args.state, args.num_envs, model.nq)
    if not glfw.init():
        raise RuntimeError("GLFW could not initialize; check DISPLAY or use --headless")
    window = glfw.create_window(1600, 1000, f"Training: {args.num_envs} environments", None, None)
    if window is None:
        glfw.terminate()
        raise RuntimeError("Could not create the MuJoCo window")
    glfw.make_context_current(window)
    glfw.swap_interval(1)
    grid = GridScene(model, args.num_envs, args.spacing)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
    last = [0.0, 0.0]

    def move(_window, x, y):
        dx, dy = x - last[0], y - last[1]
        last[:] = [x, y]
        height = max(1, glfw.get_window_size(window)[1])
        left = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        right = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
        if left or right:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_H if left else mujoco.mjtMouse.mjMOUSE_MOVE_H
            mujoco.mjv_moveCamera(model, action, dx / height, dy / height, grid.scene, grid.camera)

    glfw.set_cursor_pos_callback(window, move)
    glfw.set_scroll_callback(
        window, lambda _w, _x, y: setattr(grid.camera, "distance", grid.camera.distance * np.exp(-0.08 * y))
    )
    args.ready.write_text("ready\n")
    try:
        while not glfw.window_should_close(window) and not snapshots.header[1]:
            started = time.monotonic()
            seq, qpos = snapshots.read()
            grid.update(qpos)
            width, height = glfw.get_framebuffer_size(window)
            viewport = mujoco.MjrRect(0, 0, width, height)
            mujoco.mjr_render(viewport, grid.scene, context)
            label = (
                f"Training environments: {args.num_envs}"
                if seq
                else "Initializing / compiling; waiting for training..."
            )
            mujoco.mjr_overlay(
                mujoco.mjtFont.mjFONT_NORMAL,
                mujoco.mjtGridPos.mjGRID_TOPLEFT,
                viewport,
                label,
                "Left drag: rotate | Right drag: move | Wheel: zoom | Close: keep training",
                context,
            )
            glfw.swap_buffers(window)
            glfw.poll_events()
            time.sleep(max(0.0, 1 / args.fps - (time.monotonic() - started)))
    finally:
        context.free()
        glfw.destroy_window(window)
        glfw.terminate()
        snapshots.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--ready", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--fps", type=float, default=10)
    parser.add_argument("--spacing", type=float, default=3)
    run_window(parser.parse_args())


if __name__ == "__main__":
    main()
