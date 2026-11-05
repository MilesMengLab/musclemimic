"""Independent MuJoCo window observing batched training state snapshots."""

from __future__ import annotations

import argparse
import fcntl
import logging
import mmap
import subprocess
import sys
import threading
import time
from pathlib import Path

import mujoco
import numpy as np

HEADER_BYTES = 16
VIEWER_START_TIMEOUT_S = 30


class StateSnapshots:
    """A locked latest-frame buffer; a slow viewer never queues training frames."""

    def __init__(self, path: Path, num_envs: int, nq: int, create: bool = False) -> None:
        self.file = path.open("w+b" if create else "r+b")
        if create:
            self.file.truncate(HEADER_BYTES + num_envs * nq * np.dtype(np.float32).itemsize)
        self.memory = mmap.mmap(self.file.fileno(), 0)
        self.header = np.ndarray((2,), dtype=np.uint64, buffer=self.memory)
        self.qpos = np.ndarray((num_envs, nq), dtype=np.float32, buffer=self.memory, offset=HEADER_BYTES)

    def publish(self, qpos: np.ndarray) -> bool:
        if qpos.shape != self.qpos.shape:
            raise ValueError(f"Viewer state shape {qpos.shape} differs from {self.qpos.shape}")
        try:
            fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        try:
            self.qpos[:] = qpos
            self.header[0] += 1
        finally:
            fcntl.flock(self.file, fcntl.LOCK_UN)
        return True

    def read(self) -> tuple[int, np.ndarray]:
        fcntl.flock(self.file, fcntl.LOCK_SH)
        try:
            return int(self.header[0]), self.qpos.copy()
        finally:
            fcntl.flock(self.file, fcntl.LOCK_UN)

    def close(self) -> None:
        del self.qpos, self.header
        self.memory.close()
        self.file.close()


class TrainingView:
    """Trace a read-only JAX observer and publish into a separate renderer process."""

    def __init__(self, model, output: Path, num_envs: int, fps: float = 10, spacing: float = 3.0) -> None:
        if num_envs < 1 or not np.isfinite(fps) or fps <= 0 or not np.isfinite(spacing) or spacing <= 0:
            raise ValueError("Rendering environment count, fps and spacing must be positive")
        output.mkdir(parents=True, exist_ok=True)
        model_path, state_path = output / "model.mjb", output / "state.bin"
        ready = output / "ready"
        ready.unlink(missing_ok=True)
        mujoco.mj_saveModel(model, str(model_path))
        self.snapshots = StateSnapshots(state_path, num_envs, model.nq, create=True)
        self.snapshots.qpos[:] = model.qpos0
        self.num_envs, self.interval = num_envs, 1 / fps
        self.last_frame = 0.0
        self.lock = threading.Lock()
        self.closed = False
        self.log = (output / "viewer.log").open("w")
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "hum_exo.training_view",
                "--model",
                str(model_path),
                "--state",
                str(state_path),
                "--num-envs",
                str(num_envs),
                "--fps",
                str(fps),
                "--spacing",
                str(spacing),
                "--ready",
                str(ready),
            ],
            stdout=self.log,
            stderr=subprocess.STDOUT,
        )
        deadline = time.monotonic() + VIEWER_START_TIMEOUT_S
        while not ready.exists():
            if self.process.poll() is not None or time.monotonic() >= deadline:
                self.close()
                detail = (output / "viewer.log").read_text()[-1500:]
                raise RuntimeError(f"Training viewer failed to open. Use --headless for no window.\n{detail}")
            time.sleep(0.05)

    def __call__(self, state) -> None:
        """Observe qpos [num_envs, nq] without modifying actions or simulation state."""
        import jax

        while hasattr(state, "env_state"):
            state = state.env_state
        jax.debug.callback(self.publish, state.data.qpos[: self.num_envs])

    def publish(self, qpos) -> None:
        if self.closed or self.process.poll() is not None or not self.lock.acquire(blocking=False):
            return
        try:
            now = time.monotonic()
            if now - self.last_frame >= self.interval and self.snapshots.publish(np.asarray(qpos)):
                self.last_frame = now
        finally:
            self.lock.release()

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.snapshots.header[1] = 1
        if self.process.poll() is None:
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=3)
        self.snapshots.close()
        self.log.close()


def grid_positions(count: int, spacing: float) -> np.ndarray:
    """Return centered display-only XY offsets [m] for a square layout."""
    columns = int(np.ceil(np.sqrt(count)))
    rows = int(np.ceil(count / columns))
    return np.array(
        [((i % columns - (columns - 1) / 2) * spacing, (i // columns - (rows - 1) / 2) * spacing) for i in range(count)]
    )


class GridScene:
    """Render training poses in tiles; display offsets never reach the simulator."""

    def __init__(self, model, count: int, spacing: float) -> None:
        self.model, self.data = model, mujoco.MjData(model)
        mujoco.mj_forward(model, self.data)
        free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
        self.root = int(model.jnt_qposadr[free[0]]) if len(free) else None
        self.offsets = grid_positions(count, spacing)
        self.options, self.camera = mujoco.MjvOption(), mujoco.MjvCamera()
        self.perturb = mujoco.MjvPerturb()
        self.options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
        self.options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
        self.camera.lookat[:] = (0, 0, 0.8)
        self.camera.distance = max(4.0, np.ceil(np.sqrt(count)) * spacing * 1.3)
        self.camera.azimuth, self.camera.elevation = 135, -40
        probe = mujoco.MjvScene(model, maxgeom=20_000)
        mujoco.mjv_updateScene(model, self.data, self.options, None, self.camera, mujoco.mjtCatBit.mjCAT_ALL, probe)
        self.scene = mujoco.MjvScene(model, maxgeom=max(20_000, (probe.ngeom * 2 + 100) * count))

    def update(self, positions: np.ndarray) -> None:
        if positions.shape != (len(self.offsets), self.model.nq):
            raise ValueError("Grid poses do not match the displayed environments/model")
        for i, (qpos, offset) in enumerate(zip(positions, self.offsets, strict=True)):
            self.data.qpos[:] = qpos
            if self.root is not None:
                self.data.qpos[self.root : self.root + 2] = offset
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)
            mujoco.mj_tendon(self.model, self.data)
            if i == 0:
                mujoco.mjv_updateScene(
                    self.model, self.data, self.options, None, self.camera, mujoco.mjtCatBit.mjCAT_ALL, self.scene
                )
            else:
                mujoco.mjv_addGeoms(
                    self.model,
                    self.data,
                    self.options,
                    self.perturb,
                    mujoco.mjtCatBit.mjCAT_DYNAMIC | mujoco.mjtCatBit.mjCAT_DECOR,
                    self.scene,
                )


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
