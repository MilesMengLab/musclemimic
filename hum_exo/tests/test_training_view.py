"""Headless CLI selection and state-only batched rendering boundaries."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest

from hum_exo.train import training_arguments
from hum_exo.training_view import GridScene, StateSnapshots, TrainingView


@pytest.mark.parametrize(
    "arguments,enabled",
    [
        ([], True),
        (["--headless"], False),
        (["experiment.render.enabled=false"], False),
        (["experiment.render.enabled=true", "--headless"], False),
    ],
)
def test_cli_headless_selects_window_without_changing_environment_count(arguments, enabled):
    result = training_arguments(["experiment.env_params.num_envs=128", *arguments])
    assert "--headless" not in result
    assert "experiment.env_params.num_envs=128" in result
    assert f"experiment.render.enabled={str(enabled).lower()}" in result
    assert sum(arg.startswith("experiment.render.enabled=") for arg in result) == 1


def test_headless_override_stays_with_hydra_positional_overrides():
    arguments = training_arguments(["device=knee_exo", "--headless", "--cfg", "job", "--resolve"])
    assert arguments.index("experiment.render.enabled=false") < arguments.index("--cfg")


def test_latest_state_snapshot_is_copied_between_independent_readers(tmp_path):
    path = tmp_path / "states.bin"
    writer = StateSnapshots(path, 128, 7, create=True)
    reader = StateSnapshots(path, 128, 7)
    try:
        poses = np.arange(128 * 7, dtype=np.float32).reshape(128, 7)
        assert writer.publish(poses)
        seq, copied = reader.read()
        assert seq == 1
        np.testing.assert_array_equal(copied, poses)
        copied[:] = -1
        np.testing.assert_array_equal(reader.read()[1], poses)
        with pytest.raises(ValueError, match="shape"):
            writer.publish(poses[:1])
    finally:
        reader.close()
        writer.close()


def test_grid_scene_contains_128_agents_without_changing_original_coordinates():
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><geom type="plane" size="1 1 .1"/>'
        '<body pos="0 0 1"><freejoint/><geom type="sphere" size=".1" mass="1"/></body>'
        "</worldbody></mujoco>"
    )
    poses = np.tile(model.qpos0, (128, 1))
    poses[:, 0] = np.arange(128)
    original = poses.copy()
    grid = GridScene(model, 128, 3)
    grid.update(poses)
    np.testing.assert_array_equal(poses, original)
    spheres = [g for g in grid.scene.geoms[: grid.scene.ngeom] if g.type == mujoco.mjtGeom.mjGEOM_SPHERE]
    assert len(spheres) == 128
    assert len({tuple(np.round(g.pos[:2], 5)) for g in spheres}) == 128


def test_compiled_observer_receives_batched_states_and_does_not_change_results():
    view = TrainingView.__new__(TrainingView)
    view.num_envs = 128
    snapshots = []
    view.publish = lambda positions: snapshots.append(np.array(positions))

    @jax.jit
    def step(positions):
        result = positions + 0.1
        state = SimpleNamespace(env_state=SimpleNamespace(data=SimpleNamespace(qpos=result)))
        view(state)
        return result

    positions = jnp.zeros((128, 7))
    result = step(positions).block_until_ready()
    jax.effects_barrier()
    assert len(snapshots) == 1
    np.testing.assert_allclose(result, np.asarray(positions) + 0.1)
    np.testing.assert_allclose(snapshots[0], result)
