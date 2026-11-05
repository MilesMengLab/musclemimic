"""Physical model, restricted policy, and console regression coverage."""

import json
import shutil
from io import StringIO
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest
from omegaconf import OmegaConf
from rich.console import Console

from hum_exo.console import TrainingConsole, format_table
from hum_exo.controller import DeviceController
from hum_exo.device import load_device
from hum_exo.devices.knee_exo.model import MOTORS, build_model
from hum_exo.devices.knee_exo.mounting import nearest_bone
from hum_exo.devices.knee_exo.reference import fit_device_reference
from hum_exo.environment import MjxMyoFullBodyHumanExo
from hum_exo.geometry import capsule_surface_gaps, mesh_geometry, rendered_tendon_capsules
from hum_exo.policy import HumanExoActorCritic
from musclemimic.environments.humanoids.myofullbody import MyoFullBody


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    import hum_exo.devices.knee_exo.model as model_module

    directory = tmp_path_factory.mktemp("knee")
    assets = directory / "assets"
    shutil.copytree(load_device().spec.asset_root, assets)
    config = json.loads((assets / "mounting.json").read_text())
    config.pop("fitting_profile", None)
    (assets / "mounting.json").write_text(json.dumps(config))
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(model_module, "ASSETS", assets)
        path = build_model(directory / "model.xml")
    return mujoco.MjModel.from_xml_path(str(path))


def test_mass_axes_and_real_device_joints(model):
    device_bodies = [i for i in range(model.nbody) if model.body(i).name.startswith("hinge_")]
    assert len(device_bodies) == 8
    assert np.isclose(model.body_mass[device_bodies].sum(), 1.10216)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for side in ("r", "l"):
        human = model.joint("knee_angle_" + side).id
        motor = model.joint("exo_knee_" + side).id
        assert np.dot(data.xaxis[human], data.xaxis[motor]) > 1 - 1e-10
        delta = data.xanchor[motor] - data.xanchor[human]
        assert np.linalg.norm(np.cross(delta, data.xaxis[human])) < 1e-10
        assert np.isclose(abs(np.dot(delta, data.xaxis[human])), 0.0545, atol=1e-10)
        assert model.eq_type[model.equality("knee_cuff_" + side).id] == mujoco.mjtEq.mjEQ_WELD
        for kind in ("upper_hinge", "lower_hinge"):
            joint = model.joint(f"exo_{kind}_{side}").id
            joint_transmissions = model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT
            assert joint not in model.actuator_trnid[joint_transmissions, 0]


def test_motor_torque_limits_and_channel_direction(model):
    for name in MOTORS:
        data = mujoco.MjData(model)
        actuator = model.actuator(name)
        data.ctrl[actuator.id] = 2.0
        mujoco.mj_forward(model, data)
        assert np.isclose(data.actuator_force[actuator.id] * model.actuator_gear[actuator.id, 0], 20.0)


def test_neutral_mount_clears_leg_collision_capsules(model):
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for c in data.contact:
        geoms = [model.geom(int(i)).name for i in (c.geom1, c.geom2)]
        bodies = [model.body(model.geom_bodyid[int(i)]).name for i in (c.geom1, c.geom2)]
        if any(name.startswith("hinge_") for name in geoms) and any(
            name.startswith(("femur_", "tibia_", "patella_")) for name in bodies
        ):
            assert c.dist >= 0


def test_real_hinge_fit_matches_myoassist_bone_clearance(model):
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    bones = [
        (model.geom(i).name, mesh_geometry(model, data, i))
        for i in range(model.ngeom)
        if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH
        and model.geom_group[i] == 0
        and not model.geom(i).name.startswith("hinge_")
    ]
    for side in ("R", "L"):
        for link in ("base_link", "shank_link"):
            geom = model.geom(f"hinge_{side}_{link}_visual").id
            gap = nearest_bone(mesh_geometry(model, data, geom), bones)
            if link == "shank_link":
                assert abs(gap["distance_m"] - 0.001) < 0.00005
            else:
                assert gap["distance_m"] >= 0.005
            assert gap["crossing_edges"] == 0


def test_upper_cuff_respects_light_muscle_contact_limit(model):
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    segments, radii = rendered_tendon_capsules(model, data)
    hand_segments, hand_radii = rendered_tendon_capsules(model, data, body_roots=("ulna_r", "ulna_l"))
    for side in ("R", "L"):
        geom = model.geom(f"hinge_{side}_base_link_visual").id
        gaps = capsule_surface_gaps(mesh_geometry(model, data, geom), segments, radii, 0.005)
        assert len(gaps) > 0
        # The mounting permits up to 0.5 mm visual muscle overlap.
        assert gaps.min() >= -0.0005
        hand_gaps = capsule_surface_gaps(mesh_geometry(model, data, geom), hand_segments, hand_radii, 0.005)
        assert len(hand_gaps) > 0
        assert hand_gaps.min() >= 0


def test_reference_fit_preserves_human_and_trajectory_boundaries(model, tmp_path, monkeypatch):
    import hum_exo.devices.knee_exo.reference as reference

    monkeypatch.setattr(reference, "ROOT", tmp_path)
    q = np.tile(model.qpos0, (4, 1))
    v = np.zeros((4, model.nv))
    for side in ("r", "l"):
        q[:, model.joint("knee_angle_" + side).qposadr[0]] = [0.0, 0.1, 0.3, 0.4]
    original = q.copy()
    fitted, velocity, report = fit_device_reference(model, q, v, np.array([0, 2, 4]), 0.01)
    human_ids = [model.jnt_qposadr[i] for i in range(model.njnt) if not model.joint(i).name.startswith("exo_")]
    np.testing.assert_array_equal(fitted[:, human_ids], original[:, human_ids])
    np.testing.assert_array_equal(q, original)
    for side in ("r", "l"):
        assert report[side]["frames"] == 4
        assert np.isfinite(report[side]["max_position_error_mm"])
        for kind in ("upper_hinge", "knee", "lower_hinge"):
            joint = model.joint(f"exo_{kind}_{side}")
            position = fitted[:, joint.qposadr[0]]
            assert np.all(position >= joint.range[0]) and np.all(position <= joint.range[1])
            for start in (0, 2):
                np.testing.assert_allclose(
                    velocity[start : start + 2, joint.dofadr[0]],
                    np.gradient(position[start : start + 2], 0.01),
                )

    def unexpected_solver(*args, **kwargs):
        pytest.fail("Identical trajectory and model should reuse the fit cache")

    monkeypatch.setattr(reference, "least_squares", unexpected_solver)
    cached, _, cached_report = fit_device_reference(model, q, v, np.array([0, 2, 4]), 0.01)
    np.testing.assert_array_equal(cached, fitted)
    assert cached_report == report


def test_policy_isolation_gradients_jit_and_frozen_normalization(tmp_path):
    network = HumanExoActorCritic(
        4, (0, 1, 2), tuple(range(3, 29)), (0, 1), (2, 3), actor_hidden=(8,), exo_hidden=(8,), critic_hidden=(8,)
    )
    obs = jax.random.normal(jax.random.PRNGKey(1), (8, 29))
    variables = network.init(jax.random.PRNGKey(2), obs)
    forward = jax.jit(lambda x: network.apply(variables, x)[0].mean())
    mean = forward(obs)
    changed = forward(obs.at[:, :3].add(20))
    np.testing.assert_allclose(mean[:, 2:], changed[:, 2:], atol=1e-7)
    sensor_changed = forward(obs.at[:, 3:].add(10))
    np.testing.assert_allclose(mean[:, :2], sensor_changed[:, :2], atol=1e-7)

    def loss(params):
        dist, value = network.apply({**variables, "params": params}, obs)
        return -dist.log_prob(jnp.ones((8, 4))).mean() + (value**2).mean()

    grad = jax.jit(jax.grad(loss))(variables["params"])
    for key in ("actor_human", "actor_exo", "critic"):
        assert sum(float(jnp.linalg.norm(x)) for x in jax.tree_util.tree_leaves(grad[key])) > 0
    assert mean.shape == (8, 4)
    assert forward(obs[:1]).shape == (1, 4)
    stats = variables["run_stats"]["ObservationNorm_0"]
    payload = {"mean": np.array(stats["mean"])[3:], "var": np.array(stats["var"])[3:], "layers": np.array(2)}
    for i in range(2):
        layer = variables["params"]["actor_exo"][f"Dense_{i}"]
        payload[f"kernel_{i}"] = np.array(layer["kernel"])
        payload[f"bias_{i}"] = np.array(layer["bias"])
    weights = tmp_path / "device.npz"
    np.savez(weights, **payload)
    device = DeviceController(weights)
    np.testing.assert_allclose(device(np.asarray(obs[:, 3:])), np.clip(mean[:, 2:], -1, 1), atol=1e-6)


def test_gpu_constructor_inherits_official_solver_and_preserves_overrides(model, tmp_path, monkeypatch):
    from types import SimpleNamespace

    import hum_exo.environment as environment

    path = tmp_path / "model.xml"
    mujoco.mj_saveLastXML(str(path), model)
    monkeypatch.setattr(
        environment,
        "load_device",
        lambda _: SimpleNamespace(spec=SimpleNamespace(human_model="MyoFullBody", joints=(), sensors=())),
    )
    received = []

    def capture_init(self, **kwargs):
        received.append(kwargs)

    monkeypatch.setattr(MyoFullBody, "__init__", capture_init)
    MjxMyoFullBodyHumanExo(device_model=str(path), mjx_backend="warp")
    options = received[-1]["model_option_conf"]
    assert options["iterations"] == 4
    assert options["ls_iterations"] == 8
    assert options["disableflags"] & mujoco.mjtDisableBit.mjDSBL_EULERDAMP
    custom = {"iterations": 12, "ls_iterations": 20}
    MjxMyoFullBodyHumanExo(device_model=str(path), model_option_conf=OmegaConf.create(custom))
    assert received[-1]["model_option_conf"] == custom
    assert isinstance(received[-1]["model_option_conf"], dict)


def test_console_prints_real_metrics_and_preserves_jsonl(tmp_path):
    cfg = OmegaConf.create(
        {"experiment": {"num_envs": 2, "num_steps": 4, "total_timesteps": 16, "clip_eps": 0.2, "clip_eps_vf": 0.2}}
    )
    clock = iter([0.0, 2.0])
    stream = StringIO()
    terminal = Console(file=stream, width=140, force_terminal=True, color_system=None)
    logger = TrainingConsole(cfg, tmp_path, clock=lambda: next(clock), console=terminal)
    try:
        logger(
            {
                "max_timestep": 8,
                "jax_raw_timestep": 8,
                "ppo/value_loss": 1.5,
                "ppo/entropy": 2.0,
                "ppo/kl_divergence": 0.01,
                "learning_rate": 0.0001,
            }
        )
        assert logger.progress.tasks[0].completed == 8
        assert logger.progress.live.is_started
    finally:
        logger.close()
        logger.close()
    output = stream.getvalue()
    assert "total_timesteps" in output
    assert "value_loss" in output
    assert "50%" in output
    assert "8/16" in output
    assert "━" in output
    assert "[INFO]" not in output
    assert "[progress." not in output
    assert "ETA" not in output
    assert (tmp_path / "training_metrics.jsonl").is_file()
    assert "Early stopping" not in output
    assert "[INFO]" not in (tmp_path / "training_console.log").read_text()
    assert "| time/" in format_table({"time": {"fps": 4}})
