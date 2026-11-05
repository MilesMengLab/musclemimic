"""Installation kinematics, contact solving and saved training-model parity."""

import json
import re
import shutil
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import pytest

from hum_exo.device import load_device
from hum_exo.devices.knee_exo.fitting import KneeFittingSession, point_capsule_gaps
from hum_exo.devices.knee_exo.model import build_model
from hum_exo.visualization import alignment


@pytest.fixture(scope="module")
def fitting_source(tmp_path_factory):
    import hum_exo.devices.knee_exo.model as model_module

    directory = tmp_path_factory.mktemp("fitting_model")
    assets = directory / "assets"
    shutil.copytree(load_device().spec.asset_root, assets)
    config = json.loads((assets / "mounting.json").read_text())
    config.pop("fitting_profile", None)
    (assets / "mounting.json").write_text(json.dumps(config))
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(model_module, "ASSETS", assets)
        return build_model(directory / "human_exo.xml")


@pytest.fixture
def session(fitting_source, tmp_path, monkeypatch):
    import hum_exo.device as device_module
    import hum_exo.devices.knee_exo.model as model_module

    adapter = load_device()
    assets = tmp_path / "assets/knee_exo"
    shutil.copytree(adapter.spec.asset_root, assets)
    adapter.spec = replace(adapter.spec, asset_root=assets)
    monkeypatch.setattr(device_module, "project_root", lambda: tmp_path)
    monkeypatch.setattr(model_module, "ASSETS", assets)
    return KneeFittingSession(adapter, model_path=fitting_source)


def test_point_gap_uses_capsule_radius_and_surface():
    segments = np.array([[[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]]])
    gaps, points = point_capsule_gaps(np.array([[0.006, 0.0, 0.0], [0.004, 0.0, 0.0]]), segments, np.array([0.005]))
    np.testing.assert_allclose(gaps, [0.001, -0.001], atol=1e-12)
    np.testing.assert_allclose(points, [[0.005, 0.0, 0.0], [0.005, 0.0, 0.0]])


def test_upper_hinge_keeps_motor_and_lower_chain_intact(session):
    s = session
    before = s.data.xpos.copy()
    angle = s.parameters("R")["upper_deg"] + 0.2
    s.update("R", {"upper_deg": angle}, mirror=False)
    for link in ("motor_housing_link", "motor_output_link", "shank_link"):
        body = s.model.body("hinge_R_" + link).id
        np.testing.assert_allclose(s.data.xpos[body], before[body], atol=1e-12)
    assert not np.allclose(
        s.data.xpos[s.model.body("hinge_R_base_link").id], before[s.model.body("hinge_R_base_link").id]
    )
    for body in range(s.model.nbody):
        if s.model.body(body).name.startswith("hinge_L_"):
            np.testing.assert_allclose(s.data.xpos[body], before[body], atol=1e-12)
    s.update("R", {"twist_deg": 2.0}, mirror=True)
    assert s.parameters("L") == s.parameters("R")
    for result in alignment(s.model, s.data, s.adapter).values():
        assert result["axis_line_distance_mm"] < 1e-7
        assert result["axis_angle_deg"] < 0.001


def test_draft_roundtrip_retains_pose_and_sensor_contract(session, tmp_path):
    s = session
    s.update("R", {"upper_deg": -7.3, "offset_mm": 54.6}, mirror=True)
    expected = s.data.xpos.copy()
    draft = s.save(tmp_path / "draft.json")
    assert re.fullmatch(r"draft_\d{8}-\d{6}-\d{6}\.json", draft.name)
    assert json.loads(draft.read_text())["schema_version"] == 1
    model = mujoco.MjModel.from_xml_path(str(draft.with_name(draft.stem + "_model.xml")))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.xpos, expected, atol=1e-9)
    assert model.nsensordata == s.model.nsensordata
    assert model.nu == s.model.nu
    s.reset()
    s.load(draft)
    np.testing.assert_allclose(s.data.xpos, expected, atol=1e-9)
    with pytest.raises(ValueError, match=r"\.json"):
        s.save(tmp_path / "draft.txt")


def test_contact_solver_reduces_target_error_and_preserves_joint_limits(session):
    s = session
    initial = s.contact_readings("R")[1]["gap_mm"]
    target = initial - 0.3
    result = s.solve_contacts("R", [None, target, None, None], mirror=True, lock_axes=True)
    assert abs(result["residual_mm"][0]) < 0.1
    assert abs(s.contact_readings("R")[1]["gap_mm"] - target) < abs(initial - target)
    assert s.parameters("R") == s.parameters("L")
    with pytest.raises(ValueError, match="at least"):
        s.solve_contacts("R", [None] * 4)
    with pytest.raises(ValueError, match="range"):
        s.update("R", {"upper_deg": 99})


def test_apply_and_rebuild_use_saved_profile_without_changing_cad(session, tmp_path):
    s = session
    assets = s.adapter.spec.asset_root
    before_urdf = (assets / "urdf/knee_exo_hinge_r.urdf").read_bytes()
    original = (assets / "mounting.json").read_text()
    result = s.audit()
    assert result["applicable"]
    s.save(tmp_path / "approved.json", apply=True)
    configuration = json.loads((assets / "mounting.json").read_text())
    assert re.fullmatch(r"profiles/fitting_profile_\d{8}-\d{6}-\d{6}\.json", configuration["fitting_profile"])
    assert (assets / configuration["fitting_profile"]).is_file()
    path = build_model(tmp_path / "rebuilt.xml")
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.xpos, s.data.xpos, atol=1e-9)
    assert (assets / "urdf/knee_exo_hinge_r.urdf").read_bytes() == before_urdf
    backups = list((s.adapter.spec.generated_dir / "fitting_history").glob("*/mounting.json"))
    assert len(backups) == 1 and backups[0].read_text() == original


def test_apply_rejects_axis_error_without_mutating_assets(session, tmp_path):
    s = session
    path = s.adapter.spec.asset_root / "mounting.json"
    before = path.read_text()
    s.update("R", {"forward_mm": 5.0}, mirror=False, lock_axes=False)
    assert not s.audit()["applicable"]
    for apply in (False, True):
        with pytest.raises(ValueError, match="膝轴"):
            s.save(tmp_path / "invalid.json", apply=apply)
    assert path.read_text() == before
    assert not list(tmp_path.glob("invalid*"))


def test_reapplying_active_geometry_keeps_warnings_and_rejects_axis_changes(session, tmp_path):
    s = session
    s.update("R", {"offset_mm": 48.5, "upper_deg": -5.4, "lower_deg": -2.1, "twist_deg": 2.3})
    report = s.audit()
    assert report["applicable"]
    assert report["warnings"] and not report["failures"]
    assets = s.adapter.spec.asset_root
    path = assets / "mounting.json"
    config = json.loads(path.read_text())
    config["fitting_profile"] = "fitting_profile.json"
    path.write_text(json.dumps(config))
    active = assets / "fitting_profile.json"
    active.write_text(json.dumps(s.profile()))
    before = active.read_bytes()
    saved = s.save(tmp_path / "already_enabled.json", apply=True)
    assert s.is_applied()
    assert active.read_bytes() == before
    assert len(list((s.adapter.spec.generated_dir / "fitting_history").iterdir())) == 1
    assert json.loads(saved.read_text())["audit"]["warnings"]
    s.load(saved)
    assert s.is_applied()
    # Every generated or applied version still requires aligned axes.
    s.update("R", {"forward_mm": 1.0}, mirror=False, lock_axes=False)
    assert not s.is_applied()
    with pytest.raises(ValueError, match="膝轴"):
        s.save(tmp_path / "new_geometry.json", apply=True)
    assert active.read_bytes() == before


def test_timestamped_saves_preserve_all_versions_even_with_same_clock(session, tmp_path, monkeypatch):
    from datetime import UTC, datetime

    import hum_exo.devices.knee_exo.fitting as fitting_module

    class FrozenClock:
        @staticmethod
        def now(tz=None):
            return datetime(2026, 11, 4, 12, 0, 0, 123456, tzinfo=UTC)

    monkeypatch.setattr(fitting_module, "datetime", FrozenClock)
    report = session.audit()
    monkeypatch.setattr(session, "audit", lambda: report)
    selected = tmp_path / "用户选择的目录 with spaces" / "mount.json"
    first = session.save(selected, apply=True)
    original = first.read_bytes()
    first_config = json.loads((session.adapter.spec.asset_root / "mounting.json").read_text())
    second = session.save(first, apply=True)
    second_config = json.loads((session.adapter.spec.asset_root / "mounting.json").read_text())
    assert first != second and first.parent == second.parent == selected.parent
    assert first.read_bytes() == original
    assert not selected.exists()
    for saved in (first, second):
        assert saved.is_file()
        assert saved.with_name(saved.stem + "_model.xml").is_file()
        assert saved.with_name(saved.stem + "_model.json").is_file()
    assert first_config["fitting_profile"] != second_config["fitting_profile"]
    assert (session.adapter.spec.asset_root / first_config["fitting_profile"]).is_file()
    assert (session.adapter.spec.asset_root / second_config["fitting_profile"]).is_file()


def test_bone_hand_and_muscle_interference_are_warnings(session, tmp_path, monkeypatch):
    import hum_exo.devices.knee_exo.fitting as fitting_module

    monkeypatch.setattr(
        fitting_module, "nearest_bone", lambda *_: {"distance_m": 0, "bone": "test_bone", "crossing_edges": 1}
    )
    monkeypatch.setattr(fitting_module, "capsule_surface_gaps", lambda *_: np.array([-0.002]))
    report = session.audit()
    assert report["applicable"] and not report["failures"]
    assert any("骨骼" in warning for warning in report["warnings"])
    assert any("手部" in warning for warning in report["warnings"])
    assert any("肌肉" in warning for warning in report["warnings"])
    path = session.save(tmp_path / "warnings.json", apply=True)
    assert json.loads(path.read_text())["audit"]["warnings"]
