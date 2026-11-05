"""Select completed device checkpoints across flat and nested run layouts."""

import json
import os
from pathlib import Path

import pytest
from omegaconf import OmegaConf

import hum_exo.checkpoint as checkpoints


def make_run(root, layout, *, device="knee_exo", environment="MjxMyoFullBodyHumanExo"):
    run = root / layout
    run.mkdir(parents=True)
    OmegaConf.save(
        OmegaConf.create({"experiment": {"env_params": {"device_id": device, "env_name": environment}}}),
        run / "session_config.yaml",
    )
    (run / "human_exo.xml").write_text("<mujoco/>")
    return run


def make_checkpoint(run, relative, stamp):
    checkpoint = run / "checkpoints" / relative
    for name in ("config", "metadata", "train_state"):
        (checkpoint / name).mkdir(parents=True)
    (checkpoint / "_CHECKPOINT_METADATA").write_text("{}")
    os.utime(checkpoint, ns=(stamp, stamp))
    return checkpoint


def test_blank_selection_uses_newest_complete_matching_device(tmp_path, monkeypatch):
    monkeypatch.setattr(checkpoints, "project_root", lambda: tmp_path)
    old = make_run(tmp_path, "outputs/2026-11-05/12-36-54")
    old_checkpoint = make_checkpoint(old, "checkpoint_228", 100)
    assert checkpoints.resolve_checkpoint("   ") == old_checkpoint
    new = make_run(tmp_path, "outputs/2026-11-06/08-00-00")
    latest = make_checkpoint(new, "run-id/checkpoint_1", 200)
    make_checkpoint(new, "run-id/checkpoint_999.orbax-checkpoint-tmp", 900)
    incomplete = new / "checkpoints/run-id/checkpoint_999"
    incomplete.mkdir()
    (incomplete / "_CHECKPOINT_METADATA").touch()
    other = make_run(tmp_path, "outputs/2026-11-06/09-00-00", device="ankle_exo")
    make_checkpoint(other, "run-id/checkpoint_1000", 1000)
    official = make_run(tmp_path, "outputs/2026-11-06/10-00-00", environment="MjxMyoFullBody")
    make_checkpoint(official, "run-id/checkpoint_2000", 2000)
    for supplied in (None, "", " ", "latest"):
        assert checkpoints.resolve_checkpoint(supplied) == latest
    assert checkpoints.resolve_checkpoint(str(old_checkpoint)) == old_checkpoint
    assert checkpoints.checkpoint_run_dir(latest) == new
    assert checkpoints.checkpoint_run_dir(old_checkpoint) == old


def test_missing_checkpoint_reports_search_locations(tmp_path, monkeypatch):
    monkeypatch.setattr(checkpoints, "project_root", lambda: tmp_path)
    with pytest.raises(FileNotFoundError, match="outputs"):
        checkpoints.resolve_checkpoint("")


def test_loader_resolves_model_from_nested_run_without_changing_checkpoint(tmp_path, monkeypatch):
    run = make_run(tmp_path, "outputs/2026-11-05/13-30-00")
    directory = make_checkpoint(run, "run-id/checkpoint_229", 200)
    config = {
        "experiment": {
            "env_params": {
                "device_id": "knee_exo",
                "env_name": "MjxMyoFullBodyHumanExo",
                "device_model": "run://human_exo.xml",
            }
        }
    }

    class Manager:
        def __init__(self, parent):
            assert Path(parent) == directory.parent

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def load_checkpoint(self, path):
            assert Path(path) == directory
            return (config, {}), {"global_timestep": 30_015_488}

    monkeypatch.setattr(checkpoints, "UnifiedCheckpointManager", Manager)
    monkeypatch.setattr(checkpoints, "create_agent_state_from_orbax", lambda state: state)
    before = (directory / "_CHECKPOINT_METADATA").read_bytes()
    loaded, _, _ = checkpoints.load_checkpoint(str(directory))
    assert loaded.experiment.env_params.device_model == str(run / "human_exo.xml")
    assert (directory / "_CHECKPOINT_METADATA").read_bytes() == before
