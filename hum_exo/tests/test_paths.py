"""Relocation checks independent of a particular checkout directory name."""

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

import hum_exo.paths as paths


def test_project_and_run_paths_do_not_depend_on_checkout_name(tmp_path, monkeypatch):
    root = tmp_path / "My Renamed Project"
    root.mkdir()
    monkeypatch.setattr(paths, "project_root", lambda: root)
    assert paths.project_name() == "My Renamed Project"
    target = root / "assets/knee_exo/meshes/test.STL"
    target.parent.mkdir(parents=True)
    target.touch()
    assert paths.resolve_path("project://assets/knee_exo/meshes/test.STL") == target
    assert paths.portable_path(target) == "project://assets/knee_exo/meshes/test.STL"
    assert paths.resolve_path("/unavailable/old_repo/assets/knee_exo/meshes/test.STL") == target
    assert paths.resolve_path("run://human_exo.xml", root / "outputs/one") == root / "outputs/one/human_exo.xml"
    checkpoint = root / "outputs/2026-11-05/12-00-00/checkpoints/run-id/checkpoint_229"
    checkpoint.mkdir(parents=True)
    assert (
        paths.resolve_path("/unavailable/old_repo/outputs/2026-11-05/12-00-00/checkpoints/run-id/checkpoint_229")
        == checkpoint
    )


def test_relative_model_assets_survive_copy_and_root_rename(tmp_path, monkeypatch):
    root = tmp_path / "Original Project"
    mesh = root / "assets/demo/tetra.obj"
    mesh.parent.mkdir(parents=True)
    mesh.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n")
    xml = ET.fromstring(
        f'<mujoco><asset><mesh name="tetra" file="{mesh}"/></asset><worldbody><body><freejoint/><geom type="mesh" mesh="tetra"/></body></worldbody></mujoco>'
    )
    monkeypatch.setattr(paths, "project_root", lambda: root)
    original = paths.write_portable_xml(xml, root / "outputs/first/model.xml")
    copied = paths.copy_model(original, root / "outputs/second/model.xml")
    assert not Path(ET.parse(copied).find("asset/mesh").get("file")).is_absolute()
    before = mujoco.MjModel.from_xml_path(str(copied))
    renamed = tmp_path / "Another Arbitrary Name"
    shutil.move(root, renamed)
    monkeypatch.setattr(paths, "project_root", lambda: renamed)
    after = mujoco.MjModel.from_xml_path(str(renamed / "outputs/second/model.xml"))
    np.testing.assert_array_equal(before.mesh_vert, after.mesh_vert)


def test_historical_xml_relocates_assets_without_modifying_snapshot(tmp_path, monkeypatch):
    root = tmp_path / "new"
    texture = root / "assets/demo/picture.png"
    texture.parent.mkdir(parents=True)
    texture.touch()
    saved = root / "outputs/old/model.xml"
    saved.parent.mkdir(parents=True)
    contents = '<mujoco><asset><texture name="t" type="2d" file="/old/missing_project/assets/demo/picture.png"/></asset><worldbody/></mujoco>'
    saved.write_text(contents)
    monkeypatch.setattr(paths, "project_root", lambda: root)
    relocated = paths.resolve_model(saved)
    assert relocated != saved
    assert saved.read_text() == contents
    relative = ET.parse(relocated).find("asset/texture").get("file")
    assert (relocated.parent / relative).resolve() == texture
