"""Tools depend on reusable framework services and device-declared visuals."""

import ast
from importlib import import_module
from pathlib import Path

import mujoco

from hum_exo.device import DeviceSpec
from hum_exo.paths import project_root

from .test_devices import TestAdapter


def test_tool_entry_points_and_framework_dependency_direction():
    root = project_root()
    for command in ("play", "preview", "fit", "export"):
        assert callable(import_module(f"scripts.hum_exo.{command}").main)
    assert not (root / "knee_exo").exists()
    for source in (root / "hum_exo").glob("*.py"):
        if source.name == "train.py":
            continue
        tree = ast.parse(source.read_text())
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(item.name for item in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        assert not any(name == "argparse" or name.startswith(("scripts.", "knee_exo.")) for name in imports), source


def test_device_visual_bodies_do_not_require_knee_naming():
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body name="human"><geom type="sphere" size=".1"/>'
        '<body name="hip_shell"><geom type="sphere" size=".02"/>'
        '<body name="hip_motor"><joint name="motor_joint"/><geom type="sphere" size=".02"/></body>'
        "</body></body></worldbody></mujoco>"
    )
    spec = DeviceSpec(
        "hip_demo",
        Path("assets/hip_demo"),
        "test:Adapter",
        ("motor",),
        ("motor_joint",),
        (),
        visual_body_roots=("hip_shell",),
    )
    adapter = TestAdapter(spec)
    assert adapter.visual_body_ids(model) == {model.body("hip_shell").id, model.body("hip_motor").id}
    assert model.body("human").id not in adapter.visual_body_ids(model)
