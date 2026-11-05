"""Device-independent contracts, dispatch and actor dimensions."""

import json
import sys
from types import ModuleType, SimpleNamespace

import jax
import numpy as np
import pytest
from omegaconf import OmegaConf

import hum_exo.device as device_module
from hum_exo.controller import DeviceController
from hum_exo.device import DeviceAdapter, load_device
from hum_exo.policy import HumanExoPPO


class TestAdapter(DeviceAdapter):
    __test__ = False

    def build_model(self, output=None):
        return output

    def extend_reference(self, env):
        return {}


def test_new_device_loads_from_manifest_without_framework_changes(tmp_path, monkeypatch):
    directory = tmp_path / "assets/ankle_demo"
    directory.mkdir(parents=True)
    manifest = {
        "schema_version": 1,
        "adapter": "test_exo_adapter:TestAdapter",
        "actuators": ["ankle_motor"],
        "joints": ["ankle_motor_joint"],
        "sensors": [{"name": "encoder", "channels": ["ankle.angle"], "units": ["rad"]}],
    }
    (directory / "device.json").write_text(json.dumps(manifest))
    module = ModuleType("test_exo_adapter")
    module.TestAdapter = TestAdapter
    monkeypatch.setitem(sys.modules, "test_exo_adapter", module)
    monkeypatch.setattr(device_module, "project_root", lambda: tmp_path)
    adapter = load_device("ankle_demo")
    assert isinstance(adapter, TestAdapter)
    assert adapter.spec.input_names == ("ankle.angle",)
    assert adapter.spec.actuators == ("ankle_motor",)
    assert adapter.spec.asset_root == directory
    manifest["sensors"][0]["units"] = []
    (directory / "device.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="units"):
        load_device("ankle_demo")
    with pytest.raises(ValueError, match="Device ID"):
        load_device("../ankle_demo")


@pytest.mark.parametrize("inputs,outputs", [(1, 1), (7, 3)])
def test_policy_constructor_uses_device_dimensions(inputs, outputs):
    names = ["muscle_a", *[f"device_{i}" for i in range(outputs)], "muscle_b"]
    cfg = OmegaConf.create(
        {
            "experiment": {
                "actor_hidden_layers": [8],
                "exo_hidden_layers": [4],
                "critic_hidden_layers": [8],
                "init_std": 0.3,
                "learnable_std": True,
            }
        }
    )
    env = SimpleNamespace(
        device=SimpleNamespace(spec=SimpleNamespace(input_names=tuple(range(inputs)), actuators=tuple(names[1:-1]))),
        obs_container=SimpleNamespace(get_obs_ind_by_group=lambda _: np.arange(2, 2 + inputs)),
        mdp_info=SimpleNamespace(observation_space=SimpleNamespace(shape=(inputs + 2,))),
        _model=SimpleNamespace(actuator=lambda i: SimpleNamespace(name=names[i])),
        _action_indices=np.arange(len(names)),
    )
    network = HumanExoPPO._create_network(env, cfg)
    x = np.ones((2, inputs + 2), dtype=np.float32)
    variables = network.init(jax.random.PRNGKey(2), x)
    distribution, _ = jax.jit(lambda obs: network.apply(variables, obs))(x)
    assert distribution.mean().shape == (2, outputs + 2)
    assert len(network.exo_indices) == inputs
    assert len(network.exo_actions) == outputs


def test_numpy_controller_accepts_manifest_sized_inputs(tmp_path):
    path = tmp_path / "actor.npz"
    np.savez(path, mean=np.zeros(3), var=np.ones(3), layers=1, kernel_0=np.ones((3, 1)), bias_0=np.zeros(1))
    controller = DeviceController(path)
    np.testing.assert_allclose(controller(np.zeros(3)), [0])
    with pytest.raises(ValueError, match="3"):
        controller(np.zeros(26))
