"""Local checkpoint loading with deterministic resource cleanup."""

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf, open_dict

from hum_exo.configuration import register_resolvers
from hum_exo.device import load_device
from hum_exo.paths import resolve_path
from musclemimic.algorithms.common.checkpoint_manager import UnifiedCheckpointManager
from musclemimic.algorithms.ppo.checkpoint import create_agent_state_from_orbax


def load_checkpoint(path: str) -> tuple[Any, Any, Any]:
    """Load a specific local checkpoint directory, closing its Orbax manager."""
    register_resolvers()
    directory = resolve_path(path)
    if not directory.is_dir() or not directory.name.startswith("checkpoint_"):
        raise ValueError("Provide a complete local path ending in checkpoint_NUMBER")
    with UnifiedCheckpointManager(str(directory.parent)) as manager:
        (config, state), metadata = manager.load_checkpoint(str(directory))
    config = OmegaConf.create(config)
    if config.experiment.env_params.env_name not in (
        "MjxMyoFullBodyHumanExo",
        "MyoFullBodyHumanExo",
        "MjxMyoFullBodyKneeExo",
        "MyoFullBodyKneeExo",
    ):
        raise ValueError("Checkpoint was not trained with the hum_exo environment")
    run_dir = directory.parent.parent
    # Resolve relative to the checkpoint's actual location, even after the
    # experiment was copied to another directory or checkout name.
    with open_dict(config.experiment.env_params):
        value = config.experiment.env_params.device_model
        basename = Path(str(value).removeprefix("run://")).name
        sibling = run_dir / basename
        config.experiment.env_params.device_model = str(sibling if sibling.is_file() else resolve_path(value, run_dir))
        if "device_id" not in config.experiment.env_params:
            config.experiment.env_params.device_id = "knee_exo"
    manifest = run_dir / "device.json"
    if manifest.is_file():
        import json

        recorded = json.loads(manifest.read_text())
        current = load_device(config.experiment.env_params.device_id).spec
        saved_inputs = tuple(name for sensor in recorded["sensors"] for name in sensor["channels"])
        saved_units = tuple(unit for sensor in recorded["sensors"] for unit in sensor["units"])
        if (saved_inputs, saved_units, tuple(recorded["actuators"]), tuple(recorded["joints"])) != (
            current.input_names,
            current.input_units,
            current.actuators,
            current.joints,
        ):
            raise ValueError(
                "Device contract differs from the checkpoint: restore its asset package or migrate explicitly"
            )
    return config, create_agent_state_from_orbax(state), metadata
