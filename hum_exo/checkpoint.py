"""Local checkpoint loading with deterministic resource cleanup."""

import re
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf, open_dict

from hum_exo.configuration import register_resolvers
from hum_exo.device import load_device
from hum_exo.paths import project_root, resolve_path
from musclemimic.algorithms.common.checkpoint_manager import UnifiedCheckpointManager
from musclemimic.algorithms.ppo.checkpoint import create_agent_state_from_orbax

HUMAN_EXO_ENVIRONMENTS = (
    "MjxMyoFullBodyHumanExo",
    "MyoFullBodyHumanExo",
)


def checkpoint_run_dir(directory: Path) -> Path:
    """Locate a run snapshot in both flat and nested checkpoint layouts."""
    for ancestor in directory.parents:
        if (ancestor / "session_config.yaml").is_file():
            return ancestor
    return directory.parent.parent


def resolve_checkpoint(path: str | Path | None = None, device_id: str = "knee_exo") -> Path:
    """Resolve an explicit path or the most recently saved complete device model."""
    supplied = str(path).strip() if path is not None else ""
    if supplied and supplied != "latest":
        return resolve_path(supplied)
    candidates = []
    for root in (project_root() / "outputs",):
        for snapshot in root.rglob("session_config.yaml"):
            config = OmegaConf.load(snapshot)
            environment = OmegaConf.select(config, "experiment.env_params.env_name")
            device = OmegaConf.select(config, "experiment.env_params.device_id")
            if environment not in HUMAN_EXO_ENVIRONMENTS or device != device_id:
                continue
            for directory in (snapshot.parent / "checkpoints").rglob("checkpoint_*"):
                # Orbax commits by renaming its temporary directory. Require
                # the final name, commit metadata and all three saved items.
                if not re.fullmatch(r"checkpoint_\d+", directory.name):
                    continue
                if not (directory / "_CHECKPOINT_METADATA").is_file():
                    continue
                if not all((directory / item).is_dir() for item in ("train_state", "config", "metadata")):
                    continue
                candidates.append(directory)
    if not candidates:
        raise FileNotFoundError(f"No complete {device_id} checkpoint found under outputs/ in {project_root()}")
    return max(candidates, key=lambda p: (p.stat().st_mtime_ns, int(p.name.removeprefix("checkpoint_"))))


def load_checkpoint(path: str) -> tuple[Any, Any, Any]:
    """Load a specific local checkpoint directory, closing its Orbax manager."""
    register_resolvers()
    directory = resolve_path(path)
    if not directory.is_dir() or not directory.name.startswith("checkpoint_"):
        raise ValueError("Provide a complete local path ending in checkpoint_NUMBER")
    with UnifiedCheckpointManager(str(directory.parent)) as manager:
        (config, state), metadata = manager.load_checkpoint(str(directory))
    config = OmegaConf.create(config)
    if config.experiment.env_params.env_name not in HUMAN_EXO_ENVIRONMENTS:
        raise ValueError("Checkpoint was not trained with the hum_exo environment")
    run_dir = checkpoint_run_dir(directory)
    # Resolve relative to the checkpoint's actual location, even after the
    # experiment was copied to another directory or checkout name.
    with open_dict(config.experiment.env_params):
        value = config.experiment.env_params.device_model
        basename = Path(str(value).removeprefix("run://")).name
        sibling = run_dir / basename
        config.experiment.env_params.device_model = str(sibling if sibling.is_file() else resolve_path(value, run_dir))
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
