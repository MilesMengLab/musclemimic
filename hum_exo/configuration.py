"""Shared Hydra configuration composition for all device entry points."""

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from hum_exo.paths import project_name, project_root


def register_resolvers() -> None:
    """Resolve checkout paths and display names at runtime, after relocation."""
    OmegaConf.register_new_resolver("project_root", lambda: str(project_root()), replace=True)
    OmegaConf.register_new_resolver("project_name", project_name, replace=True)


def compose_config(device_id: str = "knee_exo") -> DictConfig:
    """Compose the common train config for one manifest-selected device."""
    register_resolvers()
    with initialize_config_dir(config_dir=str(project_root() / "hum_exo/configs"), version_base=None):
        return compose(config_name="train", overrides=[f"device={device_id}"])
