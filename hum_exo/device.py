"""Versioned device contracts and manifest-driven adapter discovery."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

from hum_exo.paths import project_root


@dataclass(frozen=True)
class DeviceSpec:
    """A device's ordered control and sensor contract, independent of joint type."""

    id: str
    asset_root: Path
    adapter: str
    actuators: tuple[str, ...]
    joints: tuple[str, ...]
    sensors: tuple[dict, ...]
    human_model: str = "MyoFullBody"

    @property
    def input_names(self) -> tuple[str, ...]:
        return tuple(name for sensor in self.sensors for name in sensor["channels"])

    @property
    def input_units(self) -> tuple[str, ...]:
        return tuple(unit for sensor in self.sensors for unit in sensor["units"])

    @property
    def generated_dir(self) -> Path:
        return project_root() / "generated" / self.id


class DeviceAdapter(ABC):
    """Implement device mechanics and reference extension without changing PPO."""

    def __init__(self, spec: DeviceSpec) -> None:
        self.spec = spec

    @abstractmethod
    def build_model(self, output: Path | None = None) -> Path:
        """Compose this device with the human and return portable MuJoCo XML."""

    @abstractmethod
    def extend_reference(self, env: Any) -> dict:
        """Initialize this device's coordinates and return fitting diagnostics."""

    def preview_axes(self) -> tuple[tuple[str, str], ...]:
        """Return optional (human joint, device joint) pairs to visualize."""
        return ()

    def metrics(self, model: Any, data: Any) -> dict:
        """Return additional device-specific evaluation quantities in SI units."""
        return {}

    def create_fitting_session(self) -> Any:
        """Return this device's installation editor backend, when supported."""
        raise NotImplementedError(f"Device {self.spec.id} does not provide interactive fitting")


def load_device(device_id: str = "knee_exo") -> DeviceAdapter:
    """Load assets/<id>/device.json and its declared adapter, without a registry edit."""
    if not re.fullmatch(r"[a-zA-Z0-9_\-]+", device_id):
        raise ValueError("Device ID must be a simple directory name")
    directory = project_root() / "assets" / device_id
    path = directory / "device.json"
    document = json.loads(path.read_text())
    if document.get("schema_version") != 1:
        raise ValueError(f"Unsupported device manifest version: {path}")
    spec = DeviceSpec(
        id=device_id,
        asset_root=directory,
        adapter=document["adapter"],
        actuators=tuple(document["actuators"]),
        joints=tuple(document["joints"]),
        sensors=tuple(document["sensors"]),
        human_model=document.get("human_model", "MyoFullBody"),
    )
    if not spec.actuators or len(set(spec.actuators)) != len(spec.actuators):
        raise ValueError("Device actuator names must be nonempty and unique")
    if not spec.input_names or len(set(spec.input_names)) != len(spec.input_names):
        raise ValueError("Device input channel names must be nonempty and unique")
    for sensor in spec.sensors:
        if len(sensor["channels"]) != len(sensor["units"]):
            raise ValueError(f"Sensor channels and units differ: {sensor['name']}")
    if len(set(spec.joints)) != len(spec.joints):
        raise ValueError("Device joint names must be unique")
    if len({sensor["name"] for sensor in spec.sensors}) != len(spec.sensors):
        raise ValueError("Device sensor names must be unique")
    module, name = spec.adapter.split(":", 1)
    cls = getattr(import_module(module), name)
    if not issubclass(cls, DeviceAdapter):
        raise TypeError(f"Adapter must implement DeviceAdapter: {spec.adapter}")
    return cls(spec)
