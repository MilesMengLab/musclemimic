"""Public host-side contracts for interactive device installation editors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class FitParameter:
    """A physical installation degree of freedom, in displayed mm or degrees."""

    key: str
    label: str
    minimum: float
    maximum: float
    step: float
    advanced: bool = False


class FittingSession(Protocol):
    """CPU-only installation editing; learning code does not depend on the UI."""

    model: Any
    data: Any
    controls: tuple[FitParameter, ...]
    sides: tuple[str, ...]
    contact_labels: tuple[str, ...]
    last_audit: dict | None

    def parameters(self, side: str) -> dict[str, float]: ...

    def update(self, side: str, values: dict[str, float], mirror: bool, lock_axes: bool) -> None: ...

    def contact_readings(self, side: str) -> list[dict]: ...

    def pick_contact(self, side: str, index: int, geom: int, origin: Any, direction: Any) -> bool: ...

    def reset(self) -> None: ...

    def solve_contacts(self, side: str, targets_mm: list[float | None], mirror: bool, lock_axes: bool) -> dict: ...

    def audit(self) -> dict: ...

    def is_applied(self) -> bool: ...

    def save(self, output: Path, apply: bool = False) -> Path:
        """Validate hard constraints and return the actual timestamped snapshot path."""
        ...

    def load(self, path: Path) -> None: ...
