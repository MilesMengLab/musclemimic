"""Discover this checkout and keep experiment/model paths relocatable."""

from __future__ import annotations

import hashlib
import os
import xml.etree.ElementTree as ET
from pathlib import Path


def project_root() -> Path:
    """Find the project marker without depending on its directory name."""
    for directory in Path(__file__).resolve().parents:
        if (directory / "pyproject.toml").is_file():
            return directory
    raise FileNotFoundError("hum_exo requires a checkout containing pyproject.toml")


def project_name() -> str:
    """Return the current checkout directory name for display and logging."""
    return project_root().name


def portable_path(path: str | Path) -> str:
    """Serialize paths inside this checkout using project://, retaining external paths."""
    path = Path(path).resolve()
    try:
        return "project://" + path.relative_to(project_root()).as_posix()
    except ValueError:
        return str(path)


def resolve_path(value: str | Path, run_dir: Path | None = None) -> Path:
    """Resolve project://, run://, relative and historical absolute paths."""
    value = str(value)
    if value.startswith("project://"):
        return project_root() / value.removeprefix("project://")
    if value.startswith("run://"):
        if run_dir is None:
            raise ValueError("run:// requires the checkpoint's experiment directory")
        return run_dir / value.removeprefix("run://")
    path = Path(value).expanduser()
    if path.exists():
        return path.resolve()
    if not path.is_absolute():
        return project_root() / path
    # Old experiments embedded the former checkout name. Only accept a known
    # project suffix that actually exists, rather than guessing by basename.
    for index, part in enumerate(path.parts):
        if part in {"assets", "hum_exo", "outputs", "generated", ".venv", "fullbody", "bimanual"}:
            candidate = project_root().joinpath(*path.parts[index:])
            if candidate.exists():
                return candidate
    return path


def write_portable_xml(root: ET.Element, output: Path, source_dir: Path | None = None) -> Path:
    """Write XML with asset paths relative to its own directory [MuJoCo]."""
    output = Path(output).resolve()
    source_dir = source_dir or output.parent
    compiler = root.find("compiler")
    folders = {"mesh": "meshdir", "texture": "texturedir"}
    for asset in root.findall("asset/*"):
        for key, value in list(asset.attrib.items()):
            if not key.startswith("file") or not value:
                continue
            path = Path(value)
            if not path.is_absolute():
                folder = compiler.get(folders.get(asset.tag, "assetdir"), "") if compiler is not None else ""
                path = source_dir / folder / path
            path = resolve_path(path)
            if not path.is_file():
                raise FileNotFoundError(f"Missing {asset.tag} asset: {path}")
            asset.set(key, os.path.relpath(path, output.parent))
    if compiler is not None:
        for key in ("meshdir", "texturedir", "assetdir"):
            compiler.attrib.pop(key, None)
        compiler.set("strippath", "false")
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root)
    ET.ElementTree(root).write(output, encoding="utf-8")
    return output


def copy_model(source: Path, output: Path) -> Path:
    """Copy a model while rebasing relative asset references to its new location."""
    source = resolve_path(source)
    return write_portable_xml(ET.parse(source).getroot(), output, source.parent)


def resolve_model(source: str | Path) -> Path:
    """Read a historical model without editing its saved XML or geometry."""
    source = resolve_path(source)
    root = ET.parse(source).getroot()
    compiler = root.find("compiler")
    absolute = any(
        Path(value).is_absolute()
        for asset in root.findall("asset/*")
        for key, value in asset.attrib.items()
        if key.startswith("file") and value
    )
    absolute = absolute or (
        compiler is not None
        and any(Path(compiler.get(key, "")).is_absolute() for key in ("meshdir", "texturedir", "assetdir"))
    )
    if not absolute:
        return source
    signature = hashlib.sha256(source.read_bytes() + str(source).encode() + str(project_root()).encode()).hexdigest()
    output = project_root() / "generated/relocated_models" / signature / source.name
    if not output.is_file():
        write_portable_xml(root, output, source.parent)
    return output
