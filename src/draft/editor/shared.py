"""Shared editor helpers: colours, YAML, and robot discovery."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


# ── Color helpers ──────────────────────────────────────────────────────────────

def rgba_to_hex(s: str) -> str:
    try:
        parts = s.strip().split()
        r, g, b = (int(float(p) * 255) for p in parts[:3])
        return f"#{r:02x}{g:02x}{b:02x}"
    except (ValueError, IndexError):
        return "#808080"


def hex_to_rgba(h: str, alpha: float = 1.0) -> str:
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255
    return f"{r:.3f} {g:.3f} {b:.3f} {alpha:.1f}"


def parse_alpha(s: str) -> float:
    try:
        return float(s.strip().split()[3])
    except (ValueError, IndexError):
        return 1.0


# ── YAML helpers ───────────────────────────────────────────────────────────────

def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"YAML not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in: {path}")
    return data


def to_python(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays to plain Python types for yaml.safe_dump."""
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_python(v) for v in obj]
    return obj.item() if hasattr(obj, "item") else obj


# ── Robot type config & discovery ─────────────────────────────────────────────

@dataclass
class RobotTypeConfig:
    name: str
    yaml_path: Path
    mjcf_filename: str
    output_prefix: str
    robot_dir: Path = None  # source folder (src/draft/robots/<name>/)
    #: Overrides on `parameters.yaml` a fresh start applies: the alphabetically
    #: first `presets/*.yaml`, or None to start from the shipped table.
    preset_path: Path | None = None

    @property
    def preset_name(self) -> str | None:
        return self.preset_path.stem if self.preset_path else None


def discover_robot_types(repo_root: Path) -> dict[str, RobotTypeConfig]:
    """Every robot under src/draft/robots/ with a tree and a `parameters.yaml`."""
    src_dir = repo_root / "src" / "draft" / "robots"
    configs: dict[str, RobotTypeConfig] = {}
    if not src_dir.is_dir():
        return configs
    for d in sorted(src_dir.iterdir()):
        if not d.is_dir():
            continue
        has_tree = (d / "tree.yaml").is_file() or (d / "tree.py").is_file()
        if not (d / "parameters.yaml").is_file() or not has_tree:
            continue
        presets = sorted((d / "presets").glob("*.yaml")) if (d / "presets").is_dir() else []
        configs[d.name] = RobotTypeConfig(
            name=d.name,
            yaml_path=d / "parameters.yaml",
            mjcf_filename=f"{d.name}.xml",
            output_prefix=d.name,
            robot_dir=d,
            preset_path=presets[0] if presets else None,
        )
    return configs


def new_timestamped_dir(base: Path, prefix: str) -> Path:
    """Create and return a new uniquely-named subdirectory under base."""
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{prefix}_{timestamp}"
    candidate = base / name
    suffix = 1
    while candidate.exists():
        name = f"{prefix}_{timestamp}_{suffix:02d}"
        candidate = base / name
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate
