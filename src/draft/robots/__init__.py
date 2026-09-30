"""The shipped robot specifications, one directory per robot.

Each holds `parameters.yaml` (flat numbers), `tree.yaml` (kinematic tree with
expressions over those numbers) and optionally `mesh.yaml` (root-body
cross-sections). Any directory with both is a robot; use `robot_dir(name)`.
"""

from pathlib import Path

#: Directory holding one subdirectory per robot.
ROBOTS_DIR = Path(__file__).resolve().parent


def robot_dir(name: str) -> Path:
    """The directory for one robot; FileNotFoundError lists what is available."""
    d = ROBOTS_DIR / name
    if not (d / "parameters.yaml").exists():
        raise FileNotFoundError(
            f"no robot {name!r} in {ROBOTS_DIR}. Available: {', '.join(available())}")
    return d


def available() -> list[str]:
    """Every robot that has a parameter table and a tree beside it."""
    return sorted(
        d.name for d in ROBOTS_DIR.iterdir()
        if d.is_dir() and (d / "parameters.yaml").exists()
        and ((d / "tree.yaml").exists() or (d / "tree.py").exists()))
