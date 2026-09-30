"""The single place checkout paths are resolved.

Package data (``draft.trends``, ``draft.robots``) is found through those
packages. Repository paths (description cache, ``generated/``, ``experiments/``)
exist only in a checkout; :func:`repo_root` raises outside one.
"""

from pathlib import Path

#: ``src/draft`` — the installed package itself.
PACKAGE_DIR = Path(__file__).resolve().parent

#: The checkout root, when there is one. A wheel installed into site-packages
#: has no ``pyproject.toml`` two levels up, and this is then ``None``.
_candidate = PACKAGE_DIR.parent.parent
REPO_ROOT = _candidate if (_candidate / "pyproject.toml").exists() else None


def repo_root() -> Path:
    """The checkout root. Raises RuntimeError outside a checkout rather than guess."""
    if REPO_ROOT is None:
        raise RuntimeError(
            "no Draft checkout found: this needs repository files (the vendor "
            "description cache, configs/, generated/) that an installed "
            "package does not carry. Run from a clone of the repository.")
    return REPO_ROOT


def description_cache() -> Path:
    """Vendor robot descriptions the twins are measured against."""
    return repo_root() / "datasets" / "robot_descriptions" / "cache_extra"


def generated_dir() -> Path:
    """Where generators write. Gitignored; created on demand."""
    d = repo_root() / "generated"
    d.mkdir(parents=True, exist_ok=True)
    return d


def configs_dir() -> Path:
    """Experiment configuration that is not part of any robot's own spec."""
    return repo_root() / "experiments"
