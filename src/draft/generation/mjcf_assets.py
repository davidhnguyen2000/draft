"""Load an MJCF whose mesh assets may have changed on disk since the last load.

MuJoCo caches meshes by path for the life of the process; passing the asset
bytes explicitly bypasses that cache, so a regenerated root mesh is picked up.
"""
from __future__ import annotations

import os
from pathlib import Path

#: What MuJoCo may reference out of a model's own directory.
ASSET_SUFFIXES = {".stl", ".obj", ".msh", ".png", ".skn"}

#: What makes a directory another model's own, rather than this one's assets.
MODEL_SUFFIXES = {".xml", ".urdf", ".mjcf"}


def read_assets(path: Path) -> dict:
    """Every asset beside a model, as `{name: bytes}`, read now.

    Subdirectories are walked, except any holding its own model (.xml/.urdf/.mjcf):
    MuJoCo matches assets by basename, so another robot's meshes would shadow
    this one's. The shallowest file wins the bare-name alias.
    """
    path = Path(path).resolve()
    root = path.parent
    assets: dict = {}
    for folder, subdirs, files in os.walk(root):
        folder = Path(folder)
        subdirs[:] = sorted(s for s in subdirs
                            if not _holds_model(folder / s))
        for name in sorted(files):
            f = folder / name
            if f.suffix.lower() not in ASSET_SUFFIXES:
                continue
            rel = str(f.relative_to(root))
            assets[rel] = f.read_bytes()
            assets.setdefault(f.name, assets[rel])
    return assets


def _holds_model(folder: Path) -> bool:
    """Is this directory some other model's own folder?"""
    return any(f.suffix.lower() in MODEL_SUFFIXES
               for f in folder.iterdir() if f.is_file())


def load_model(path: Path):
    """`MjModel.from_xml_path`, reading the assets rather than trusting the cache."""
    import mujoco
    path = Path(path).resolve()
    try:
        assets = read_assets(path)
        if assets:
            return mujoco.MjModel.from_xml_string(path.read_text(), assets)
    except Exception:                                   # noqa: BLE001
        pass                                            # fall back rather than fail
    return mujoco.MjModel.from_xml_path(str(path))


def load_spec(path: Path):
    """`MjSpec.from_file`, reading the assets rather than trusting the cache."""
    import mujoco
    path = Path(path).resolve()
    try:
        assets = read_assets(path)
        if assets:
            return mujoco.MjSpec.from_string(path.read_text(), assets=assets)
    except Exception:                                   # noqa: BLE001
        pass
    return mujoco.MjSpec.from_file(str(path))
