"""The paper's three machines, pinned by what they are rather than how they are written.

``experiments/machines.json`` holds a fingerprint of each compiled machine and a
hash of each input that built it. The fingerprint covers the model's physics
(tree, masses, inertia tensors, joints, geoms, contact filters, custom numerics),
rounded to 1e-9, so it survives a MuJoCo upgrade that reorders the XML or writes
its defaults differently, and fails on any change a policy could feel. The input
hashes are over parsed content, so editing a comment does not move them; they
are reported only to explain a mismatch.

    from draft.lineup import machines
    machines.check(["bear"])     # [] when the built machine matches the pin
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from ..paths import repo_root

ROOT = repo_root()
PIN = ROOT / "experiments" / "machines.json"
VARIANTS_DIR = ROOT / "experiments" / "quadruped_variants"
ROBOT_DIR = ROOT / "src" / "draft" / "robots" / "quadruped"
TRENDS_DIR = ROOT / "src" / "draft" / "trends" / "data"

#: Absolute rounding before hashing. Two MuJoCo versions compile the same XML to
#: within ~1e-16; a real edit moves something by far more than this.
DECIMALS = 9


def designs() -> list[str]:
    return sorted(p.stem for p in VARIANTS_DIR.glob("*.yaml"))


def xml_path(name: str) -> Path:
    return ROOT / "generated" / name / "quadruped.xml"


def input_files(names: list[str]) -> list[Path]:
    return [ROBOT_DIR / "parameters.yaml", ROBOT_DIR / "tree.yaml",
            TRENDS_DIR / "link_trends.json", TRENDS_DIR / "actuator_trends.json",
            TRENDS_DIR / "actuator_catalog.json",
            *(VARIANTS_DIR / f"{n}.yaml" for n in sorted(names))]


def content_hash(path: Path) -> str:
    """sha256 of a YAML or JSON file's parsed content, blind to comments and layout."""
    data = (json.loads(path.read_text()) if path.suffix == ".json"
            else yaml.safe_load(path.read_text()))
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def fingerprint(xml: Path) -> str:
    """sha256 over the compiled model's physics."""
    import mujoco

    m = mujoco.MjModel.from_xml_path(str(xml))
    obj = mujoco.mjtObj

    def names(kind, n):
        return [mujoco.mj_id2name(m, kind, i) or "" for i in range(n)]

    def rot(quats):
        out = np.zeros((len(quats), 9))
        for i, q in enumerate(quats):
            mujoco.mju_quat2Mat(out[i], q)
        return out

    # Inertia as a full tensor in the body frame: the principal axes of a
    # symmetric body are not unique, the tensor is.
    R = rot(m.body_iquat).reshape(-1, 3, 3)
    inertia = R @ (m.body_inertia[:, :, None] * np.eye(3)) @ R.transpose(0, 2, 1)

    fields = {
        "opt": [m.opt.timestep, *m.opt.gravity],
        "body_name": names(obj.mjOBJ_BODY, m.nbody),
        "body_parentid": m.body_parentid, "body_pos": m.body_pos,
        "body_rot": rot(m.body_quat), "body_mass": m.body_mass,
        "body_ipos": m.body_ipos, "body_inertia": inertia,
        "jnt_name": names(obj.mjOBJ_JOINT, m.njnt),
        "jnt_type": m.jnt_type, "jnt_bodyid": m.jnt_bodyid, "jnt_pos": m.jnt_pos,
        "jnt_axis": m.jnt_axis, "jnt_limited": m.jnt_limited, "jnt_range": m.jnt_range,
        "jnt_actfrclimited": m.jnt_actfrclimited, "jnt_actfrcrange": m.jnt_actfrcrange,
        "dof_armature": m.dof_armature, "dof_damping": m.dof_damping,
        "dof_frictionloss": m.dof_frictionloss,
        "geom_name": names(obj.mjOBJ_GEOM, m.ngeom),
        "geom_type": m.geom_type, "geom_bodyid": m.geom_bodyid, "geom_size": m.geom_size,
        "geom_pos": m.geom_pos, "geom_rot": rot(m.geom_quat),
        "geom_contype": m.geom_contype, "geom_conaffinity": m.geom_conaffinity,
        "geom_condim": m.geom_condim, "geom_friction": m.geom_friction,
        "exclude_signature": np.sort(m.exclude_signature),
        "numeric_name": names(obj.mjOBJ_NUMERIC, m.nnumeric),
        "numeric_data": m.numeric_data,
    }

    def canon(v):
        if isinstance(v, list) and (not v or isinstance(v[0], str)):
            return v
        # + 0.0 turns -0.0 into 0.0.
        return (np.round(np.asarray(v, dtype=float), DECIMALS) + 0.0).ravel().tolist()

    payload = {k: canon(v) for k, v in fields.items()}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def pin(names: list[str]) -> Path:
    """Write the fingerprint of each built machine, and its inputs, to the pin."""
    PIN.write_text(json.dumps({
        "machines": {n: {"xml": str(xml_path(n).relative_to(ROOT)),
                         "fingerprint": fingerprint(xml_path(n))} for n in sorted(names)},
        "inputs": {str(p.relative_to(ROOT)): content_hash(p) for p in input_files(names)},
    }, indent=1) + "\n")
    return PIN


def check(names: list[str]) -> list[str]:
    """What differs between the built machines and the pin; empty when they match."""
    if not PIN.exists():
        return [f"{PIN.relative_to(ROOT)} is missing"]
    want = json.loads(PIN.read_text())
    bad = []
    for n in names:
        if n not in want["machines"]:
            bad.append(f"{n} is not pinned")
            continue
        xml = xml_path(n)
        if not xml.exists():
            bad.append(f"{xml.relative_to(ROOT)} is missing")
        elif fingerprint(xml) != want["machines"][n]["fingerprint"]:
            bad.append(f"{xml.relative_to(ROOT)} differs from its pinned fingerprint")
    if bad:
        for p in input_files(names):
            rel = str(p.relative_to(ROOT))
            if rel in want["inputs"] and content_hash(p) != want["inputs"][rel]:
                bad.append(f"  input changed since the pin: {rel}")
    return bad
