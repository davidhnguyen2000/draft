"""Per-group inertia error of a quadruped twin against its target.

For each anatomical group (trunk, hip_link, thigh, shank), all as target / twin:
    `mass_ratio`   mass.
    `k_gyr_ratio`  radius of gyration about the group's own COM.
    `com_shift`    COM offset in the root frame at the neutral pose, per leg length.
The root frame is used for the COM because offsets from the driving joint mix
kinematics into the comparison. Quadrupeds only: the two body plans err differently, so
they are not pooled.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .measure import (FAMILY, FAMILY_GROUPS, RobotMeasurer, _load_mjcf,
                      _load_urdf, detect_side)
from .targets import GENERATED_QUADRUPED_ROLES, REGISTRY

#: Quadruped groups, head to toe (`aux`, for unclaimed bodies, is excluded).
QUAD_GROUPS = ("trunk", "hip_link", "thigh", "shank")

#: Groups that exist once per leg. `trunk` is the robot's, and is compared whole.
PER_LIMB = ("hip_link", "thigh", "shank")

#: Leg quadrant → y sign that folds left and right legs into one frame. Front/rear
#: is not folded: vendor URDFs do not mirror rear legs fore-aft.
_MIRROR = {"FL": 1.0, "FR": -1.0, "RL": 1.0, "RR": -1.0}


@dataclass(frozen=True)
class GroupError:
    """One group's target-over-twin error (1.0 / zero shift = exact)."""
    group: str
    mass_ratio: float
    k_gyr_ratio: float
    #: (fore, outboard, up) in the mirrored root frame, per unit of leg length.
    com_shift: np.ndarray
    #: Masses behind the ratio, for reporting.
    twin_mass_kg: float
    target_mass_kg: float

    def as_dict(self) -> dict:
        return {"group": self.group,
                "mass_ratio": float(self.mass_ratio),
                "k_gyr_ratio": float(self.k_gyr_ratio),
                "com_shift_per_leg": [float(x) for x in self.com_shift],
                "twin_mass_kg": float(self.twin_mass_kg),
                "target_mass_kg": float(self.target_mass_kg)}


# ── group composition ────────────────────────────────────────────────────────

def _groups_of(segs, roles) -> dict[int, str]:
    """Segment index → anatomical group, restating `RobotMeasurer.measure`'s rule."""
    def ancestors(i):
        while segs[i].parent >= 0:
            i = segs[i].parent
            yield i

    has_child = {i: False for i in roles}
    for i in roles:
        for a in ancestors(i):
            if a in roles:
                if FAMILY[roles[a]] == FAMILY[roles[i]]:
                    has_child[a] = True
                break

    out = {}
    for i, s in enumerate(segs):
        if i in roles:
            limb, housing = FAMILY_GROUPS[FAMILY[roles[i]]]
            out[i] = housing if has_child[i] else limb
        elif i == 0:
            out[i] = "trunk"
        else:
            out[i] = out.get(s.parent, "aux")
    return out


def _compose(segs, idx: list[int]) -> tuple[float, np.ndarray, np.ndarray]:
    """(mass, world COM, inertia about that COM) for a set of segments."""
    mass = sum(segs[i].mass for i in idx)
    com = sum(segs[i].mass * segs[i].com for i in idx) / mass
    I = np.zeros((3, 3))
    for i in idx:
        d = segs[i].com - com
        I += segs[i].inertia + segs[i].mass * (float(d @ d) * np.eye(3)
                                               - np.outer(d, d))
    return float(mass), np.asarray(com, float), I


def _k_gyr(mass: float, I: np.ndarray) -> float:
    """sqrt(tr(I) / 2m) — the RMS distance of the mass from its own COM."""
    return math.sqrt(max(float(np.trace(I)), 0.0) / (2.0 * mass))


def _measure(path: Path, role_map, category: str = "quadruped") -> dict:
    """Per group: mass, k_gyr and COM in the mirrored root frame, averaged over legs."""
    path = Path(path)
    sk = _load_urdf(path) if path.suffix.lower() == ".urdf" else _load_mjcf(path)
    segs = sk.segs
    M = RobotMeasurer(role_map, category)
    roles = {i: r for i, s in enumerate(segs) if s.actuated
             for r in [M.role_of(s.name)] if r is not None}
    group_of = _groups_of(segs, roles)

    # side, inherited down the tree like the group
    side_of: dict = {}
    for i, s in enumerate(segs):
        own = detect_side(s.name, category) if i in roles else None
        side_of[i] = own if own is not None else side_of.get(s.parent)

    # Origin: the trunk (first massive segment; segs[0] is a massless base).
    # Only differences are used, so standing height drops out.
    trunk_i = next(i for i, s_ in enumerate(segs) if s_.mass > 0)
    root = np.asarray(segs[trunk_i].pos, float)

    buckets: dict[tuple[str, str | None], list[int]] = {}
    for i, s in enumerate(segs):
        if s.mass <= 0:
            continue
        g = group_of[i]
        key = (g, side_of.get(i) if g in PER_LIMB else None)
        buckets.setdefault(key, []).append(i)

    out: dict[str, dict] = {}
    for g in QUAD_GROUPS:
        sides = [k[1] for k in buckets if k[0] == g]
        if not sides:
            continue
        masses, ks, coms = [], [], []
        for sd in sides:
            mass, com, I = _compose(segs, buckets[(g, sd)])
            sy = _MIRROR.get(sd or "FL", 1.0)
            rel = com - root
            masses.append(mass)
            ks.append(_k_gyr(mass, I))
            coms.append(np.array([rel[0], sy * rel[1], rel[2]]))
        out[g] = {"mass": float(np.mean(masses)),
                  "k_gyr": float(np.mean(ks)),
                  "com": np.mean(coms, axis=0),
                  "n_limbs": len(sides)}
    return out


def _leg_length(meas_json: dict) -> float:
    f = meas_json["target"]["features_m"]
    return float(f["thigh"] + f["shank"])


# ── the pair, and the population ─────────────────────────────────────────────

def pair_error(key: str, twins_dir: Path) -> tuple[dict[str, GroupError], float]:
    """One twin pair's per-group error, and the leg length it is scaled by."""
    d = json.loads((Path(twins_dir) / key / "measurements.json").read_text())
    target = next(t for t in REGISTRY if t.key == key)
    if target.category != "quadruped":
        # Role map and groups below are quadruped-only.
        raise ValueError(f"{key} is a {target.category}; this module is "
                         f"quadruped-only (see the module docstring)")
    tw = _measure(Path(d["twin"]["source"]), GENERATED_QUADRUPED_ROLES)
    sh = _measure(Path(d["target"]["source"]), target.role_map)
    leg = _leg_length(d)
    out = {}
    for g in QUAD_GROUPS:
        if g not in tw or g not in sh:
            continue
        out[g] = GroupError(
            group=g,
            mass_ratio=sh[g]["mass"] / tw[g]["mass"],
            k_gyr_ratio=sh[g]["k_gyr"] / tw[g]["k_gyr"],
            com_shift=(sh[g]["com"] - tw[g]["com"]) / leg,
            twin_mass_kg=tw[g]["mass"], target_mass_kg=sh[g]["mass"])
    return out, leg


def population(keys, twins_dir: Path) -> dict[str, list[GroupError]]:
    """group → one `GroupError` per twin pair that has that group."""
    out: dict[str, list[GroupError]] = {}
    for k in keys:
        errs, _ = pair_error(k, twins_dir)
        for g, e in errs.items():
            out.setdefault(g, []).append(e)
    return out


def ranges(pop: dict[str, list[GroupError]]) -> dict[str, dict]:
    """Per group, the min/max interval over the observed pairs.

    Ratios are bracketed in log space (x1.4 and x1/1.4 are equal errors); COM
    shifts linearly.
    """
    out = {}
    for g, es in pop.items():
        lm = [math.log(e.mass_ratio) for e in es]
        lk = [math.log(e.k_gyr_ratio) for e in es]
        cs = np.array([e.com_shift for e in es])
        out[g] = {
            "n": len(es),
            "log_mass_ratio": [min(lm), max(lm)],
            "log_k_gyr_ratio": [min(lk), max(lk)],
            "com_shift_per_leg_lo": cs.min(axis=0).tolist(),
            "com_shift_per_leg_hi": cs.max(axis=0).tolist(),
            "mass_ratio": [math.exp(min(lm)), math.exp(max(lm))],
            "k_gyr_ratio": [math.exp(min(lk)), math.exp(max(lk))],
        }
    return out


# ── reporting ────────────────────────────────────────────────────────────────

def print_table(keys, twins: Path) -> None:
    print(f"\n{'':6s}{'group':14s}{'twin kg':>9s}{'real kg':>9s}{'x mass':>8s}"
          f"{'x k_gyr':>9s}{'fore':>8s}{'out':>8s}{'up':>8s}{'|d| mm':>9s}")
    print("-" * 88)
    for k in keys:
        errs, leg = pair_error(k, twins)
        for g in QUAD_GROUPS:
            if g not in errs:
                continue
            e = errs[g]
            c = e.com_shift
            print(f"{k.upper():6s}{g:14s}{e.twin_mass_kg:9.3f}{e.target_mass_kg:9.3f}"
                  f"{e.mass_ratio:8.2f}{e.k_gyr_ratio:9.2f}"
                  f"{c[0]:8.4f}{c[1]:8.4f}{c[2]:8.4f}"
                  f"{float(np.linalg.norm(c)) * leg * 1e3:9.1f}")
    print("\nratios are REAL / TWIN — the correction, not the error. COM shift is in\n"
          "leg lengths, in the robot's root frame, left/right folded.")


def print_bodies(keys, twins: Path) -> None:
    """Every shared body, ranked by how much of the robot's mass it misplaces."""
    from draft.twins.dynamics_parity import link_table
    for k in keys:
        L = link_table(k)
        tot = sum(r["m_s"] for r in L["rows"])
        print(f"\n=== {k.upper()} — per shared body ===")
        print(f"{'body':26s}{'real kg':>9s}{'twin kg':>9s}{'x mass':>8s}"
              f"{'|dI|/|I|':>10s}{'dCOM mm':>9s}{'dm/M':>8s}")
        for r in sorted(L["rows"], key=lambda r: -abs(r["m_t"] - r["m_s"])):
            print(f"{r['name']:26s}{r['m_s']:9.3f}{r['m_t']:9.3f}"
                  f"{r['m_t'] / r['m_s']:8.2f}{r['I_rel'] * 100:9.1f}%"
                  f"{r['com_off_mm']:9.1f}{(r['m_t'] - r['m_s']) / tot:8.3f}")
        print("  NOTE: a body-name match is not an anatomical match. The generated\n"
              "  tree gives each actuator its own body, so a Go2 thigh (one link,\n"
              "  motor inside) matches only the twin's bare thigh shell and reads\n"
              "  0.09x. The group table above is the like-for-like comparison.")
