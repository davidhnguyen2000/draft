"""Perturb a quadruped design's inertias by the error the twin study measured.

Applies `draft.twins.inertia_mismatch` ranges (per group: mass, radius of
gyration, COM) to a generated quadruped. Kept free of `mjlab` so it runs on CPU;
`draft.tasks.quadruped.inertia_dr` hands these ranges to the simulator.

Each draw is shared across all four legs of a replica: the error is systematic
in a trend, not an independent build tolerance per body.

Parametrisation uses the pseudo-inertia matrix (Rucker & Wensing 2022), which
stays physically consistent for any perturbation size:
    alpha  mass and inertia x e^(2a); COM and k_gyr fixed.  a = 0.5 ln(mass ratio)
    d      dilation: k_gyr and COM offset x e^d, mass fixed.
    t      COM translation. Composed COM is c' = e^d c + e^-a t.
For multi-body groups `d` is solved numerically (`_solve_d`), since dilation does
not move bodies relative to each other. Shear terms stay zero (not measured).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: The four groups. Bodies are assigned by `measure.py`'s rule restated on a
#: compiled model (checked in `tests/test_inertia_dr.py`).
GROUPS = ("trunk", "hip_link", "thigh", "shank")

#: joint name → (family, group when it has a same-family joint below it,
#:               group when it does not)
_JOINT_RULE = (
    (re.compile(r"hip_roll"), "hip", "hip_link", "thigh"),
    (re.compile(r"hip_pitch"), "hip", "hip_link", "thigh"),
    (re.compile(r"knee"), "knee", "shank", "shank"),
)


def body_groups(model) -> dict[str, list[int]]:
    """group → body ids for a compiled generated quadruped (unjointed bodies → trunk)."""
    import mujoco

    name = lambda i: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
    joint_of: dict[int, str] = {}
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        joint_of[int(model.jnt_bodyid[j])] = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_JOINT, j)

    def family(b: int) -> str | None:
        jn = joint_of.get(b)
        return next((f for pat, f, _, _ in _JOINT_RULE if pat.search(jn)), None) if jn else None

    # does some descendant of b carry a joint of b's own family?
    kids: dict[int, list[int]] = {}
    for i in range(1, model.nbody):
        kids.setdefault(int(model.body_parentid[i]), []).append(i)

    def has_child_of_family(b: int, fam: str) -> bool:
        stack = list(kids.get(b, []))
        while stack:
            c = stack.pop()
            if family(c) == fam:
                return True
            stack.extend(kids.get(c, []))
        return False

    out: dict[int, str] = {}
    for i in range(1, model.nbody):
        jn = joint_of.get(i)
        rule = next((r for r in _JOINT_RULE if jn and r[0].search(jn)), None)
        if rule is not None:
            _, fam, housing, limb = rule
            out[i] = housing if has_child_of_family(i, fam) else limb
        else:
            out[i] = out.get(int(model.body_parentid[i]), "trunk")

    groups: dict[str, list[int]] = {g: [] for g in GROUPS}
    for i, g in out.items():
        if model.body_mass[i] > 0:
            groups.setdefault(g, []).append(i)
    return {g: v for g, v in groups.items() if v}


def group_inertia(model, body_ids, data=None) -> tuple[float, np.ndarray, float]:
    """(mass, world COM, k_gyr) of a set of bodies at the rest pose.

    `k_gyr = sqrt(tr(I) / 2m)` about the set's own COM, as `draft.twins.measure` reports.
    """
    import mujoco
    if data is None:
        data = mujoco.MjData(model)
        data.qpos[:] = model.qpos0
        mujoco.mj_forward(model, data)
    mass = float(sum(model.body_mass[b] for b in body_ids))
    com = sum(float(model.body_mass[b]) * np.array(data.xipos[b])
              for b in body_ids) / mass
    I = np.zeros((3, 3))
    for b in body_ids:
        R = np.array(data.ximat[b]).reshape(3, 3)
        Ib = R @ np.diag(np.asarray(model.body_inertia[b], float)) @ R.T
        r = np.array(data.xipos[b]) - com
        I += Ib + float(model.body_mass[b]) * (float(r @ r) * np.eye(3) - np.outer(r, r))
    return mass, com, float(np.sqrt(max(np.trace(I), 0.0) / (2.0 * mass)))


def limb_sets(model, group: str, body_ids) -> list[list[int]]:
    """A group's bodies split per leg (k_gyr is per limb); the trunk stays whole."""
    import mujoco
    if group == "trunk":
        return [list(body_ids)]
    by: dict[str, list[int]] = {}
    for b in body_ids:
        n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        by.setdefault(n[:2], []).append(b)
    return list(by.values())


def leg_side(model, body_id: int) -> float:
    """+1 for left-side bodies and the trunk, -1 for right-side ones (outboard = ±y)."""
    import mujoco
    n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
    return -1.0 if n[:2] in ("fr", "rr") else 1.0


# ── the pseudo-inertia perturbation, in numpy ────────────────────────────────

def _J(mass: float, com: np.ndarray, I_com: np.ndarray) -> np.ndarray:
    """The 4x4 pseudo-inertia of one body, about its own body frame origin."""
    I_o = I_com + mass * (float(com @ com) * np.eye(3) - np.outer(com, com))
    S = 0.5 * np.trace(I_o) * np.eye(3) - I_o          # second moment matrix
    J = np.zeros((4, 4))
    J[:3, :3], J[:3, 3], J[3, :3], J[3, 3] = S, mass * com, mass * com, mass
    return J


def _from_J(J: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    mass = float(J[3, 3])
    com = J[:3, 3] / mass
    I_o = np.trace(J[:3, :3]) * np.eye(3) - J[:3, :3]
    I_com = I_o - mass * (float(com @ com) * np.eye(3) - np.outer(com, com))
    return mass, com, I_com


def perturb_body(mass: float, com: np.ndarray, I_com: np.ndarray,
                 alpha: float, d: float, t: np.ndarray):
    """One body's (mass, COM, inertia about the COM) after the perturbation."""
    L = np.linalg.cholesky(_J(mass, np.asarray(com, float), np.asarray(I_com, float)))
    U = np.diag([np.exp(alpha + d)] * 3 + [np.exp(alpha)])
    U[:3, 3] = t
    Ln = U @ L
    return _from_J(Ln @ Ln.T)


# ── the measured spec, resolved onto one robot ───────────────────────────────

@dataclass(frozen=True)
class GroupRange:
    """What one group's perturbation is drawn from, for one robot."""
    group: str
    body_ids: tuple[int, ...]
    alpha: tuple[float, float]
    d: tuple[float, float]
    #: (fore, outboard, up) COM shift in METRES for this robot, lo and hi.
    com_lo: np.ndarray
    com_hi: np.ndarray


def load_spec(path: Path | str | None = None) -> dict:
    """Load `experiments/twin_inertia_mismatch.json` (or `path`)."""
    if path:
        p = Path(path)
    else:
        try:
            from draft.paths import repo_root
            root = repo_root()
        except ModuleNotFoundError:
            root = next(d for d in Path(__file__).resolve().parents
                        if (d / "experiments").is_dir())
        p = root / "experiments" / "twin_inertia_mismatch.json"
    if not p.exists():
        raise FileNotFoundError(
            f"no inertia-mismatch spec at {p} — run "
            f"scripts/make_twins.py")
    return json.loads(p.read_text())


def _solve_d(model, group: str, body_ids, want_log_k: float, tol: float = 1e-4) -> float:
    """The dilation `d` that gives this group the requested log `k_gyr` ratio.

    Needed because `d` does not move bodies apart, so multi-body groups fall
    short of `e^d`. Secant iteration on `log k(d)`.
    """
    import copy
    if abs(want_log_k) < 1e-12:
        return 0.0
    one = limb_sets(model, group, body_ids)[0]      # the legs are identical
    k0 = group_inertia(model, one)[2]

    def achieved(d: float) -> float:
        m2 = copy.deepcopy(model)
        rng = [GroupRange("x", tuple(body_ids), (0.0, 0.0), (d, d),
                          np.zeros(3), np.zeros(3))]
        apply_numpy(m2, rng, {"x": (0.0, d, np.zeros(3))})
        return float(np.log(group_inertia(m2, one)[2] / k0))

    x0, x1 = want_log_k, want_log_k * 1.2 + 1e-3
    f0, f1 = achieved(x0) - want_log_k, achieved(x1) - want_log_k
    for _ in range(30):
        if abs(f1) < tol or abs(f1 - f0) < 1e-15:
            break
        x0, x1 = x1, x1 - f1 * (x1 - x0) / (f1 - f0)
        f0, f1 = f1, achieved(x1) - want_log_k
    return float(x1)


def resolve(model, leg_length: float, spec: dict | None = None) -> list[GroupRange]:
    """The spec resolved onto this robot: body ids, metres, and solved dilations."""
    spec = spec or load_spec()
    groups = body_groups(model)
    out = []
    for g, r in spec["ranges"].items():
        if g not in groups:
            continue
        ids = groups[g]
        out.append(GroupRange(
            group=g, body_ids=tuple(ids),
            alpha=(0.5 * r["log_mass_ratio"][0], 0.5 * r["log_mass_ratio"][1]),
            d=(_solve_d(model, g, ids, r["log_k_gyr_ratio"][0]),
               _solve_d(model, g, ids, r["log_k_gyr_ratio"][1])),
            com_lo=np.asarray(r["com_shift_per_leg_lo"], float) * leg_length,
            com_hi=np.asarray(r["com_shift_per_leg_hi"], float) * leg_length))
    return out


def sample(ranges: list[GroupRange], rng) -> dict[str, tuple[float, float, np.ndarray]]:
    """One replica's draw, one per group: group → (alpha, d, root-frame COM shift)."""
    out = {}
    for r in ranges:
        out[r.group] = (float(rng.uniform(*r.alpha)),
                        float(rng.uniform(*r.d)),
                        rng.uniform(r.com_lo, r.com_hi))
    return out


def apply_numpy(model, ranges: list[GroupRange], draw: dict) -> None:
    """Write one draw into a compiled `MjModel` in place.

    Body frames are root-aligned at neutral, so the COM shift only needs the y fold.
    """
    import mujoco

    for r in ranges:
        alpha, d, shift = draw[r.group]
        for b in r.body_ids:
            m0 = float(model.body_mass[b])
            ipos = np.asarray(model.body_ipos[b], float).copy()
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, np.asarray(model.body_iquat[b], float))
            R = R.reshape(3, 3)
            I0 = R @ np.diag(np.asarray(model.body_inertia[b], float)) @ R.T

            want = np.array([shift[0], leg_side(model, b) * shift[1], shift[2]])
            # c' = e^d c + e^-a t  =>  t = e^a (want - (e^d - 1) c)
            t = np.exp(alpha) * (want - (np.exp(d) - 1.0) * ipos)
            m1, c1, I1 = perturb_body(m0, ipos, I0, alpha, d, t)

            w, V = np.linalg.eigh(I1)
            if np.linalg.det(V) < 0:
                V[:, 0] = -V[:, 0]
            q = np.zeros(4)
            mujoco.mju_mat2Quat(q, V.flatten())
            model.body_mass[b] = m1
            model.body_ipos[b] = c1
            model.body_inertia[b] = np.maximum(w, 1e-12)
            model.body_iquat[b] = q
