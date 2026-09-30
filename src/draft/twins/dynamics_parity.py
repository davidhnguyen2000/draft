"""Rigid-body agreement between a twin and its target, with no policy.

Both models are put in the same state (joints matched by name, base at the
origin, zero velocity) and compared on M(q), g(q), and per-link mass and inertia.
Used by `scripts/figures/fig5_twin_dynamics.py`; needs `scripts/make_twins.py`
(build and adapt steps) to have run.
"""
from __future__ import annotations

import mujoco
import numpy as np

from draft.paths import repo_root

TWINS = repo_root() / "generated" / "twins"
#: Sampling half-range for joints MuJoCo marks unlimited.
_FREE_JOINT_SPAN = np.pi / 2

#: Axis error (deg) above which a twin joint counts as misaligned. Sits in a gap
#: of the observed distribution: any threshold from ~7 to 15 selects the same joints.
AXIS_MISALIGNED_DEG = 8.0


# ── the two models ────────────────────────────────────────────────────────────

def _stem(key: str) -> str:
    """The MJCF filename built for this key (quadruped or humanoid)."""
    return "quadruped.xml" if (TWINS / key / "quadruped.xml").exists() else "humanoid.xml"


def _model(key: str, side: str) -> mujoco.MjModel:
    stem = _stem(key)
    rel = stem if side == "twin" else f"shipped_rl/{stem}"
    p = TWINS / key / rel
    if not p.exists():
        raise FileNotFoundError(
            f"{key}: no {side} model at {p.relative_to(repo_root())} — "
            f"run scripts/make_twins.py first")
    return mujoco.MjModel.from_xml_path(str(p))


def _hinges(m):
    return {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j): j
            for j in range(m.njnt) if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE}


def _shared_ranges(tw, sh, order) -> np.ndarray:
    """Per joint, the overlap of the two robots' limits — reachable on both."""
    jt, js = _hinges(tw), _hinges(sh)
    out = np.zeros((len(order), 2))
    for i, n in enumerate(order):
        a, b = jt[n], js[n]
        ra = tw.jnt_range[a] if tw.jnt_limited[a] else np.array([-_FREE_JOINT_SPAN,
                                                                 _FREE_JOINT_SPAN])
        rb = sh.jnt_range[b] if sh.jnt_limited[b] else np.array([-_FREE_JOINT_SPAN,
                                                                 _FREE_JOINT_SPAN])
        lo, hi = max(ra[0], rb[0]), min(ra[1], rb[1])
        if hi <= lo:                       # ranges do not overlap at all
            lo = hi = 0.5 * (max(ra[0], rb[0]) + min(ra[1], rb[1]))
        out[i] = (lo, hi)
    return out


def _set(m, d, order, q) -> None:
    d.qpos[:] = 0.0
    if m.nq >= 7 and m.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE:
        d.qpos[3] = 1.0
    jm = _hinges(m)
    for n, v in zip(order, q):
        d.qpos[int(m.jnt_qposadr[jm[n]])] = v
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)


def _dense_mass_matrix(m, d) -> np.ndarray:
    """Dense joint-space mass matrix on MuJoCo before and after 3.14.

    3.14 renamed `qM` and reordered `mj_fullM`'s arguments; dispatch on the field.
    """
    full = np.zeros((m.nv, m.nv))
    if hasattr(d, "qM"):
        mujoco.mj_fullM(m, full, d.qM)         # MuJoCo < 3.14
    else:
        mujoco.mj_fullM(m, d, full)            # MuJoCo >= 3.14
    return full


def _M_g(m, d, order, drop_armature):
    full = _dense_mass_matrix(m, d)
    if drop_armature:
        full = full - np.diag(np.array(m.dof_armature, float))
    jm = _hinges(m)
    idx = [int(m.jnt_dofadr[jm[n]]) for n in order]
    return full[np.ix_(idx, idx)], np.array([d.qfrc_bias[i] for i in idx])


# ── whole-robot agreement ─────────────────────────────────────────────────────

def axis_angles(key: str) -> np.ndarray:
    """Per shared joint, the angle between the twin's axis and the target's (deg)."""
    tw, sh = _model(key, "twin"), _model(key, "shipped")

    def ax(m):
        d = mujoco.MjData(m); d.qpos[:] = m.qpos0; mujoco.mj_forward(m, d)
        R = np.array(d.xmat[1]).reshape(3, 3)
        return {n: R.T @ np.array(d.xaxis[j]) for n, j in _hinges(m).items()}
    a, b = ax(tw), ax(sh)
    common = [n for n in a if n in b]
    return np.degrees(np.arccos(np.clip([abs(float(a[n] @ b[n])) for n in common], -1, 1)))


def misaligned(key: str) -> list[str]:
    """Joints whose twin axis is off by at least `AXIS_MISALIGNED_DEG`, worst first."""
    tw, sh = _model(key, "twin"), _model(key, "shipped")
    order = [x for x in _hinges(tw) if x in _hinges(sh)]
    ang = axis_angles(key)
    bad = [(a, n) for n, a in zip(order, ang) if a >= AXIS_MISALIGNED_DEG]
    return [n for _, n in sorted(bad, reverse=True)]


def per_config(key: str, n: int, seed: int = 0, drop_armature: bool = True,
               hold: list[str] | None = None) -> dict:
    """Relative M and g error at `n` random configurations.

    `drop_armature` removes the rotor term from both diagonals: the adapter copies
    the twin's armature onto the target, so it would only inflate agreement.
    `hold` pins the named joints at neutral on both models instead of sweeping them.
    """
    tw, sh = _model(key, "twin"), _model(key, "shipped")
    order = [x for x in _hinges(tw) if x in _hinges(sh)]
    lo_hi = _shared_ranges(tw, sh, order)
    # Neutral, clipped into the shared range.
    frozen = np.array([x in set(hold or ()) for x in order])
    q_hold = np.clip(0.0, lo_hi[:, 0], lo_hi[:, 1])
    dt, ds = mujoco.MjData(tw), mujoco.MjData(sh)
    rng = np.random.default_rng(seed)
    err, gerr = [], []
    for _ in range(n):
        # Always draw, so held and unheld runs share poses in the free joints.
        q = np.where(frozen, q_hold, rng.uniform(lo_hi[:, 0], lo_hi[:, 1]))
        _set(tw, dt, order, q); _set(sh, ds, order, q)
        Mt, gt = _M_g(tw, dt, order, drop_armature)
        Ms, gs = _M_g(sh, ds, order, drop_armature)
        err.append(np.linalg.norm(Mt - Ms) / np.linalg.norm(Ms))
        # qvel is zero, so qfrc_bias is gravity only; ||g|| never nears zero.
        gerr.append(np.linalg.norm(gt - gs) / np.linalg.norm(gs))
    return {"key": key, "order": order, "err": np.array(err),
            "g_err": np.array(gerr), "drop_armature": drop_armature,
            "held": [x for x in order if x in set(hold or ())]}


# ── link by link ──────────────────────────────────────────────────────────────

def _bodies(m) -> dict[str, int]:
    return {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i): i for i in range(1, m.nbody)}


def _frames(m):
    """Per body at the rest pose: (R, p, COM) in the root frame."""
    d = mujoco.MjData(m)
    d.qpos[:] = 0.0
    if m.nq >= 7 and m.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE:
        d.qpos[3] = 1.0
    mujoco.mj_forward(m, d)
    Rr = np.array(d.xmat[1]).reshape(3, 3)
    pr = np.array(d.xpos[1])
    out = {}
    for n, i in _bodies(m).items():
        R = Rr.T @ np.array(d.xmat[i]).reshape(3, 3)
        p = Rr.T @ (np.array(d.xpos[i]) - pr)
        c = Rr.T @ (np.array(d.xipos[i]) - pr)
        out[n] = (R, p, c)
    return out


def _tensor(m, bid: int) -> np.ndarray:
    """Full inertia tensor of one body, in that body's own frame."""
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(m.body_iquat[bid], float))
    R = R.reshape(3, 3)
    return R @ np.diag(np.asarray(m.body_inertia[bid], float)) @ R.T


def link_table(key: str) -> dict:
    """Every body both models share: mass, COM offset and principal moments."""
    tw, sh = _model(key, "twin"), _model(key, "shipped")
    bt, bs = _bodies(tw), _bodies(sh)
    ft, fs = _frames(tw), _frames(sh)
    rows = []
    for n in [n for n in bt if n in bs]:
        it, i_s = bt[n], bs[n]
        mt, ms = float(tw.body_mass[it]), float(sh.body_mass[i_s])
        if ms <= 1e-9 or mt <= 1e-9:
            continue                                   # massless marker frames
        It, Is = _tensor(tw, it), _tensor(sh, i_s)
        # COM in the root frame: body_ipos is convention-dependent, and not every
        # body has a joint to anchor to.
        ut, us = ft[n][2], fs[n][2]
        rows.append({
            "name": n, "m_t": mt, "m_s": ms,
            "com_off_mm": float(np.linalg.norm(ut - us)) * 1e3,
            "com_arm_s_mm": float(np.linalg.norm(us)) * 1e3,
            "eig_t": np.sort(np.linalg.eigvalsh(It)),
            "eig_s": np.sort(np.linalg.eigvalsh(Is)),
            "I_rel": float(np.linalg.norm(It - Is) / max(np.linalg.norm(Is), 1e-15)),
        })
    return {"key": key, "rows": rows}
