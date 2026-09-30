"""One measurement schema, applied identically to a target robot and its twin.

Measures per-joint rows (role, side, placement, torque, speed), body mass per
anatomical group, and the scalar lengths the parametric tree is parameterised
by. A body belongs to the group of the joint that drives it (fixed sub-bodies
merged in); a joint drives a housing if a same-family joint sits below it,
otherwise the limb link. URDF targets load via Pinocchio, MJCF twins via MuJoCo
(Pinocchio's MJCF loader does not compute mesh inertia); both reduce to the
same `_Skeleton`.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from draft.generation.mjcf_assets import (  # noqa: E402
    load_model, load_spec, read_assets)

# ── Canonical roles ───────────────────────────────────────────────────────────
HUMANOID_ROLES = [
    "torso_yaw", "torso_roll", "torso_pitch",
    "hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll",
    "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
    "wrist_yaw", "wrist_pitch", "wrist_roll",
]
QUADRUPED_ROLES = ["hip_roll", "hip_pitch", "knee"]

# role → family; the family decides housing vs limb link.
FAMILY = {
    "torso_yaw": "torso", "torso_roll": "torso", "torso_pitch": "torso",
    "hip_pitch": "hip", "hip_roll": "hip", "hip_yaw": "hip",
    "knee": "knee",
    "ankle_pitch": "ankle", "ankle_roll": "ankle",
    "shoulder_pitch": "shoulder", "shoulder_roll": "shoulder",
    "shoulder_yaw": "shoulder",
    "elbow": "elbow",
    "wrist_yaw": "wrist", "wrist_pitch": "wrist", "wrist_roll": "wrist",
    "neck": "neck", "digit": "digit", "aux": "aux",
}

# family → (group for the deepest member, group for the members above it)
FAMILY_GROUPS = {
    "hip": ("thigh", "hip_link"),
    "knee": ("shank", "shank"),
    "ankle": ("foot", "ankle_link"),
    "shoulder": ("upper_arm", "shoulder_link"),
    "elbow": ("forearm", "forearm"),
    "wrist": ("hand", "wrist_link"),
    "torso": ("trunk", "trunk"),
    "neck": ("trunk", "trunk"),
    "digit": ("hand", "hand"),
    "aux": ("aux", "aux"),
}

GROUP_ORDER_HUMANOID = ["trunk", "hip_link", "thigh", "shank", "ankle_link",
                        "foot", "shoulder_link", "upper_arm", "forearm",
                        "wrist_link", "hand", "aux"]
GROUP_ORDER_QUADRUPED = ["trunk", "hip_link", "thigh", "shank", "aux"]

#: Groups that exist once per limb rather than once per robot.
PER_LIMB_GROUPS = {g for g in GROUP_ORDER_HUMANOID if g != "trunk"}


# ── backend-neutral skeleton ──────────────────────────────────────────────────

@dataclass
class _Seg:
    """One moving body and its driving joint, world frame at the neutral pose.

    `pos` is the joint anchor, `com` the mass centre, `principal` the inertia
    eigenvalues about the COM.
    """
    name: str
    parent: int
    pos: np.ndarray
    axis: np.ndarray
    effort: float
    velocity: float
    lower: float
    upper: float
    mass: float
    com: np.ndarray
    principal: np.ndarray
    actuated: bool
    #: full 3x3 inertia about `com` in world axes, for parallel-axis sums.
    inertia: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))


@dataclass
class _Skeleton:
    segs: list                     # index 0 is the floating/fixed base
    tips: list                     # (name, world position) of fixed frames & sites
    n_dof: int

    @property
    def total_mass(self) -> float:
        return float(sum(s.mass for s in self.segs))


def _principal(inertia_3x3: np.ndarray) -> np.ndarray:
    w = np.linalg.eigvalsh(np.asarray(inertia_3x3, float))
    return np.clip(w, 0.0, None)


def _load_urdf(path: Path) -> _Skeleton:
    import pinocchio as pin
    model = pin.buildModelFromUrdf(str(path), pin.JointModelFreeFlyer())
    data = model.createData()
    q = pin.neutral(model)
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)

    local = {"JointModelRX": [1, 0, 0], "JointModelRY": [0, 1, 0],
             "JointModelRZ": [0, 0, 1]}
    segs = [_Seg("base", -1, np.zeros(3), np.zeros(3), 0.0, 0.0, 0.0, 0.0,
                 0.0, np.zeros(3), np.zeros(3), False)]
    index = {0: 0}
    n_dof = 0
    for j in range(1, model.njoints):
        sn = model.joints[j].shortname()
        actuated = sn.startswith(("JointModelR", "JointModelP"))
        n_dof += int(actuated)
        oMi = data.oMi[j]
        R = np.asarray(oMi.rotation)
        I = model.inertias[j]
        axis = R @ np.array(local.get(sn, [0, 0, 1]), float) if actuated else np.zeros(3)
        iv, iq = model.idx_vs[j], model.idx_qs[j]
        index[j] = len(segs)
        segs.append(_Seg(
            name=model.names[j], parent=index[model.parents[j]],
            pos=np.asarray(oMi.translation).copy(), axis=axis,
            effort=float(model.effortLimit[iv]) if actuated else 0.0,
            velocity=float(model.velocityLimit[iv]) if actuated else 0.0,
            lower=float(model.lowerPositionLimit[iq]) if actuated else 0.0,
            upper=float(model.upperPositionLimit[iq]) if actuated else 0.0,
            mass=float(I.mass),
            com=np.asarray(oMi.act(pin.SE3(np.eye(3), I.lever)).translation).copy(),
            principal=_principal(I.inertia),
            actuated=actuated,
            inertia=R @ np.asarray(I.inertia, float) @ R.T,
        ))
    # by index: a URDF joint and its child link may share a name
    tips = [(fr.name, np.asarray(data.oMf[fi].translation).copy())
            for fi, fr in enumerate(model.frames)]
    return _Skeleton(segs, tips, n_dof)


def model_assets(path: Path) -> dict:
    """Every asset a model references, read fresh from disk (see `mjcf_assets.read_assets`)."""
    return read_assets(path)


def spec_from_file(path: Path):
    """`MjSpec.from_file`, but reading the assets rather than trusting the cache."""
    return load_spec(path)


def compile_mjcf(path: Path):
    """Compile an MJCF with freshly read assets, bypassing MuJoCo's asset cache.

    MuJoCo caches meshes by file name per process; the twin builder rewrites
    `torso_mesh.stl` on each fixed-point pass, so the cache would go stale.
    """
    return load_model(path)




def _load_mjcf(path: Path) -> _Skeleton:
    """Read an MJCF through MuJoCo, merging fixed sub-bodies into their driver.

    Matches Pinocchio's URDF behaviour: parallel-axis sum about the merged COM.
    """
    import mujoco
    model = compile_mjcf(path)
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    mujoco.mj_forward(model, data)

    def body_inertia_world(b: int) -> tuple[float, np.ndarray, np.ndarray]:
        """(mass, world COM, 3x3 inertia about the COM in world axes)."""
        m = float(model.body_mass[b])
        com = np.asarray(data.xipos[b], float).copy()
        R = np.asarray(data.ximat[b], float).reshape(3, 3)
        Ic = R @ np.diag(np.asarray(model.body_inertia[b], float)) @ R.T
        return m, com, Ic

    # driver[b] = index of the nearest ancestor body (inclusive) carrying a joint
    driver = {0: 0}
    for b in range(1, model.nbody):
        driver[b] = b if model.body_jntnum[b] > 0 else driver[int(model.body_parentid[b])]

    # merge every body into its driver
    merged: dict[int, list[int]] = {}
    for b in range(model.nbody):
        merged.setdefault(driver[b], []).append(b)

    order = sorted(merged)
    index = {b: i for i, b in enumerate(order)}
    segs, n_dof = [], 0
    for b in order:
        mass, com, Ic = 0.0, np.zeros(3), np.zeros((3, 3))
        for c in merged[b]:
            mc, cc, Icc = body_inertia_world(c)
            if mc <= 0:
                continue
            mass, com = mass + mc, com + mc * cc
        com = com / mass if mass > 0 else np.asarray(data.xpos[b], float).copy()
        for c in merged[b]:
            mc, cc, Icc = body_inertia_world(c)
            if mc <= 0:
                continue
            d = cc - com
            Ic = Ic + Icc + mc * (float(d @ d) * np.eye(3) - np.outer(d, d))

        jadr = int(model.body_jntadr[b])
        njnt = int(model.body_jntnum[b])
        actuated = False
        pos = np.asarray(data.xpos[b], float).copy()
        axis = np.zeros(3)
        eff = vel = lo = hi = 0.0
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or f"body{b}"
        if njnt > 0:
            jt = int(model.jnt_type[jadr])
            pos = np.asarray(data.xanchor[jadr], float).copy()
            if jt in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
                actuated = True
                n_dof += 1
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jadr) or name
                axis = np.asarray(data.xaxis[jadr], float).copy()
                lo, hi = (float(x) for x in model.jnt_range[jadr])
                # peak torque: actuator forcerange, else the joint's own limit
                eff = _mjcf_effort(model, jadr)
                vel = _mjcf_velocity(model, name)
        # resolve through the driver: the parent body may have been merged away
        parent = -1 if b == 0 else index[driver[int(model.body_parentid[b])]]
        segs.append(_Seg(
            name=name, parent=parent,
            pos=pos, axis=axis, effort=eff, velocity=vel, lower=lo, upper=hi,
            mass=mass, com=com, principal=_principal(Ic - mass * 0.0),
            actuated=actuated, inertia=Ic,
        ))
    # the base's parent must not point at itself
    if segs:
        segs[0].parent = -1

    tips = [(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, s) or f"site{s}",
             np.asarray(data.site_xpos[s], float).copy()) for s in range(model.nsite)]
    return _Skeleton(segs, tips, n_dof)


def _mjcf_effort(model, jadr: int) -> float:
    import mujoco
    for a in range(model.nu):
        if (model.actuator_trntype[a] == int(mujoco.mjtTrn.mjTRN_JOINT)
                and model.actuator_trnid[a][0] == jadr):
            lo, hi = model.actuator_forcerange[a]
            if hi > lo:
                return float(max(abs(lo), abs(hi)))
    lo, hi = model.jnt_actfrcrange[jadr]
    return float(max(abs(lo), abs(hi))) if hi > lo else 0.0


def _mjcf_velocity(model, joint_name: str) -> float:
    """Speed limit from the `<joint>_velocity_limit` numeric the generator emits."""
    import mujoco
    for n in range(model.nnumeric):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_NUMERIC, n)
        if name == f"{joint_name}_velocity_limit":
            adr = int(model.numeric_adr[n])
            return float(model.numeric_data[adr])
    return 0.0


# ── measurement schema ────────────────────────────────────────────────────────

@dataclass
class JointRow:
    name: str
    role: str
    side: str | None
    pos: np.ndarray
    effort: float
    velocity: float
    group: str
    body_mass: float
    #: World-frame rotation axis at the neutral pose (used by the slenderness fit).
    axis: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass
class GroupInertia:
    """Composed rigid-body inertia of one anatomical group on one (left/front-left) limb.

    `principal`   principal moments about the group COM, kg·m² (frame-free).
    `k_gyr`       radius of gyration sqrt(tr(I) / 2m), m.
    `com_offset`  distance from the driving joint to the group COM, m (None if
                  the group has no mapped joint).
    """
    mass: float
    principal: np.ndarray
    k_gyr: float
    com_offset: float | None
    n_bodies: int

    def as_dict(self) -> dict:
        return {"mass_kg": float(self.mass),
                "principal_kgm2": [float(x) for x in self.principal],
                "k_gyr_m": float(self.k_gyr),
                "com_offset_m": (None if self.com_offset is None
                                 else float(self.com_offset)),
                "n_bodies": int(self.n_bodies)}


@dataclass
class Measurement:
    label: str
    category: str
    source: Path
    total_mass: float
    n_dof: int
    joints: list = field(default_factory=list)
    group_mass: dict = field(default_factory=dict)
    group_mass_per_limb: dict = field(default_factory=dict)
    group_box: dict = field(default_factory=dict)
    #: group → GroupInertia, composed over ONE limb (see `_fill_group_inertia`)
    group_inertia: dict = field(default_factory=dict)
    features: dict = field(default_factory=dict)
    role_pos: dict = field(default_factory=dict)
    chain: dict = field(default_factory=dict)

    @property
    def roles(self) -> set:
        return {j.role for j in self.joints}

    def torque(self, role: str):
        vals = [j.effort for j in self.joints if j.role == role and j.effort > 0]
        return float(np.mean(vals)) if vals else None

    def speed(self, role: str):
        vals = [j.velocity for j in self.joints if j.role == role and j.velocity > 0]
        return float(np.mean(vals)) if vals else None

    def installed_power_W(self) -> float:
        """Σ τ·ω/4 over the actuated joints — the catalogue's peak-power convention."""
        return float(sum(j.effort * j.velocity / 4.0 for j in self.joints
                         if j.effort > 0 and j.velocity > 0))

    def as_dict(self) -> dict:
        num = lambda v: ([float(x) for x in v] if isinstance(v, (list, tuple))  # noqa: E731
                         else float(v))
        return {
            "label": self.label, "category": self.category, "source": str(self.source),
            "total_mass_kg": self.total_mass, "n_dof": self.n_dof,
            "installed_power_W": self.installed_power_W(),
            "group_mass_kg": {k: float(v) for k, v in self.group_mass.items()},
            "group_mass_per_limb_kg": {k: float(v)
                                       for k, v in self.group_mass_per_limb.items()},
            "group_box_m": {k: [float(x) for x in v] for k, v in self.group_box.items()},
            "group_inertia": {k: v.as_dict() for k, v in self.group_inertia.items()},
            "features_m": {k: num(v) for k, v in self.features.items()},
            "chain": self.chain,
            "joints": [{"name": j.name, "role": j.role, "side": j.side,
                        "pos": [float(x) for x in j.pos],
                        "effort_Nm": float(j.effort),
                        "velocity_rad_s": float(j.velocity),
                        "group": j.group, "body_mass_kg": float(j.body_mass),
                        "axis": [float(x) for x in j.axis]}
                       for j in self.joints],
        }


# ── side detection ────────────────────────────────────────────────────────────

_QUAD_SIDE = [(r"(^|_)FL(_|$)|(^|_)LF(_|$)", "FL"), (r"(^|_)FR(_|$)|(^|_)RF(_|$)", "FR"),
              (r"(^|_)RL(_|$)|(^|_)LH(_|$)", "RL"), (r"(^|_)RR(_|$)|(^|_)RH(_|$)", "RR"),
              (r"^fl_", "FL"), (r"^fr_", "FR"), (r"^rl_", "RL"), (r"^rr_", "RR")]
#: `left`/`right` are matched case-insensitively.
_HUM_SIDE = [(r"(?i:left)|(^|_)l_|_l$|^L(?=[A-Z])", "L"),
             (r"(?i:right)|(^|_)r_|_r$|^R(?=[A-Z])", "R")]


def detect_side(name: str, category: str):
    for pat, side in (_QUAD_SIDE if category == "quadruped" else _HUM_SIDE):
        if re.search(pat, name):
            return side
    return None


def inertia_box(mass: float, principal: np.ndarray) -> np.ndarray:
    """Side lengths of the uniform solid box with this mass and these principal inertias.

    The same size proxy `datasets/robot_descriptions/` uses. Returns zeros for a
    tensor no solid body can have.
    """
    if mass <= 0:
        return np.zeros(3)
    w = np.asarray(principal, float)
    sq = np.array([6.0 / mass * (w.sum() - 2.0 * wi) for wi in w])
    if np.any(sq < 0):
        return np.zeros(3)
    return np.sort(np.sqrt(sq))[::-1]


class RobotMeasurer:
    """Measures a robot description against a joint-name → canonical-role map."""

    def __init__(self, role_map, category: str):
        self.role_map = [(re.compile(p), r) for p, r in role_map]
        self.category = category

    def role_of(self, name: str):
        for pat, role in self.role_map:
            if pat.search(name):
                return role
        return None

    def measure(self, path: Path, label: str) -> Measurement:
        path = Path(path)
        sk = (_load_urdf(path) if path.suffix.lower() == ".urdf"
              else _load_mjcf(path))
        segs = sk.segs

        roles = {i: r for i, s in enumerate(segs) if s.actuated
                 for r in [self.role_of(s.name)] if r is not None}

        # Housing if a same-family joint sits below; the deepest drives the limb.
        def ancestors(i):
            while segs[i].parent >= 0:
                i = segs[i].parent
                yield i

        has_child_of_family = {i: False for i in roles}
        for i in roles:
            for a in ancestors(i):
                if a in roles:
                    if FAMILY[roles[a]] == FAMILY[roles[i]]:
                        has_child_of_family[a] = True
                    break

        rows, group_of = [], {}
        for i, s in enumerate(segs):
            if i in roles:
                limb, housing = FAMILY_GROUPS[FAMILY[roles[i]]]
                group_of[i] = housing if has_child_of_family[i] else limb
            elif i == 0:
                group_of[i] = "trunk"
            else:
                # Unmapped joints inherit the parent's group so masses sum to total.
                group_of[i] = group_of.get(s.parent, "aux")
            if i in roles:
                rows.append(JointRow(
                    name=s.name, role=roles[i],
                    side=detect_side(s.name, self.category), pos=s.pos,
                    effort=s.effort, velocity=s.velocity,
                    group=group_of[i], body_mass=s.mass,
                    axis=np.asarray(s.axis, float)))

        m = Measurement(label=label, category=self.category, source=path,
                        total_mass=sk.total_mass, n_dof=sk.n_dof,
                        joints=sorted(rows, key=lambda r: (r.role, r.side or "")))

        masses, biggest = {}, {}
        for i, s in enumerate(segs):
            g = group_of[i]
            masses[g] = masses.get(g, 0.0) + s.mass
            if s.mass > biggest.get(g, (0.0, None))[0]:
                biggest[g] = (s.mass, s)
        m.group_mass = masses
        n_limbs = 4 if self.category == "quadruped" else 2
        m.group_mass_per_limb = {g: (v / n_limbs if g in PER_LIMB_GROUPS else v)
                                 for g, v in masses.items()}
        m.group_box = {g: inertia_box(s.mass, s.principal) for g, (_, s) in biggest.items()}
        self._fill_group_inertia(segs, roles, group_of, m)

        self._fill_chain(segs, roles, m)
        self._fill_features(m)
        self._fill_tip_features(segs, roles, sk.tips, m)
        return m

    # ── composed inertia per anatomical group ────────────────────────────────
    def _fill_group_inertia(self, segs, roles, group_of, m: Measurement) -> None:
        """Compose each group's bodies into one rigid body, on ONE limb.

        Side is inherited from the driving joint, as `group_of` is, so this
        matches `group_mass_per_limb`.
        """
        pref = "FL" if self.category == "quadruped" else "L"
        # side per segment, inherited down the tree like `group_of` is
        side_of: dict = {}
        for i, s in enumerate(segs):
            own = detect_side(s.name, self.category) if i in roles else None
            side_of[i] = own if own is not None else side_of.get(s.parent)

        buckets: dict = {}
        for i, s in enumerate(segs):
            g = group_of[i]
            if g in PER_LIMB_GROUPS and side_of.get(i) != pref:
                continue                        # one limb only
            if s.mass <= 0:
                continue
            buckets.setdefault(g, []).append(i)

        for g, idx in buckets.items():
            mass = sum(segs[i].mass for i in idx)
            com = sum(segs[i].mass * segs[i].com for i in idx) / mass
            I = np.zeros((3, 3))
            for i in idx:
                d = segs[i].com - com
                I += segs[i].inertia + segs[i].mass * (float(d @ d) * np.eye(3)
                                                       - np.outer(d, d))
            # tr(I) = 2*Sigma m r^2, so this is the RMS distance of mass from the COM
            k = math.sqrt(max(float(np.trace(I)), 0.0) / (2.0 * mass))
            # Driving joint = shallowest mapped joint (loaders list parents first).
            # No mapped joint (e.g. a quadruped trunk) gives None, not 0.
            anchors = [segs[i].pos for i in idx if i in roles]
            offset = float(np.linalg.norm(com - anchors[0])) if anchors else None
            m.group_inertia[g] = GroupInertia(
                mass=mass, principal=_principal(I), k_gyr=k,
                com_offset=offset, n_bodies=len(idx))

    # ── chain order ──────────────────────────────────────────────────────────
    def _fill_chain(self, segs, roles, m: Measurement) -> None:
        pref = "FL" if self.category == "quadruped" else "L"
        by_family: dict = {}
        for i, role in roles.items():
            side = detect_side(segs[i].name, self.category)
            if side not in (None, pref):
                continue
            depth, k = 0, i
            while segs[k].parent >= 0:
                k = segs[k].parent
                depth += 1
            by_family.setdefault(FAMILY[role], []).append((depth, role))
        m.chain = {fam: [r for _, r in sorted(v)] for fam, v in by_family.items()}

    # ── scalar lengths ───────────────────────────────────────────────────────
    def _fill_features(self, m: Measurement) -> None:
        pref = "FL" if self.category == "quadruped" else "L"
        pos: dict = {}
        for j in m.joints:
            if j.role not in pos or j.side == pref:
                pos.setdefault(j.role, j.pos)
                if j.side == pref:
                    pos[j.role] = j.pos
        m.role_pos = pos

        def d(a, b):
            return (float(np.linalg.norm(pos[a] - pos[b]))
                    if a in pos and b in pos else None)

        axes: dict = {}
        for j in m.joints:
            if np.linalg.norm(j.axis) > 0 and (j.role not in axes or j.side == pref):
                axes[j.role] = np.asarray(j.axis, float) / np.linalg.norm(j.axis)

        def axis_gap(a, b):
            """`d`, but between the two joint AXES where they are parallel."""
            if a not in pos or b not in pos:
                return None
            v = np.asarray(pos[b] - pos[a], float)
            ua, ub = axes.get(a), axes.get(b)
            if ua is not None and ub is not None and abs(float(ua @ ub)) > 0.999:
                v = v - ua * float(v @ ua)
            return float(np.linalg.norm(v))

        f: dict = {}
        if self.category == "humanoid":
            hip, ank = m.chain.get("hip", []), m.chain.get("ankle", [])
            sho, wri = m.chain.get("shoulder", []), m.chain.get("wrist", [])
            tor = m.chain.get("torso", [])
            if len(hip) >= 2:
                f["hip_seg_0"] = d(hip[0], hip[1])
            if len(hip) >= 3:
                f["hip_seg_1"] = d(hip[1], hip[2])
            if hip:
                f["thigh"] = d(hip[-1], "knee")
                # Hip roll to knee: the full span, so it cannot drift when the
                # packing gate resizes the hip links.
                f["thigh_total"] = d("hip_roll", "knee") or d(hip[-1], "knee")
                f["hip_z"], f["hip_y"] = float(pos[hip[0]][2]), float(abs(pos[hip[0]][1]))
            if len(hip) >= 2:
                # Roll-axis drop below hip pitch. Recorded only: the tree lays
                # `hip_seg_0` out laterally, so this is a known height gap.
                f["hip_drop"] = float(pos[hip[0]][2] - pos[hip[1]][2])
            if ank:
                f["shank"] = d("knee", ank[0])
                # Knee to the last ankle axis: the full span must agree, not its split.
                f["shank_total"] = d("knee", ank[-1])
                f["ankle_seg"] = d(ank[0], ank[-1]) or 0.0
                f["ankle_z"] = float(pos[ank[-1]][2])
            if len(sho) >= 2:
                f["shoulder_seg_0"] = d(sho[0], sho[1])
            if len(sho) >= 3:
                f["shoulder_seg_1"] = d(sho[1], sho[2])
            if sho:
                f["shoulder_z"] = float(pos[sho[0]][2])
                f["shoulder_y"] = float(abs(pos[sho[0]][1]))
                f["bicep"] = d(sho[-1], "elbow")
            # Lateral half-span of the roll axes (`*_seg_0` is 3D and includes z).
            if "shoulder_roll" in pos:
                f["shoulder_roll_y"] = float(abs(pos["shoulder_roll"][1]))
            if "hip_roll" in pos:
                f["hip_roll_y"] = float(abs(pos["hip_roll"][1]))
            if wri:
                f["forearm"] = d("elbow", wri[0])
                if len(wri) >= 2:
                    f["wrist_seg_0"] = d(wri[0], wri[1])
                if len(wri) >= 3:
                    f["wrist_seg_1"] = d(wri[1], wri[2])
            if tor:
                f["waist_z"] = float(pos[tor[0]][2])
                f["waist_span"] = d(tor[0], tor[-1]) or 0.0
            box = m.group_box.get("foot")
            if box is not None and np.any(box > 0):
                f["foot_length"], f["foot_width"], _ = (float(x) for x in box)
        else:
            # Between the parallel hip-pitch and knee AXES, not their anchors.
            f["thigh"] = axis_gap("hip_pitch", "knee")
            if "hip_roll" in pos:
                f["abad_x"] = float(abs(pos["hip_roll"][0]))
                f["abad_y"] = float(abs(pos["hip_roll"][1]))
            if "hip_roll" in pos and "hip_pitch" in pos:
                f["hip_dx"] = float(abs(pos["hip_pitch"][0] - pos["hip_roll"][0]))
                f["hip_dy"] = float(abs(pos["hip_pitch"][1] - pos["hip_roll"][1]))
            if "knee" in pos:
                f["knee_z"] = float(pos["knee"][2])
                # Lateral leg-plane position (the hip-pitch anchor's y is arbitrary).
                f["leg_plane_y"] = float(abs(pos["knee"][1]))
            box = m.group_box.get("trunk")
            if box is not None and np.any(box > 0):
                f["trunk_box"] = [float(x) for x in box]
        m.features = {k: v for k, v in f.items() if v is not None}

    # ── terminal segments ────────────────────────────────────────────────────
    def _fill_tip_features(self, segs, roles, tips, m: Measurement) -> None:
        """Lengths of the segments with no child joint to measure to.

        Uses a fixed contact frame if one exists, else `2·|com − joint|` (a
        uniform rod). The inertia box overestimates these and is not used.
        """
        pref = "FL" if self.category == "quadruped" else "L"
        seg_of_role = {}
        for i, r in roles.items():
            side = detect_side(segs[i].name, self.category)
            if side in (None, pref) and r not in seg_of_role:
                seg_of_role[r] = segs[i]

        def frame_tip(pattern, role):
            s = seg_of_role.get(role)
            if s is None:
                return None
            best = None
            for name, p in tips:
                if not re.search(pattern, name, re.I):
                    continue
                if detect_side(name, self.category) not in (None, pref):
                    continue
                dist = float(np.linalg.norm(p - s.pos))
                best = dist if best is None or dist > best else best
            return best

        def rod_len(role, axis=None):
            s = seg_of_role.get(role)
            if s is None or s.mass <= 0:
                return None
            v = s.com - s.pos
            return float(2.0 * (abs(v[axis]) if axis is not None else np.linalg.norm(v)))

        if self.category == "quadruped":
            shank = frame_tip(r"foot|toe|contact", "knee") or rod_len("knee")
            if shank:
                m.features["shank"] = shank
        else:
            ank, wri = m.chain.get("ankle", []), m.chain.get("wrist", [])
            if ank:
                h = rod_len(ank[-1], axis=2)
                if h:
                    m.features["foot_h"] = h
            if wri:
                h = rod_len(wri[-1])
                if h:
                    m.features["hand_length"] = h
            if "forearm" not in m.features:
                d = frame_tip(r"hand|(^|_)ee($|_)|end_effector", "elbow") or rod_len("elbow")
                if d:
                    m.features["forearm"] = d


def _primitive_half_extents(model, g: int) -> np.ndarray:
    """Local half-extents of a non-mesh geom, per MuJoCo's `size` convention."""
    import mujoco
    s = np.asarray(model.geom_size[g], float)
    t = int(model.geom_type[g])
    if t == int(mujoco.mjtGeom.mjGEOM_BOX):
        return s[:3].copy()
    if t == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        return np.full(3, s[0])
    if t == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        return np.array([s[0], s[0], s[1]])
    if t == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        return np.array([s[0], s[0], s[1] + s[0]])
    if t == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
        return s[:3].copy()
    return np.full(3, float(np.max(s)))


def _mjcf_group_geoms(path: Path, role_map, category: str):
    """Yield `(group, world AABB, joint anchor)` for each anatomical group.

    Bodies are attributed as in `_load_mjcf`; the geometric extent is measured,
    not the inertia.
    """
    import mujoco
    model = compile_mjcf(path)
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    mujoco.mj_forward(model, data)

    compiled = [(re.compile(p), r) for p, r in role_map]

    def role_of(name):
        return next((r for pat, r in compiled if pat.search(name)), None)

    driver = {0: 0}
    for b in range(1, model.nbody):
        driver[b] = b if model.body_jntnum[b] > 0 else driver[int(model.body_parentid[b])]

    # role and chain depth per driving body, to find each family's terminal joint
    role_of_body, anchor = {}, {}
    for b in range(1, model.nbody):
        if model.body_jntnum[b] <= 0:
            continue
        jadr = int(model.body_jntadr[b])
        if model.jnt_type[jadr] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jadr) or ""
        r = role_of(name)
        if r is None:
            continue
        role_of_body[b] = (r, name)
        anchor[b] = np.asarray(data.xanchor[jadr], float).copy()

    has_child_of_family = {b: False for b in role_of_body}
    for b in role_of_body:
        p = int(model.body_parentid[b])
        while p > 0:
            if p in role_of_body:
                if FAMILY[role_of_body[p][0]] == FAMILY[role_of_body[b][0]]:
                    has_child_of_family[p] = True
                break
            p = int(model.body_parentid[p])

    boxes: dict = {}
    for b in range(model.nbody):
        d = driver[b]
        if d not in role_of_body:
            continue
        fam = FAMILY[role_of_body[d][0]]
        limb, housing = FAMILY_GROUPS[fam]
        group = housing if has_child_of_family[d] else limb
        side = detect_side(role_of_body[d][1], category)
        pref = "FL" if category == "quadruped" else "L"
        if side not in (None, pref):
            continue
        for g in range(model.ngeom):
            if model.geom_bodyid[g] != b:
                continue
            pos = data.geom_xpos[g]
            if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mid = model.geom_dataid[g]
                v0, nv = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
                v = model.mesh_vert[v0:v0 + nv].reshape(-1, 3)
                w = v @ np.asarray(data.geom_xmat[g]).reshape(3, 3).T + pos
                lo, hi = w.min(axis=0), w.max(axis=0)
            else:
                half = _primitive_half_extents(model, g)
                R = np.abs(np.asarray(data.geom_xmat[g]).reshape(3, 3))
                ext = R @ half
                lo, hi = np.asarray(pos) - ext, np.asarray(pos) + ext
            # Collidable geoms get their own bucket: visual shells carry trim.
            solid = bool(model.geom_contype[g] or model.geom_conaffinity[g])
            for key in (("solid", group) if solid else ("any", group),
                        ("any", group)):
                prev = boxes.get(key)
                if prev is None:
                    boxes[key] = [lo.copy(), hi.copy(), anchor[d]]
                else:
                    prev[0] = np.minimum(prev[0], lo)
                    prev[1] = np.maximum(prev[1], hi)
            # Also keep the single largest-footprint geom, free of attached parts.
            area = float((hi[0] - lo[0]) * (hi[1] - lo[1]))
            best = boxes.get(("dominant", group))
            if best is None or area > best[3]:
                boxes[("dominant", group)] = [lo.copy(), hi.copy(),
                                              anchor[d], area]

    out: dict = {}
    for (bucket, group), val in boxes.items():
        out.setdefault(group, {})[bucket] = tuple(val[:3])
    return out


# ── The trunk's drawn cross-section ───────────────────────────────────────────
# A trunk is a shell, so its inertia box has the wrong aspect; the section and
# chamfer are read off the vendor's hull instead.


def _hull_2d(pts: np.ndarray) -> np.ndarray:
    """Convex hull of `(n, 2)` points, counter-clockwise (monotone chain)."""
    p = np.unique(np.round(np.asarray(pts, float), 9), axis=0)
    if len(p) < 3:
        return p
    p = p[np.lexsort((p[:, 1], p[:, 0]))]

    def chain(seq):
        out: list = []
        for q in seq:
            while len(out) >= 2 and np.cross(out[-1] - out[-2], q - out[-2]) <= 0:
                out.pop()
            out.append(q)
        return out[:-1]

    return np.asarray(chain(p) + chain(p[::-1]))


def _halfplanes(hull: np.ndarray):
    """`(A, b)` with `A x <= b` for a CCW hull, or None if it excludes the origin.

    The origin must be strictly inside (`b_i > 0`) for `_max_chamfered_rect`.
    """
    if len(hull) < 3:
        return None
    e = np.roll(hull, -1, axis=0) - hull
    A = np.column_stack([e[:, 1], -e[:, 0]])          # outward normal of a CCW edge
    n = np.linalg.norm(A, axis=1)
    keep = n > 1e-12
    if keep.sum() < 3:
        return None
    A, hull = A[keep] / n[keep, None], hull[keep]
    b = np.einsum("ij,ij->i", A, hull)
    return (A, b) if np.all(b > 1e-6) else None


def _max_chamfered_rect(A: np.ndarray, b: np.ndarray) -> tuple:
    """Largest origin-centred `rect_outline` inside `{x : A x <= b}`.

    Scale is closed-form at fixed (aspect, chamfer); those two are found by a
    grid then pattern search. Returns `(lu, lv, chamfer_fraction, area)`.
    """
    from draft.generation.mounting import rect_outline

    def measure(r: float, c: float):
        # `rect_outline` rejects degenerate chamfers, so c stays in (C_LO, C_HI).
        V = np.asarray(rect_outline(1.0, r, c), float)
        s = float(np.min(b / np.maximum((A @ V.T).max(axis=1), 1e-12)))
        return s, s * s * (r - 2.0 * min(1.0, r) ** 2 * c * c)

    C_LO, C_HI = 0.01, 0.49
    best = (-1.0, 1.0, 0.2)
    for r in np.geomspace(0.15, 6.0, 60):
        for c in np.linspace(C_LO, C_HI, 25):
            _, a = measure(r, c)
            if a > best[0]:
                best = (a, r, c)
    _, r, c = best
    for step_r, step_c in ((0.05, 0.04), (0.01, 0.008), (0.002, 0.0016),
                           (4e-4, 3e-4)):
        moved = True
        while moved:
            moved = False
            for dr, dc in ((step_r, 0), (-step_r, 0), (0, step_c), (0, -step_c),
                           (step_r, step_c), (-step_r, -step_c),
                           (step_r, -step_c), (-step_r, step_c)):
                rr, cc = r * (1 + dr), float(np.clip(c + dc, C_LO, C_HI))
                _, a = measure(rr, cc)
                if a > best[0] + 1e-12:
                    best, r, c, moved = (a, rr, cc), rr, cc, True
    a, r, c = best
    s, _ = measure(r, c)
    return s, s * r, c, a


def trunk_section_fit(path: Path, role_map, category: str,
                      n_stations: int = 25) -> dict:
    """The largest constant cross-section the vendor's trunk shell will hold.

    Fits the largest chamfered `torso_ly` x `torso_lz` rectangle inside the
    shell at stations over the hip-roll span and reports the median (the ends
    taper). Sections use plane-triangle intersection since the shells are open
    meshes. Quadrupeds only.
    """
    import mujoco
    import trimesh

    if category != "quadruped":
        return {}
    try:
        model = compile_mjcf(Path(path))
    except Exception:                                   # noqa: BLE001
        return {}
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    mujoco.mj_forward(model, data)

    # The floating base — the body the twin's own trunk mesh corresponds to.
    root = next((b for b in range(1, model.nbody)
                 if model.body_jntnum[b] > 0
                 and model.jnt_type[model.body_jntadr[b]] == mujoco.mjtJoint.mjJNT_FREE),
                1)
    R = np.asarray(data.xmat[root], float).reshape(3, 3)
    p0 = np.asarray(data.xpos[root], float)

    # The trunk's largest mesh by bounding-box volume: the body shell.
    shell, best = None, -1.0
    for g in range(model.ngeom):
        if model.geom_bodyid[g] != root or model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = model.geom_dataid[g]
        v0, nv = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        f0, nf = model.mesh_faceadr[mid], model.mesh_facenum[mid]
        V = model.mesh_vert[v0:v0 + nv].reshape(-1, 3)
        F = model.mesh_face[f0:f0 + nf].reshape(-1, 3)
        Rg = np.asarray(data.geom_xmat[g], float).reshape(3, 3)
        local = ((V @ Rg.T + data.geom_xpos[g]) - p0) @ R
        vol = float(np.prod(local.max(axis=0) - local.min(axis=0)))
        if vol > best:
            shell, best = trimesh.Trimesh(vertices=local, faces=F, process=False), vol
    if shell is None:
        return {}

    # Section over the hip-roll spacing: the body between the legs.
    compiled = [(re.compile(pat), r) for pat, r in role_map]
    hips = [abs(float(((np.asarray(data.xanchor[j], float) - p0) @ R)[0]))
            for j in range(model.njnt)
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
            and FAMILY.get(next((r for pat, r in compiled
                                 if pat.search(mujoco.mj_id2name(
                                     model, mujoco.mjtObj.mjOBJ_JOINT, j) or "")),
                                None)) == "hip"]
    half = max(hips) if hips else float(shell.extents[0]) / 2
    if half <= 0:
        return {}

    rows = []
    for x in np.linspace(-half, half, n_stations):
        segs = trimesh.intersections.mesh_plane(
            shell, plane_normal=np.array([1.0, 0.0, 0.0]),
            plane_origin=np.array([float(x), 0.0, 0.0]))
        if segs is None or len(segs) == 0:
            continue                                    # past the end of the shell
        hp = _halfplanes(_hull_2d(np.asarray(segs).reshape(-1, 3)[:, 1:]))
        if hp is None:
            continue
        rows.append(_max_chamfered_rect(*hp))
    if len(rows) < 3:
        return {}
    ly, lz, ch, area = np.median(np.asarray(rows, float), axis=0)
    return {"trunk_fit_ly": float(ly), "trunk_fit_lz": float(lz),
            "trunk_fit_chamfer": float(ch), "trunk_fit_area": float(area),
            "trunk_fit_stations": float(len(rows))}


def mjcf_shape_features(path: Path, role_map, category: str) -> dict:
    """Foot and hand dimensions read off the geometry, not off the inertia.

    Inertia boxes under-report thin or shell-like parts, so the collision and
    visual geometry is measured directly.
    """
    try:
        boxes = _mjcf_group_geoms(path, role_map, category)
    except Exception:                                   # noqa: BLE001
        return {}
    out: dict = {}
    # Geometric envelope of each group, read by
    # `synthesize.TwinSynthesizer.aspect_overrides`.
    for group, buckets in boxes.items():
        bb = buckets.get("dominant") or buckets.get("solid") or buckets.get("any")
        if not bb:
            continue
        # Per world axis, unsorted: the fit needs to know which way it points.
        for axis, v in zip("xyz", (bb[1] - bb[0])):
            out[f"fitbox_{group}_{axis}"] = float(v)
    if category != "humanoid":
        out.update(trunk_section_fit(path, role_map, category))
        return out
    # Foot prefers collidable geometry (the sole); hand prefers all geometry
    # (its collision proxy is often a token sphere).
    foot = boxes.get("foot", {})
    fb = foot.get("dominant") or foot.get("solid") or foot.get("any")
    if fb:
        lo, hi, anchor = fb
        ext = hi - lo
        out["foot_length"] = float(max(ext[0], ext[1]))
        out["foot_width"] = float(min(ext[0], ext[1]))
        out["foot_h"] = float(anchor[2] - lo[2])        # ankle joint down to sole
    hand = boxes.get("hand", {})
    hb = hand.get("any") or hand.get("solid")
    if hb:
        lo, hi, anchor = hb
        ext = hi - lo
        out["hand_length"] = float(np.linalg.norm(anchor - (lo + hi) / 2) * 2)
        # Smallest extent: the parametric hand is a sphere.
        out["hand_girth"] = float(np.min(ext))
    out.update(_hand_blob(path, role_map))
    out.update(_head_extent(path, role_map, category))
    return out


def _hand_blob(path: Path, role_map) -> dict:
    """The target's hand as the equal-volume sphere `render._blob_hands` draws.

    Returns its radius and reach from the last wrist axis, which `synthesize`
    uses to size and place the twin's hand.
    """
    import mujoco
    from .render import HAND_MESH_RE, _equal_volume_sphere
    try:
        model = compile_mjcf(Path(path))
    except Exception:                                   # noqa: BLE001
        return {}
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    mujoco.mj_forward(model, data)

    # The last wrist axis, by role: whatever the deepest `wrist_*` joint is.
    compiled = [(re.compile(p), r) for p, r in role_map]
    anchor, best = None, -1
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        role = next((r for pat, r in compiled if pat.search(name)), None)
        if FAMILY.get(role) != "wrist" or not name.startswith(("left", "L_")):
            continue
        depth = len(_body_chain(model, int(model.jnt_bodyid[j])))
        if depth > best:
            anchor, best = np.asarray(data.xanchor[j], float), depth
    if anchor is None:
        return {}

    for g in range(model.ngeom):
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = model.geom_dataid[g]
        mesh = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mid) or ""
        if not HAND_MESH_RE.search(mesh) or not mesh.startswith(("left", "L_")):
            continue
        v0, nv = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        f0, nf = model.mesh_faceadr[mid], model.mesh_facenum[mid]
        R = np.asarray(data.geom_xmat[g], float).reshape(3, 3)
        verts = model.mesh_vert[v0:v0 + nv].reshape(-1, 3) @ R.T + data.geom_xpos[g]
        faces = model.mesh_face[f0:f0 + nf].reshape(-1, 3)
        centre, radius = _equal_volume_sphere(verts, faces)
        # Seat the sphere where the hand starts, as `_blob_hands` does.
        ray = centre - anchor
        reach = float(np.linalg.norm(ray))
        if reach > 1e-9:
            u = ray / reach
            reach = float(((verts - anchor) @ u).min()) + radius
        return {"hand_blob_r": float(radius),
                "hand_blob_reach": float(max(reach, radius))}
    return {}


def _body_chain(model, bid: int) -> tuple:
    out = []
    while bid > 0:
        out.append(bid)
        bid = int(model.body_parentid[bid])
    return tuple(reversed(out))


#: Slabs in the above-shoulder width profile.
_HEAD_BINS = 24


def _head_extent(path: Path, role_map, category: str) -> dict:
    """Head size, read off whatever the shipped model puts above its shoulders.

    Anything entirely above the shoulder line counts as head and neck.
    """
    import mujoco
    if category != "humanoid":
        return {}
    try:
        model = mujoco.MjModel.from_xml_path(str(Path(path).resolve()))
    except Exception:                                   # noqa: BLE001
        return {}
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    mujoco.mj_forward(model, data)

    compiled = [(re.compile(p), r) for p, r in role_map]
    shoulder_z = shoulder_y = None
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        role = next((r for pat, r in compiled if pat.search(name)), None)
        if role and FAMILY.get(role) == "shoulder":
            z, y = float(data.xanchor[j][2]), abs(float(data.xanchor[j][1]))
            shoulder_z = z if shoulder_z is None else max(shoulder_z, z)
            shoulder_y = y if shoulder_y is None else max(shoulder_y, y)
    if shoulder_z is None:
        return {}

    # Points above the shoulder line (a box cannot separate head from neck).
    pts = []
    for g in range(model.ngeom):
        pos = data.geom_xpos[g]
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = model.geom_dataid[g]
            v0, nv = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
            v = model.mesh_vert[v0:v0 + nv].reshape(-1, 3)
            w = v @ np.asarray(data.geom_xmat[g]).reshape(3, 3).T + pos
        else:
            r = float(np.max(model.geom_size[g]))
            w = np.asarray(pos) + r * np.array(
                [[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)],
                dtype=float)
        if w[:, 2].min() < shoulder_z:   # not entirely above the shoulders
            continue
        pts.append(w)
    if not pts:
        return {}
    pts = np.vstack(pts)
    crown = float(pts[:, 2].max())
    if crown <= shoulder_z:
        return {}

    # Width profile up the neck; the head is the widest part.
    bins = np.linspace(shoulder_z, crown, _HEAD_BINS + 1)
    prof = np.zeros(_HEAD_BINS)
    for i, (a, b) in enumerate(zip(bins[:-1], bins[1:])):
        s = pts[(pts[:, 2] >= a) & (pts[:, 2] < b)]
        if len(s) >= 3:
            prof[i] = 0.25 * float((s[:, 0].max() - s[:, 0].min())
                                   + (s[:, 1].max() - s[:, 1].min()))
    # Peak over the upper span, so a wide collar is ignored.
    upper = range(int(0.4 * _HEAD_BINS), _HEAD_BINS)
    radius = float(max(prof[i] for i in upper))
    if radius <= 0:
        return {}
    # Cap at the shoulder half-width.
    if shoulder_y:
        radius = min(radius, 0.9 * shoulder_y)
    # `neck_length` places the head centre so the crown matches the target's.
    return {"head_radius": radius,
            "head_crown": crown - shoulder_z,
            "neck_length": max(crown - shoulder_z - radius, 0.01)}


def shipped_palette(path: Path, role_map, category: str) -> dict:
    """The shipped model's own colours, per anatomical group.

    Weighted by geom volume. Returns `{slot: (r, g, b)}` for `_trunk`, `_limb`
    and `_all`.
    """
    import mujoco
    try:
        model = mujoco.MjModel.from_xml_path(str(Path(path).resolve()))
        boxes = _mjcf_group_geoms(path, role_map, category)
    except Exception:                                   # noqa: BLE001
        return {}
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    mujoco.mj_forward(model, data)

    driver = {0: 0}
    for b in range(1, model.nbody):
        driver[b] = b if model.body_jntnum[b] > 0 else driver[int(model.body_parentid[b])]

    weight: dict = {}
    for g in range(model.ngeom):
        if model.geom_group[g] >= 3:                    # collision-only groups
            continue
        mat = model.geom_matid[g]
        rgba = (model.mat_rgba[mat] if mat >= 0 else model.geom_rgba[g])
        if float(rgba[3]) < 0.35:
            continue
        size = np.asarray(model.geom_size[g], float)
        vol = float(np.prod(np.clip(size, 1e-3, None)))
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = model.geom_dataid[g]
            v0, nv = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
            v = model.mesh_vert[v0:v0 + nv].reshape(-1, 3)
            vol = float(np.prod(np.clip(v.max(axis=0) - v.min(axis=0), 1e-3, None)))
        key = tuple(np.round(np.asarray(rgba[:3], float), 3))
        b = int(model.geom_bodyid[g])
        is_trunk = driver[b] == driver[1] if model.nbody > 1 else True
        slot = "_trunk" if is_trunk else "_limb"
        weight.setdefault(slot, {})[key] = weight.get(slot, {}).get(key, 0.0) + vol
        weight.setdefault("_all", {})[key] = weight.get("_all", {}).get(key, 0.0) + vol

    out = {}
    for slot, table in weight.items():
        if table:
            out[slot] = max(table.items(), key=lambda kv: kv[1])[0]
    return out


def mjcf_extent(path: Path) -> dict:
    """Vertical extent of the collision geometry — the standing height, in effect.

    Used for whole-robot plots, never as a generator input.
    """
    import mujoco
    try:
        model = mujoco.MjModel.from_xml_path(str(path))
    except Exception:
        return {}
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    mujoco.mj_forward(model, data)

    lo, hi = math.inf, -math.inf
    for g in range(model.ngeom):
        pos = data.geom_xpos[g]
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = model.geom_dataid[g]
            v0, nv = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
            verts = model.mesh_vert[v0:v0 + nv].reshape(-1, 3)
            world = verts @ np.asarray(data.geom_xmat[g]).reshape(3, 3).T + pos
            zlo, zhi = float(world[:, 2].min()), float(world[:, 2].max())
        else:
            r = float(np.max(model.geom_size[g]))
            zlo, zhi = float(pos[2] - r), float(pos[2] + r)
        lo, hi = min(lo, zlo), max(hi, zhi)
    return {"z_min": lo, "z_max": hi, "extent_z": hi - lo,
            "total_mass": float(np.sum(model.body_mass))}
