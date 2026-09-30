"""Rewrite a vendor quadruped MJCF into the RL env's naming contract.

The adapted model keeps the vendor's masses, inertias, geometry, limits and
meshes, so the same env (`make_quadruped_entity_cfg`) runs on twin and target.
It renames joints/bodies to the tree's names and adds contract items: velocity
limit numerics and actuator forcerange (from the URDF), foot sites and foot geom
names, and IMU sensors. Joint axis signs and zero poses are checked against the
twin and relabelled where they differ by convention; a remaining zero-pose error
above `_ZERO_POSE_TOL_DEG` raises. Everything is logged to `rl_adapter_report.yaml`.
"""

from __future__ import annotations

import math
import re
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
import yaml

from .measure import detect_side
from .targets import Target
from .zero_align import bake_joint_rotation, align_root_origin, normalize_root_frame

# ── the naming contract draft.tasks.quadruped expects ────────────────────────
_LEGS = ("fl", "fr", "rl", "rr")
_SIDE_TO_PREFIX = {"FL": "fl", "FR": "fr", "RL": "rl", "RR": "rr"}

#: canonical role -> the body that role's joint drives, in tree names
_QUAD_ROLE_BODY = {
    "hip_roll": "roll_link",
    "hip_pitch": "upper_leg_link",
    "knee": "lower_leg_link",
}
_QUAD_ROLES = ("hip_roll", "hip_pitch", "knee")

#: Max zero-pose disagreement (deg) after alignment; beyond it the pair is not comparable.
_ZERO_POSE_TOL_DEG = 0.75

#: Alignment is iterated because a shank's direction depends on the thigh's zero.
_ALIGN_PASSES = 4

_TORSO = "torso_link"


@dataclass
class AdaptReport:
    key: str
    source_mjcf: str = ""
    twin_mjcf: str = ""
    joint_renames: dict = field(default_factory=dict)
    body_renames: dict = field(default_factory=dict)
    flipped_joints: list = field(default_factory=list)
    zero_pose_error_deg: dict = field(default_factory=dict)
    lateral_offset_mm: dict = field(default_factory=dict)
    zero_offsets_deg: dict = field(default_factory=dict)
    actuator_internals: dict = field(default_factory=dict)
    effort_nm: dict = field(default_factory=dict)
    velocity_rad_s: dict = field(default_factory=dict)
    foot_radius_m: dict = field(default_factory=dict)
    total_mass_kg: float = 0.0
    root_origin_shift_m: list = field(default_factory=list)
    root_yaw_removed_deg: float = 0.0
    rest_height_m: float = 0.0
    notes: list = field(default_factory=list)

    def dump(self, path: Path) -> None:
        path.write_text(yaml.safe_dump(
            {k: v for k, v in self.__dict__.items()}, sort_keys=False, default_flow_style=False))


# ── URDF limits ───────────────────────────────────────────────────────────────
def urdf_joint_limits(urdf: Path) -> dict[str, tuple[float, float]]:
    """joint name -> (effort N·m, no-load speed rad/s) from the vendor URDF."""
    root = ET.parse(str(urdf)).getroot()
    out: dict[str, tuple[float, float]] = {}
    for j in root.findall("joint"):
        if j.get("type") not in ("revolute", "continuous"):
            continue
        lim = j.find("limit")
        if lim is None:
            continue
        eff, vel = lim.get("effort"), lim.get("velocity")
        if eff is None or vel is None:
            continue
        out[str(j.get("name"))] = (float(eff), float(vel))
    return out


# ── frame probing ─────────────────────────────────────────────────────────────
def _probe_frames(model: mujoco.MjModel, joint_names: dict[str, str],
                  foot_site: dict[str, str] | None = None) -> dict:
    """World-frame joint axes/anchors at the neutral pose (root upright at origin)."""
    data = mujoco.MjData(model)
    data.qpos[:] = 0.0
    if model.nq >= 7 and model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE:
        data.qpos[3] = 1.0  # identity quaternion
    mujoco.mj_forward(model, data)
    out: dict = {"axis": {}, "anchor": {}, "foot": {}}
    for key, name in joint_names.items():
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if i < 0:
            raise KeyError(f"joint '{name}' not in model")
        out["axis"][key] = np.array(data.xaxis[i])
        out["anchor"][key] = np.array(data.xanchor[i])
    for key, name in (foot_site or {}).items():
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        if i >= 0:
            out["foot"][key] = np.array(data.site_xpos[i])
    return out


#: Per-DOF actuator internals (not shape or mass) matched across the pair.
_ACTUATOR_INTERNALS = ("armature", "damping", "frictionloss")


def _match_actuator_internals(spec: mujoco.MjSpec, ship: mujoco.MjModel,
                              twin: mujoco.MjModel, twin_spec: mujoco.MjSpec,
                              rep: AdaptReport) -> None:
    """Copy the twin's armature, damping and frictionloss onto the shipped model.

    None of the three is measured on the real robot (vendors omit them or tune
    them for their own controller), so matching them leaves mass, inertia and
    geometry as the only difference. Originals are recorded in the report.
    """
    for new_joint in rep.joint_renames.values():
        t_id = mujoco.mj_name2id(twin, mujoco.mjtObj.mjOBJ_JOINT, new_joint)
        s_id = mujoco.mj_name2id(ship, mujoco.mjtObj.mjOBJ_JOINT, new_joint)
        if t_id < 0 or s_id < 0:
            continue
        t_dof, s_dof = int(twin.jnt_dofadr[t_id]), int(ship.jnt_dofadr[s_id])
        jnt, t_jnt = spec.joint(new_joint), twin_spec.joint(new_joint)
        before = {"armature": float(ship.dof_armature[s_dof]),
                  "damping": float(ship.dof_damping[s_dof]),
                  "frictionloss": float(ship.dof_frictionloss[s_dof])}
        after = {"armature": float(twin.dof_armature[t_dof]),
                 "damping": float(twin.dof_damping[t_dof]),
                 "frictionloss": float(twin.dof_frictionloss[t_dof])}
        # Copy from the twin's spec: MjsJoint damping is a 3-vector, keep its shape.
        for name in _ACTUATOR_INTERNALS:
            value = getattr(t_jnt, name)
            setattr(jnt, name, np.array(value, dtype=float)
                    if isinstance(value, np.ndarray) else float(value))
        rep.actuator_internals[new_joint] = {
            "vendor": {k: round(v, 6) for k, v in before.items()},
            "twin": {k: round(v, 6) for k, v in after.items()}}
    rep.notes.append(
        "actuator internals (armature/damping/frictionloss) taken from the twin — "
        "none of the three is measured from the real robot, and the vendor's "
        "damping alone would consume most of the peak torque; see "
        "_match_actuator_internals")


def _SEGMENTS(leg: str):
    """(segment, the joint that swings it, start point, end point) for one leg."""
    return (
        ("thigh", f"{leg}_hip_pitch", ("anchor", f"{leg}_hip_pitch"), ("anchor", f"{leg}_knee")),
        ("shank", f"{leg}_knee", ("anchor", f"{leg}_knee"), ("foot", leg)),
    )


def _signed_angle(v_from: np.ndarray, v_to: np.ndarray, axis: np.ndarray) -> float:
    """Rotation (rad) about ``axis`` carrying ``v_from`` onto ``v_to``, after
    projecting both onto the plane perpendicular to the axis."""
    axis = axis / np.linalg.norm(axis)
    p_from = v_from - float(np.dot(v_from, axis)) * axis
    p_to = v_to - float(np.dot(v_to, axis)) * axis
    if np.linalg.norm(p_from) < 1e-9 or np.linalg.norm(p_to) < 1e-9:
        return 0.0
    return float(math.atan2(float(np.dot(axis, np.cross(p_from, p_to))),
                            float(np.dot(p_from, p_to))))


def _zero_pose_errors(ship: mujoco.MjModel, twin: mujoco.MjModel) -> dict[str, float]:
    """Per-segment zero-pose disagreement (deg), measured about each joint's own axis.

    The axis-parallel component is a lateral offset, reported separately.
    """
    names = {f"{leg}_{role}": f"{leg}_{role}" for leg in _LEGS for role in _QUAD_ROLES}
    feet = {leg: f"{leg}_foot" for leg in _LEGS}
    p_ship, p_twin = _probe_frames(ship, names, feet), _probe_frames(twin, names, feet)
    out: dict[str, float] = {}
    for leg in _LEGS:
        for seg, joint, a, b in _SEGMENTS(leg):
            v_ship = p_ship[b[0]][b[1]] - p_ship[a[0]][a[1]]
            v_twin = p_twin[b[0]][b[1]] - p_twin[a[0]][a[1]]
            out[f"{leg}_{seg}"] = math.degrees(
                _signed_angle(v_twin, v_ship, p_twin["axis"][joint]))
    return out


def _align_zero_reference(spec: mujoco.MjSpec, model: mujoco.MjModel, twin: mujoco.MjModel,
                          rep: AdaptReport, key: str) -> mujoco.MjModel:
    """Make ``q = 0`` the same physical pose on the shipped robot as on the twin.

    A pure relabelling: each child body is rotated about its joint's axis (baked
    into the frame, so ``qpos0`` is the aligned pose). Iterated, since a shank's
    direction depends on the thigh's zero.
    """
    for _ in range(_ALIGN_PASSES):
        errs = _zero_pose_errors(model, twin)
        if max(abs(e) for e in errs.values()) <= _ZERO_POSE_TOL_DEG:
            break
        for leg in _LEGS:
            for seg, joint, _a, _b in _SEGMENTS(leg):
                delta = math.radians(errs[f"{leg}_{seg}"])
                if abs(delta) < math.radians(_ZERO_POSE_TOL_DEG):
                    continue
                bake_joint_rotation(spec, joint, -delta)
                rep.zero_offsets_deg[joint] = round(
                    rep.zero_offsets_deg.get(joint, 0.0) + math.degrees(delta), 3)
        model = spec.compile()

    errs = _zero_pose_errors(model, twin)
    rep.zero_pose_error_deg = {k: round(v, 3) for k, v in errs.items()}
    worst = max(abs(v) for v in errs.values()) if errs else 0.0
    if worst > _ZERO_POSE_TOL_DEG:
        raise ValueError(
            f"{key}: shipped and twin still disagree about the zero pose by {worst:.1f} deg "
            f"after {_ALIGN_PASSES} alignment passes — the same policy output would be a "
            f"different pose on each, so the pair is not comparable. Per segment: "
            f"{rep.zero_pose_error_deg}")
    if rep.zero_offsets_deg:
        rep.notes.append(
            "shifted joint zero references (and their ranges) onto the tree's convention: "
            + ", ".join(f"{k} {v:+.1f} deg" for k, v in sorted(rep.zero_offsets_deg.items())))
    return model


def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return math.degrees(math.acos(float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))))


# ── the adapter ───────────────────────────────────────────────────────────────
def localise_assets(spec: mujoco.MjSpec, out_dir: Path, rep=None) -> int:
    """Copy every referenced mesh into ``<out_dir>/assets/`` so the model is portable.

    Meshes are referenced by basename; a basename collision raises.
    """
    if not spec.meshes:
        return 0
    assets = Path(out_dir) / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    root = Path(spec.meshdir) if spec.meshdir else Path(out_dir)
    seen: dict[str, str] = {}
    copied = 0
    for mesh in spec.meshes:
        if not mesh.file:
            continue
        src = Path(mesh.file)
        if not src.is_absolute():
            src = root / src
        base = src.name
        if base in seen and seen[base] != str(src):
            raise ValueError(
                f"asset basename collision localising {out_dir}: '{base}' comes from "
                f"both {seen[base]} and {src} — they would overwrite each other")
        if not src.exists():
            raise FileNotFoundError(
                f"{out_dir}: mesh '{mesh.file}' resolves to {src}, which does not "
                f"exist — the written model would be unloadable anywhere")
        if base not in seen:
            shutil.copyfile(src, assets / base)
            copied += 1
        seen[base] = str(src)
        mesh.file = base
    # Absolute in memory (`to_xml()` re-reads meshes and a relative meshdir does
    # not resolve reliably); `localised_xml` makes it relative in the file.
    spec.meshdir = str(assets.resolve())
    if rep is not None:
        rep.notes.append(
            f"copied {copied} mesh assets into assets/ and made meshdir relative — the "
            f"model pointed at absolute paths on the machine that built it, so it could "
            f"not load anywhere else (this is what training on another machine needs)")
    return copied


_MESHDIR_RE = re.compile(r'(<compiler\b[^>]*\bmeshdir=")([^"]*)(")')


def localised_xml(spec: mujoco.MjSpec) -> str:
    """`to_xml()` with the absolute `meshdir` replaced by the relative `assets`."""
    return _MESHDIR_RE.sub(r'\1assets\3', spec.to_xml(), count=1)


def normalise_geom_groups(spec: mujoco.MjSpec, rep=None) -> tuple[int, int]:
    """Move colliders to group 3 and visuals to group 2; drop coincident duplicate visuals.

    Rendering only (avoids z-fighting); no physics reads geom group.
    """
    moved = 0
    for geom in spec.geoms:
        collides = bool(geom.contype or geom.conaffinity)
        want = 3 if collides else 2
        if int(geom.group) != want:
            geom.group = want
            moved += 1

    seen: dict[tuple, object] = {}
    dupes = []
    for body in spec.bodies:
        for geom in list(body.geoms):
            if geom.contype or geom.conaffinity:
                continue                      # never dedupe a collider
            key = (body.name, int(geom.type), geom.meshname or "",
                   tuple(np.round(np.asarray(geom.pos, float), 6)),
                   tuple(np.round(np.asarray(geom.size, float), 6)))
            if key in seen:
                dupes.append(geom)
            else:
                seen[key] = geom
    for geom in dupes:
        spec.delete(geom)

    if rep is not None and (moved or dupes):
        rep.notes.append(
            f"regrouped {moved} geoms (collision -> 3, visual -> 2) and deleted "
            f"{len(dupes)} coincident duplicate visuals — the source model drew "
            f"collision and visual meshes in the same visible groups, which "
            f"z-fights and reads as the surface flickering between colours")
    return moved, len(dupes)


def _classify_joints(spec: mujoco.MjSpec, target: Target) -> dict[str, tuple[str, str]]:
    """vendor joint name -> (leg prefix, canonical role), for the 12 leg joints."""
    out: dict[str, tuple[str, str]] = {}
    for jnt in spec.joints:
        if jnt.type == mujoco.mjtJoint.mjJNT_FREE or not jnt.name:
            continue
        role = next((r for pat, r in target.role_map if re.search(pat, jnt.name)), None)
        if role not in _QUAD_ROLES:
            continue
        side = detect_side(jnt.name, "quadruped")
        if side is None:
            continue
        out[jnt.name] = (_SIDE_TO_PREFIX[side], role)
    return out


def _rigid_subtree(body: mujoco.MjsBody) -> list[mujoco.MjsBody]:
    """``body`` plus every descendant rigidly attached to it (e.g. a URDF foot body)."""
    out, stack = [body], [body]
    while stack:
        for child in stack.pop().bodies:
            if not list(child.joints):
                out.append(child)
                stack.append(child)
    return out


def _foot_geom(body: mujoco.MjsBody) -> mujoco.MjsGeom | None:
    """The foot contact sphere: the lowest collision sphere rigidly on the shank."""
    spheres = [g for b in _rigid_subtree(body) for g in b.geoms
               if g.type == mujoco.mjtGeom.mjGEOM_SPHERE and (g.contype or g.conaffinity)]
    if not spheres:
        return None
    return min(spheres, key=lambda g: float(g.pos[2]))


def adapt_quadruped(
    target: Target,
    shipped_mjcf: Path,
    twin_mjcf: Path,
    out_dir: Path,
) -> tuple[Path, AdaptReport]:
    """Rewrite a shipped quadruped MJCF into the RL env's naming contract.

    Returns the adapted robot XML path and its `AdaptReport`.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rep = AdaptReport(key=target.key, source_mjcf=str(shipped_mjcf), twin_mjcf=str(twin_mjcf))

    spec = mujoco.MjSpec.from_file(str(shipped_mjcf))
    # Absolute asset paths while working; `localise_assets` makes them portable.
    src_dir = Path(shipped_mjcf).parent
    if spec.meshdir and not Path(spec.meshdir).is_absolute():
        spec.meshdir = str((src_dir / spec.meshdir).resolve())
    elif not spec.meshdir:
        spec.meshdir = str(src_dir.resolve())
    if spec.texturedir and not Path(spec.texturedir).is_absolute():
        spec.texturedir = str((src_dir / spec.texturedir).resolve())

    jmap = _classify_joints(spec, target)
    if len(jmap) != 12:
        raise ValueError(f"{target.key}: classified {len(jmap)} leg joints, expected 12 "
                         f"({sorted(jmap)})")

    limits = urdf_joint_limits(Path(target.urdf))

    # ── 1. joints: rename, and carry each one's limits to its new name ────────
    vendor_of_new: dict[str, str] = {}
    for jnt in list(spec.joints):
        if jnt.name not in jmap:
            continue
        leg, role = jmap[jnt.name]
        new = f"{leg}_{role}"
        vendor_of_new[new] = jnt.name
        rep.joint_renames[jnt.name] = new

    # ── 2. bodies: the body a joint lives in IS the link that joint drives ────
    body_of_new_joint: dict[str, mujoco.MjsBody] = {}
    for body in spec.bodies:
        for jnt in body.joints:
            if jnt.name in jmap:
                leg, role = jmap[jnt.name]
                body_of_new_joint[f"{leg}_{role}"] = body

    # Never rename the worldbody.
    world = spec.worldbody
    robot_bodies = [b for b in spec.bodies if b.id != world.id]

    root_body = next(
        (b for b in robot_bodies if any(j.type == mujoco.mjtJoint.mjJNT_FREE for j in b.joints)),
        None)
    if root_body is None:
        # URDF-derived model: fixed base, so give the sole top-level body a free joint.
        tops = [b for b in world.bodies]
        if len(tops) != 1:
            raise ValueError(f"{target.key}: no free joint and {len(tops)} top-level bodies — "
                             f"cannot identify the trunk")
        root_body = tops[0]
        root_body.add_freejoint()
        rep.notes.append("added a free joint to the trunk (URDF-derived model had a fixed base)")

    # Rename in two passes so a vendor name can never collide with a new name.
    _stash = "__adapt__"
    for body in robot_bodies:
        if body.name:
            body.name = _stash + body.name
    for jnt in spec.joints:
        if jnt.name:
            jnt.name = _stash + jnt.name
    for geom in spec.geoms:
        if geom.name:
            geom.name = _stash + geom.name
    for site in spec.sites:
        if site.name:
            site.name = _stash + site.name

    rep.body_renames[root_body.name.removeprefix(_stash)] = _TORSO
    root_body.name = _TORSO
    for new_joint, body in body_of_new_joint.items():
        leg, role = new_joint.split("_", 1)
        new_body = f"{leg}_{_QUAD_ROLE_BODY[role]}"
        rep.body_renames[body.name.removeprefix(_stash)] = new_body
        body.name = new_body
    for jnt in spec.joints:
        raw = jnt.name.removeprefix(_stash)
        if raw in jmap:
            leg, role = jmap[raw]
            jnt.name = f"{leg}_{role}"
        elif jnt.type == mujoco.mjtJoint.mjJNT_FREE:
            jnt.name = f"{_TORSO}_freejoint"
        else:
            jnt.name = raw
    for geom in spec.geoms:
        geom.name = geom.name.removeprefix(_stash)
    for site in spec.sites:
        site.name = site.name.removeprefix(_stash)

    # Route every by-name reference (lights, cameras, excludes, actuators,
    # sensors) through the rename maps.
    def _remap(name: str) -> str:
        raw = name.removeprefix(_stash)
        return rep.body_renames.get(raw, raw)

    #: Rename map per sensor target type (geoms/sites only lose the stash prefix).
    _SENSOR_MAP = {mujoco.mjtObj.mjOBJ_BODY: rep.body_renames,
                   mujoco.mjtObj.mjOBJ_XBODY: rep.body_renames,
                   mujoco.mjtObj.mjOBJ_JOINT: rep.joint_renames}

    def _remap_sensor(objtype, name: str) -> str:
        raw = name.removeprefix(_stash)
        return _SENSOR_MAP.get(objtype, {}).get(raw, raw)

    for obj in list(spec.lights) + list(spec.cameras):
        if getattr(obj, "targetbody", ""):
            obj.targetbody = _remap(obj.targetbody)
    for pair in spec.excludes:
        pair.bodyname1 = _remap(pair.bodyname1)
        pair.bodyname2 = _remap(pair.bodyname2)
    for act in spec.actuators:
        raw = act.target.removeprefix(_stash) if act.target else ""
        act.target = rep.joint_renames.get(raw, raw)
    # Vendor sensors are kept (read-only) and renamed.
    for sen in spec.sensors:
        if sen.objname:
            sen.objname = _remap_sensor(sen.objtype, sen.objname)
        if getattr(sen, "refname", ""):
            sen.refname = _remap_sensor(sen.reftype, sen.refname)

    # ── 3. feet: name the contact sphere, and put a site at its centre ────────
    for leg in _LEGS:
        geom = _foot_geom(spec.body(f"{leg}_lower_leg_link"))
        if geom is None:
            raise ValueError(f"{target.key}: no collision sphere on {leg}_lower_leg_link — "
                             f"the env's foot contact sensor has nothing to match")
        geom.name = f"{leg}_lower_leg_link_sphere"
        rep.foot_radius_m[leg] = float(geom.size[0])

    # The site goes on the shank body itself: ``measure_quadruped_leg_length``
    # reads its local offset as the shank length.
    probe = spec.compile()
    pdata = mujoco.MjData(probe)
    mujoco.mj_forward(probe, pdata)
    for leg in _LEGS:
        gid = mujoco.mj_name2id(probe, mujoco.mjtObj.mjOBJ_GEOM, f"{leg}_lower_leg_link_sphere")
        bid = mujoco.mj_name2id(probe, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_lower_leg_link")
        r_shank = pdata.xmat[bid].reshape(3, 3)
        local = r_shank.T @ (pdata.geom_xpos[gid] - pdata.xpos[bid])
        site = spec.body(f"{leg}_lower_leg_link").add_site()
        site.name = f"{leg}_foot"
        site.pos = local
        site.size = np.array([0.005, 0.005, 0.005])

    # mujoco_warp requires zero margin under MULTICCD, and the twin has none.
    n_margin = sum(1 for g in spec.geoms if float(g.margin) or float(g.gap))
    for geom in spec.geoms:
        geom.margin = 0.0
        geom.gap = 0.0
    if n_margin:
        rep.notes.append(f"zeroed contact margin/gap on {n_margin} geoms "
                         f"(mujoco_warp requires it under MULTICCD; the twin has none)")

    # ── 3a. only the feet may touch the ground ───────────────────────────────
    # The env's collision filter matches by name and cannot reach unnamed vendor
    # geoms, so disable them here to match the twin's four foot spheres.
    foot_geoms = {f"{leg}_lower_leg_link_sphere" for leg in _LEGS}
    n_off = 0
    for geom in spec.geoms:
        if geom.name in foot_geoms:
            continue
        if geom.contype or geom.conaffinity:
            geom.contype = 0
            geom.conaffinity = 0
            n_off += 1
    rep.notes.append(f"disabled collision on {n_off} non-foot geoms so the shipped "
                     f"model collides exactly where the twin does (4 foot spheres); "
                     f"the env's name-based filter cannot reach unnamed vendor geoms")

    # The env references a torso 'imu' site; add one if missing.
    if not any(s.name == "imu" for s in spec.sites):
        imu = spec.body(_TORSO).add_site()
        imu.name = "imu"
        imu.pos = np.zeros(3)
        imu.size = np.array([0.005, 0.005, 0.005])
        rep.notes.append("added an 'imu' site on the trunk (vendor model had none)")

    # ── 3b. the IMU sensor set the observation space is built from ───────────
    # Observations read these by name; the generator emits them, vendors do not.
    existing_sensors = {s.name for s in spec.sensors}
    _IMU = (
        ("imu_ang_vel", mujoco.mjtSensor.mjSENS_GYRO),
        ("imu_lin_vel", mujoco.mjtSensor.mjSENS_VELOCIMETER),
        ("imu_lin_acc", mujoco.mjtSensor.mjSENS_ACCELEROMETER),
    )
    for name, stype in _IMU:
        if name in existing_sensors:
            continue
        sensor = spec.add_sensor()
        sensor.name = name
        sensor.type = stype
        sensor.objtype = mujoco.mjtObj.mjOBJ_SITE
        sensor.objname = "imu"
    if "root_angmom" not in existing_sensors:
        sensor = spec.add_sensor()
        sensor.name = "root_angmom"
        sensor.type = mujoco.mjtSensor.mjSENS_SUBTREEANGMOM
        sensor.objtype = mujoco.mjtObj.mjOBJ_BODY
        sensor.objname = _TORSO
    rep.notes.append("added the imu_* / root_angmom sensor set the observation space reads")

    # ── 4. actuator forcerange + velocity-limit numerics ──────────────────────
    have_act = {a.target: a for a in spec.actuators}
    for new_joint, vendor_joint in vendor_of_new.items():
        if vendor_joint not in limits:
            raise KeyError(f"{target.key}: URDF declares no effort/velocity for '{vendor_joint}'")
        effort, vel = limits[vendor_joint]
        rep.effort_nm[new_joint] = effort
        rep.velocity_rad_s[new_joint] = vel
        num = spec.add_numeric()
        num.name = f"{new_joint}_velocity_limit"
        num.data = [vel]
    for act in spec.actuators:
        if act.target in rep.effort_nm:
            e = rep.effort_nm[act.target]
            act.forcerange = np.array([-e, e])
            act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE

    # Drop vendor keyframes; the env sets its own init state.
    for k in list(spec.keys):
        spec.delete(k)

    # ── 4b. actuator internals are taken from the twin, deliberately ─────────
    model = spec.compile()
    twin = mujoco.MjModel.from_xml_path(str(twin_mjcf))
    _match_actuator_internals(spec, model, twin,
                              mujoco.MjSpec.from_file(str(twin_mjcf)), rep)

    # ── 5. compile, then align frames against the twin ───────────────────────
    model = spec.compile()

    probe_names = {f"{leg}_{role}": f"{leg}_{role}" for leg in _LEGS for role in _QUAD_ROLES}
    foot_names = {leg: f"{leg}_foot" for leg in _LEGS}
    p_ship = _probe_frames(model, probe_names, foot_names)
    p_twin = _probe_frames(twin, probe_names, foot_names)

    for name in probe_names:
        if float(np.dot(p_ship["axis"][name], p_twin["axis"][name])) < 0.0:
            rep.flipped_joints.append(name)

    for leg in _LEGS:
        for seg, joint, a, b in _SEGMENTS(leg):
            axis = p_twin["axis"][joint]
            axis = axis / np.linalg.norm(axis)
            v_ship = p_ship[b[0]][b[1]] - p_ship[a[0]][a[1]]
            v_twin = p_twin[b[0]][b[1]] - p_twin[a[0]][a[1]]
            rep.lateral_offset_mm[f"{leg}_{seg}"] = round(
                1000.0 * (float(np.dot(v_ship, axis)) - float(np.dot(v_twin, axis))), 2)

    if rep.flipped_joints:
        for name in rep.flipped_joints:
            jnt = spec.joint(name)
            jnt.axis = -np.array(jnt.axis, dtype=float)
            lo, hi = float(jnt.range[0]), float(jnt.range[1])
            jnt.range = np.array([-hi, -lo])
        model = spec.compile()
        rep.notes.append(f"negated {len(rep.flipped_joints)} joint axes to match the twin's sign "
                         f"convention: {rep.flipped_joints}")

    model = _align_zero_reference(spec, model, twin, rep, target.key)

    # ── 5b. stand the model on the floor, facing +x, at its rest pose ─────────
    # Root placement (origin, yaw, height) is a convention; match it to the twin.
    origin_shift = align_root_origin(spec, twin, _TORSO)
    yaw, height = normalize_root_frame(spec, _TORSO)
    model = spec.compile()
    rep.root_origin_shift_m = [round(float(v), 5) for v in origin_shift]
    if float(np.linalg.norm(origin_shift)) > 1e-4:
        rep.notes.append(
            f"slid the root body's ORIGIN by {np.round(origin_shift, 3).tolist()} m onto the "
            f"twin's — the two descriptions put `torso_link` at different points of the "
            f"trunk, and every body position a policy reads is expressed in that frame")
    rep.root_yaw_removed_deg = yaw
    rep.rest_height_m = height
    if abs(yaw) > 1e-3:
        rep.notes.append(f"root body carried a {yaw:+.1f} deg yaw in the world; removed so "
                         f"the pair faces the same way")

    rep.total_mass_kg = float(model.body_mass.sum())

    # ── 6. write ─────────────────────────────────────────────────────────────
    # Make mesh paths portable (as for the humanoids).
    normalise_geom_groups(spec, rep)
    localise_assets(spec, out_dir, rep)
    robot_xml = out_dir / "quadruped.xml"
    robot_xml.write_text(_single_valued_sphere_sizes(localised_xml(spec)))
    _write_scene(out_dir, robot_xml.name)
    rep.dump(out_dir / "rl_adapter_report.yaml")
    return robot_xml, rep


_SPHERE_SIZE_RE = re.compile(
    r'(<geom\b[^>]*\bname="(?:fl|fr|rl|rr)_lower_leg_link_sphere"[^>]*\bsize=")([^"]+)(")')


def _single_valued_sphere_sizes(xml: str) -> str:
    """Write foot-sphere ``size`` as one number (``measure_foot_sphere_radius``
    parses it with ``float()``)."""
    return _SPHERE_SIZE_RE.sub(lambda m: m.group(1) + m.group(2).split()[0] + m.group(3), xml)


def _write_scene(out_dir: Path, robot_file: str) -> None:
    (out_dir / "scene.xml").write_text(
        f'<mujoco model="scene">\n'
        f'  <include file="{robot_file}"/>\n'
        f'  <statistic center="0 0 0.3" extent="1.2"/>\n'
        f'  <visual>\n'
        f'    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>\n'
        f'    <rgba haze="0.15 0.25 0.35 1"/>\n'
        f'    <global azimuth="140" elevation="-20"/>\n'
        f'  </visual>\n'
        f'  <asset>\n'
        f'    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" '
        f'rgb2="0 0 0" width="512" height="3072"/>\n'
        f'    <texture type="2d" name="groundplane" builtin="checker" mark="edge" '
        f'rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" '
        f'width="300" height="300"/>\n'
        f'    <material name="groundplane" texture="groundplane" texuniform="true" '
        f'texrepeat="5 5" reflectance="0.2"/>\n'
        f'  </asset>\n'
        f'  <worldbody>\n'
        f'    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>\n'
        f'    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>\n'
        f'  </worldbody>\n'
        f'</mujoco>\n')
