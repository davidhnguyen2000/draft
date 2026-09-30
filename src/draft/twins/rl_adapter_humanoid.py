"""Rewrite a vendor humanoid MJCF into the RL env's naming contract.

Unlike the quadruped case, the vendor roots at the pelvis and the tree at
``torso_link``, so the trunk chain is rerooted first (:mod:`.reroot`, checked by
FK). Joints are then paired with the twin by measured axis direction, the rest
pose is aligned, and the same contract additions as :mod:`.rl_adapter` are made.
Vendor masses, inertias, link lengths and joint limits are kept.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
import yaml

from .render import PairRenderer   # for _blob_hands
from .reroot import _mat, _quat_conj, _quat_mul, reroot, verify_reroot
from .rl_adapter import (AdaptReport, _match_actuator_internals,
                         localise_assets, localised_xml, normalise_geom_groups,
                         urdf_joint_limits)
from .targets import GENERATED_HUMANOID_ROLES, Target
from .zero_align import (align_rest_pose, align_root_origin, chain_probes,
                         normalize_joint_refs, normalize_root_frame)

#: FK agreement the reroot must reach.
_REROOT_TOL_M = 1e-6
_REROOT_TOL_DEG = 1e-3

#: Per-joint zero-reference agreement with the twin, after alignment.
_ZERO_TOL_DEG = 1.0
_ALIGN_PASSES = 4
#: Max times the joint pairing may be re-measured and the model rebuilt.
_MAX_PAIRING_PASSES = 3

_TORSO = "torso_link"


@dataclass
class HumanoidAdaptReport(AdaptReport):
    reroot_error_mm: float = 0.0
    reroot_error_deg: float = 0.0
    axis_role_mismatches: list = field(default_factory=list)
    worst_zero_error_deg: float = 0.0
    unobservable_zero_joints: list = field(default_factory=list)
    root_origin_shift_m: list = field(default_factory=list)
    root_yaw_removed_deg: float = 0.0
    rest_height_m: float = 0.0
    welded_joints: list = field(default_factory=list)


def _actuated(model: mujoco.MjModel) -> list[tuple[str, str]]:
    """(joint name, the body it drives), in model order, free joint excluded."""
    out = []
    for i in range(model.njnt):
        if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        out.append((mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i),
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.jnt_bodyid[i]))))
    return out


def _neutral(model: mujoco.MjModel) -> mujoco.MjData:
    data = mujoco.MjData(model)
    data.qpos[:] = 0.0
    if model.nq >= 7 and model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE:
        data.qpos[3] = 1.0
    mujoco.mj_forward(model, data)
    return data


def _body_frame(model, data, name):
    i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    return np.array(data.xpos[i]), np.array(data.xquat[i])


def _rel_to_torso(model, data, name):
    """A body's pose relative to the torso — the frame both models share."""
    p_t, q_t = _body_frame(model, data, _TORSO)
    p_b, q_b = _body_frame(model, data, name)
    return _mat(q_t).T @ (p_b - p_t), _quat_mul(_quat_conj(q_t), q_b)


def _site_local(model, data, site_name: str, body_name: str):
    """A site's position in its body's frame (works across differing geometry)."""
    si = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if si < 0:
        return None
    p_b, q_b = _body_frame(model, data, body_name)
    return _mat(q_b).T @ (np.array(data.site_xpos[si]) - p_b)


def _site_rel_to_torso(model, data, site_name: str):
    """A site's position in the TORSO frame — the frame both models share."""
    si = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if si < 0:
        return None
    p_t, q_t = _body_frame(model, data, _TORSO)
    return _mat(q_t).T @ (np.array(data.site_xpos[si]) - p_t)


def _transplant_site(spec, model, data, twin, twin_data, site_name: str, body_name: str):
    """Add ``site_name`` on ``body_name`` where the twin has it, via the torso frame.

    Only valid after the rest pose is aligned (step 7). The body frames themselves
    may be oriented differently, so their local offsets are not copied.
    """
    p_site = _site_rel_to_torso(twin, twin_data, site_name)
    if p_site is None:
        return None
    p_b, q_b = _rel_to_torso(model, data, body_name)
    local = _mat(q_b).T @ (p_site - p_b)
    site = spec.body(body_name).add_site()
    site.name = site_name
    site.pos = local
    site.size = np.array([0.005] * 3)
    return local


def _hand_blob_local(model, data, side: str):
    """Centre of the hand blob (from ``_blob_hands``) in ``<side>_hand_link``'s frame.

    Several same-place blobs may exist (one per render group); the first is used.
    """
    body = f"{side}_hand_link"
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        if "_hand_blob_" not in name or not name.startswith(side):
            continue
        p_b, q_b = _body_frame(model, data, body)
        return _mat(q_b).T @ (np.array(data.geom_xpos[gid]) - p_b)
    return None


def _geom_points(model: mujoco.MjModel, gid: int) -> np.ndarray:
    """Points bounding one geom, in its BODY's frame — exact for meshes and boxes."""
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, np.asarray(model.geom_quat[gid], float))
    rot = rot.reshape(3, 3)
    pos = np.asarray(model.geom_pos[gid], float)
    gtype, size = int(model.geom_type[gid]), np.asarray(model.geom_size[gid], float)
    if gtype == int(mujoco.mjtGeom.mjGEOM_MESH):
        mid = int(model.geom_dataid[gid])
        adr, num = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
        local = np.asarray(model.mesh_vert[adr:adr + num], float).reshape(-1, 3)
    elif gtype == int(mujoco.mjtGeom.mjGEOM_BOX):
        local = np.array([[sx, sy, sz] for sx in (-size[0], size[0])
                          for sy in (-size[1], size[1]) for sz in (-size[2], size[2])])
    else:                                   # sphere, capsule, cylinder, …
        r = float(model.geom_rbound[gid])
        local = np.array([[sx, sy, sz] for sx in (-r, r) for sy in (-r, r) for sz in (-r, r)])
    return local @ rot.T + pos


def _sole_corners(model: mujoco.MjModel, body_name: str) -> list[np.ndarray] | None:
    """Four sole footprint corners in the foot body's frame, front pair first
    (same order as the twin's contact spheres; indices 0/1 straddle the toe)."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        return None
    pts = [_geom_points(model, g) for g in range(model.ngeom)
           if int(model.geom_bodyid[g]) == bid
           and (model.geom_contype[g] or model.geom_conaffinity[g])]
    if not pts:
        return None
    p = np.vstack(pts)
    x0, x1 = float(p[:, 0].min()), float(p[:, 0].max())
    y0, y1 = float(p[:, 1].min()), float(p[:, 1].max())
    z0 = float(p[:, 2].min())
    return [np.array([x1, y0, z0]), np.array([x1, y1, z0]),
            np.array([x0, y0, z0]), np.array([x0, y1, z0])]


_AXIS_WORDS = ("yaw", "pitch", "roll")


def _family(role: str | None) -> str:
    """A role without its axis word: `wrist_yaw` and `wrist_roll` are one family."""
    if not role:
        return "?"
    head, _, tail = role.rpartition("_")
    return head if head and tail in _AXIS_WORDS else role


def _side(name: str) -> str:
    return next((s for s in ("left", "right") if name.startswith(s)), "")


def _axes_in_root(model: mujoco.MjModel) -> dict[str, np.ndarray]:
    """Every actuated joint's axis, in the root body's frame, at ``q = 0``."""
    data = _neutral(model)
    _p, q_t = _body_frame(model, data, _TORSO)
    rot = _mat(q_t).T
    out = {}
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if name and model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE:
            out[name] = rot @ np.asarray(data.xaxis[j], float)
    return out


def permutation_by_axis_direction(twin: mujoco.MjModel, ship: mujoco.MjModel,
                                  twin_role) -> dict[str, str]:
    """twin joint -> shipped joint, matched within each family by axis direction.

    Axis names and chain order differ between vendors, so neither is used.
    Greedy on |dot| within one side and family (e.g. a wrist); identity where the
    pairing is already right. The caller rebuilds with the result.
    """
    a_twin, a_ship = _axes_in_root(twin), _axes_in_root(ship)
    groups: dict[tuple, list[str]] = {}
    for name in a_twin:
        groups.setdefault((_side(name), _family(twin_role(name))), []).append(name)

    out = {n: n for n in a_twin}
    for members in groups.values():
        members = [m for m in members if m in a_ship]
        if len(members) < 2:
            continue
        scores = sorted(((abs(float(np.dot(a_twin[t], a_ship[c]))), t, c)
                         for t in members for c in members), reverse=True)
        taken_t, taken_c = set(), set()
        for _score, t, c in scores:
            if t in taken_t or c in taken_c:
                continue
            taken_t.add(t)
            taken_c.add(c)
            out[t] = c
    return out


def adapt_humanoid(target: Target, shipped_mjcf: Path, twin_mjcf: Path,
                   out_dir: Path, pairing: dict[str, str] | None = None,
                   _attempt: int = 0) -> tuple[Path, HumanoidAdaptReport]:
    """Adapt a vendor humanoid onto the tree's naming contract.

    Pairing by axis direction needs aligned poses, which need a pairing, so this
    recurses: pair by chain depth, re-pair, align, re-pair (at most
    `_MAX_PAIRING_PASSES` rebuilds).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rep = HumanoidAdaptReport(key=target.key, source_mjcf=str(shipped_mjcf),
                              twin_mjcf=str(twin_mjcf))

    # ── 0. flatten includes and absolutise assets ────────────────────────────
    spec = mujoco.MjSpec.from_file(str(shipped_mjcf))
    src_dir = Path(shipped_mjcf).parent
    spec.meshdir = str((src_dir / spec.meshdir).resolve()) if spec.meshdir \
        else str(src_dir.resolve())
    if spec.texturedir and not Path(spec.texturedir).is_absolute():
        spec.texturedir = str((src_dir / spec.texturedir).resolve())

    # ── 0a. weld away the joints the tree has no equivalent for ──────────────
    # Vendor joints with no tree role (e.g. H2's neck) are removed so the pair is 1:1.
    tree_roles = {r for _p, r in GENERATED_HUMANOID_ROLES}
    for jnt in list(spec.joints):
        if jnt.type == mujoco.mjtJoint.mjJNT_FREE or not jnt.name:
            continue
        role = next((r for p, r in target.role_map if re.search(p, jnt.name)), None)
        if role in tree_roles:
            continue
        rep.welded_joints.append(jnt.name)
        for act in list(spec.actuators):
            if act.target == jnt.name:
                spec.delete(act)
        spec.delete(jnt)
    if rep.welded_joints:
        rep.notes.append(f"welded {len(rep.welded_joints)} vendor joints the parametric "
                         f"tree has no role for: {', '.join(rep.welded_joints)}")

    # URDF-derived targets have no free joint; add one.
    root_body = next((b for b in spec.bodies
                      if b.name and b.id != spec.worldbody.id
                      and b.parent.id == spec.worldbody.id), None)
    if root_body is not None and not any(
            j.type == mujoco.mjtJoint.mjJNT_FREE for j in root_body.joints):
        root_body.add_freejoint()
        rep.notes.append(f"added the free joint '{root_body.name}' needs to float "
                         f"(the source description has none)")

    flat = out_dir / "_flat.xml"
    flat.write_text(spec.to_xml())

    # ── 1. reroot at the torso, and prove it changed nothing ─────────────────
    rerooted = reroot(flat, _TORSO, out_dir / "_rerooted.xml")
    err_m, err_deg = verify_reroot(flat, rerooted)
    rep.reroot_error_mm = round(err_m * 1000.0, 6)
    rep.reroot_error_deg = round(err_deg, 6)
    if err_m > _REROOT_TOL_M or err_deg > _REROOT_TOL_DEG:
        raise ValueError(
            f"{target.key}: rerooting changed the kinematics — {err_m * 1000:.4f} mm / "
            f"{err_deg:.4f} deg FK disagreement with the original. Refusing to continue.")

    # ── 2. pair joints by chain position against the twin ────────────────────
    twin = mujoco.MjModel.from_xml_path(str(twin_mjcf))
    ship = mujoco.MjModel.from_xml_path(str(rerooted))
    twin_joints, ship_joints = _actuated(twin), _actuated(ship)
    if len(twin_joints) != len(ship_joints):
        raise ValueError(f"{target.key}: twin has {len(twin_joints)} actuated joints, "
                         f"shipped has {len(ship_joints)} — cannot pair by position")

    role_of = lambda n: next((r for p, r in target.role_map if re.search(p, n)), None)  # noqa: E731
    twin_role = lambda n: next(  # noqa: E731
        (r for p, r in __import__("draft.twins.targets", fromlist=["x"]).GENERATED_HUMANOID_ROLES
         if re.search(p, n)), None)

    # Bodies pair by chain depth. Joints pair by depth first, then by measured
    # axis direction (`permutation_by_axis_direction`).
    joint_pairs = pairing or {tj: sj for (tj, _tb), (sj, _sb)
                              in zip(twin_joints, ship_joints)}
    for tj, sj in joint_pairs.items():
        if twin_role(tj) != role_of(sj):
            rep.axis_role_mismatches.append({
                "joint": tj, "tree_axis": twin_role(tj), "vendor_axis": role_of(sj),
                "vendor_joint": sj,
                "note": "same physical axis, different vendor label — the two "
                        "descriptions name axes in different reference poses"})

    def _rebuild_with(perm: dict[str, str], when: str):
        """Recompose the vendor pairing through ``perm`` and start over."""
        vendor_of = {new: raw for raw, new in rep.joint_renames.items()}
        corrected = {t: vendor_of[s] for t, s in perm.items() if s in vendor_of}
        moved = [f"{t}<-{vendor_of.get(s, s)}" for t, s in perm.items() if t != s]
        print(f"    re-pairing {len(moved)} joints by measured axis direction "
              f"({when}): {', '.join(moved)}")
        return adapt_humanoid(target, shipped_mjcf, twin_mjcf, out_dir,
                              pairing=corrected, _attempt=_attempt + 1)

    limits = urdf_joint_limits(Path(target.urdf))

    # ── 3. rename joints and the bodies they drive onto the tree's names ─────
    spec = mujoco.MjSpec.from_file(str(rerooted))
    _stash = "__adapt__"
    world = spec.worldbody
    for body in spec.bodies:
        if body.id != world.id and body.name:
            body.name = _stash + body.name
    for jnt in spec.joints:
        if jnt.name:
            jnt.name = _stash + jnt.name

    rename_body: dict[str, str] = {}
    rename_joint: dict[str, str] = {}
    for tj, sj in joint_pairs.items():
        rename_joint[sj] = tj
    for (_tj, tb), (_sj, sb) in zip(twin_joints, ship_joints):
        rename_body[sb] = tb
    rename_body[_TORSO] = _TORSO

    # On a name clash with a contract name, the vendor's body/joint gets `_vendor`.
    claimed_bodies = set(rename_body.values())
    claimed_joints = set(rename_joint.values())

    for jnt in spec.joints:
        raw = jnt.name.removeprefix(_stash)
        if jnt.type == mujoco.mjtJoint.mjJNT_FREE:
            jnt.name = f"{_TORSO}_freejoint"
        elif raw in rename_joint:
            jnt.name = rename_joint[raw]
            rep.joint_renames[raw] = jnt.name
        else:
            jnt.name = f"{raw}_vendor" if raw in claimed_joints else raw
    for body in spec.bodies:
        if body.id == world.id or not body.name:
            continue
        raw = body.name.removeprefix(_stash)
        if raw in rename_body:
            body.name = rename_body[raw]
            rep.body_renames[raw] = body.name
        else:
            body.name = f"{raw}_vendor" if raw in claimed_bodies else raw
            if raw in claimed_bodies:
                rep.notes.append(f"vendor body '{raw}' renamed to '{body.name}' — the "
                                 f"contract needs that name for the link a joint drives")
    for geom in spec.geoms:
        geom.name = geom.name.removeprefix(_stash)
    for site in spec.sites:
        site.name = site.name.removeprefix(_stash)

    def _remap(name: str) -> str:
        raw = name.removeprefix(_stash)
        return rename_body.get(raw, raw)

    for obj in list(spec.lights) + list(spec.cameras):
        if getattr(obj, "targetbody", ""):
            obj.targetbody = _remap(obj.targetbody)
    for pair in spec.excludes:
        pair.bodyname1 = _remap(pair.bodyname1)
        pair.bodyname2 = _remap(pair.bodyname2)
    for act in spec.actuators:
        raw = act.target.removeprefix(_stash) if act.target else ""
        act.target = rename_joint.get(raw, raw)

    for k in list(spec.keys):
        spec.delete(k)

    # ── 4. limits, sensors, margins ──────────────────────────────────────────
    urdf_of_new = {rename_joint[sj]: sj for sj, _ in
                   [(s, b) for s, b in ship_joints]}
    for new_joint, vendor_joint in urdf_of_new.items():
        base = vendor_joint  # URDF and MJCF agree on joint names for these targets
        if base not in limits:
            raise KeyError(f"{target.key}: URDF has no effort/velocity for '{base}'")
        effort, vel = limits[base]
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
    # Set the joint's actfrcrange to the same URDF effort: mjlab may read either field.
    for jnt in spec.joints:
        if jnt.name in rep.effort_nm:
            e = rep.effort_nm[jnt.name]
            jnt.actfrcrange = np.array([-e, e])
            jnt.actfrclimited = mujoco.mjtLimited.mjLIMITED_TRUE

    n_margin = sum(1 for g in spec.geoms if float(g.margin) or float(g.gap))
    for geom in spec.geoms:
        geom.margin = 0.0
        geom.gap = 0.0
    if n_margin:
        rep.notes.append(f"zeroed contact margin/gap on {n_margin} geoms")

    # ── 4b. un-gate MJX-style collision geoms ────────────────────────────────
    # MJX models disable all geom collisions and re-enable them via <contact>
    # pairs naming a scene floor that is not in the robot file. Re-enable the
    # primitive geoms; meshes (the visual layer) stay non-colliding.
    prims = [g for g in spec.geoms if g.type != mujoco.mjtGeom.mjGEOM_MESH]
    if prims and not any(g.contype or g.conaffinity for g in spec.geoms):
        for geom in prims:
            geom.contype, geom.conaffinity = 1, 1
        rep.notes.append(
            f"model gated ALL collisions off behind <contact> pairs (the MJX "
            f"convention) and every pair named the scene's floor, so the robot "
            f"collided with nothing on its own — re-enabled the {len(prims)} "
            f"primitive collision geoms; meshes left non-colliding")

    # ── 5. the marker sites the twin carries ─────────────────────────────────
    # Foot markers go on the vendor's own sole; the hand marker is placed in step 8b.
    twin_data = _neutral(twin)
    _shape_model = spec.compile()      # geometry, post-rename, to measure soles on
    for side in ("left", "right"):
        foot_body = f"{side}_foot_link"
        try:
            body = spec.body(foot_body)
        except (KeyError, ValueError):
            raise ValueError(f"{target.key}: no '{foot_body}' after renaming — "
                             f"the ankle-roll body did not map through") from None
        spheres = [g for g in body.geoms
                   if g.type == mujoco.mjtGeom.mjGEOM_SPHERE and (g.contype or g.conaffinity)]
        spheres.sort(key=lambda g: (-float(g.pos[0]), float(g.pos[1])))
        if len(spheres) >= 4:
            # Sole already modelled as four contact spheres: use them.
            corners = [np.array(g.pos, dtype=float) for g in spheres[:4]]
            for geom, name in zip(spheres, [f"{side}_foot_link_sphere_{i}" for i in range(4)]):
                geom.name = name
        else:
            # Otherwise use the footprint corners of the collidable foot geometry.
            corners = _sole_corners(_shape_model, foot_body)
            if corners is None:
                raise ValueError(
                    f"{target.key}: {foot_body} has {len(spheres)} contact spheres and no "
                    f"collidable geometry to take a sole footprint from")
        for i, pos in enumerate(corners):
            site = body.add_site()
            site.name = f"{side}_foot_link_sphere_{i}"
            site.pos = np.asarray(pos, dtype=float)
            site.size = np.array([0.005] * 3)
        toe = body.add_site()
        toe.name = f"{side}_foot_link_toe_sphere"
        toe.pos = 0.5 * (np.asarray(corners[0], float) + np.asarray(corners[1], float))
        toe.size = np.array([0.005] * 3)

    # Vendor without a head body: add a massless 'head_link' marker at the twin's offset.
    added_head = not any(b.name == "head_link" for b in spec.bodies)
    if added_head:
        p_rel, q_rel = _rel_to_torso(twin, twin_data, "head_link")
        head = spec.body(_TORSO).add_body()
        head.name = "head_link"
        head.pos = p_rel
        head.quat = q_rel
        head.mass = 0.0
        head.inertia = np.zeros(3)
        head.explicitinertial = True
        rep.notes.append("added a massless 'head_link' marker at the twin's head offset "
                         "(G1 has no head body; the retargeter tracks a site on it)")

    if not any(s.name == "imu" for s in spec.sites):
        imu = spec.body(_TORSO).add_site()
        imu.name = "imu"
        imu.pos = np.zeros(3)
        imu.size = np.array([0.005] * 3)

    existing_sensors = {s.name for s in spec.sensors}
    for name, stype, otype, oname in (
        ("imu_ang_vel", mujoco.mjtSensor.mjSENS_GYRO, mujoco.mjtObj.mjOBJ_SITE, "imu"),
        ("imu_lin_vel", mujoco.mjtSensor.mjSENS_VELOCIMETER, mujoco.mjtObj.mjOBJ_SITE, "imu"),
        ("imu_lin_acc", mujoco.mjtSensor.mjSENS_ACCELEROMETER, mujoco.mjtObj.mjOBJ_SITE, "imu"),
        ("root_angmom", mujoco.mjtSensor.mjSENS_SUBTREEANGMOM, mujoco.mjtObj.mjOBJ_BODY, _TORSO),
    ):
        if name in existing_sensors:
            continue
        sensor = spec.add_sensor()
        sensor.name, sensor.type = name, stype
        sensor.objtype, sensor.objname = otype, oname

    model = spec.compile()

    # ── 6. axis sign, then zero reference, both against the twin ─────────────
    def axes(m, d):
        out = {}
        for i in range(m.njnt):
            n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
            if n and m.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE:
                p_t, q_t = _body_frame(m, d, _TORSO)
                out[n] = _mat(q_t).T @ np.array(d.xaxis[i])
        return out

    a_twin = axes(twin, twin_data)
    a_ship = axes(model, _neutral(model))
    for name, axis in a_ship.items():
        if name in a_twin and float(np.dot(axis, a_twin[name])) < 0.0:
            rep.flipped_joints.append(name)
    for name in rep.flipped_joints:
        jnt = spec.joint(name)
        jnt.axis = -np.array(jnt.axis, dtype=float)
        lo, hi = float(jnt.range[0]), float(jnt.range[1])
        jnt.range = np.array([-hi, -lo])
    if rep.flipped_joints:
        model = spec.compile()
        rep.notes.append(f"negated {len(rep.flipped_joints)} joint axes onto the twin's "
                         f"sign convention")

    # ── 6b. is each joint already on the axis its name claims? ───────────────
    # Checked both before and after aligning the rest pose: some permutations
    # break alignment, others only show once the limbs are aligned.
    if _attempt == 0:
        perm = permutation_by_axis_direction(twin, model, twin_role)
        if any(t != s for t, s in perm.items()):
            return _rebuild_with(perm, "before aligning the rest pose")

    # ── 7. rest pose: q = 0 must be the same pose on both robots ─────────────
    # Measured by where each link points, not by body frames (see zero_align).
    normalize_joint_refs(spec)
    # Foot markers only.
    model, offsets, residual, blind = align_rest_pose(
        spec, twin, chain_probes(twin, site_ok=lambda n: "foot" in n),
        passes=25, tol_deg=_ZERO_TOL_DEG, root=_TORSO)
    rep.zero_offsets_deg = offsets
    rep.zero_pose_error_deg = residual
    rep.unobservable_zero_joints = blind
    rep.worst_zero_error_deg = round(
        max((abs(v) for v in residual.values()), default=0.0), 3)

    # ── 7b. now that both robots stand alike, re-check the axis pairing ──────
    if _attempt < _MAX_PAIRING_PASSES:
        perm = permutation_by_axis_direction(twin, model, twin_role)
        if any(t != s for t, s in perm.items()):
            return _rebuild_with(perm, "after aligning the rest pose")

    # ── 7c. axis signs, re-checked in the aligned pose ───────────────────────
    # Aligning a limb rotates every axis below it; a flip does not move q = 0.
    a_twin = _axes_in_root(twin)
    late = [n for n, axis in _axes_in_root(model).items()
            if n in a_twin and float(np.dot(axis, a_twin[n])) < 0.0]
    for name in late:
        jnt = spec.joint(name)
        jnt.axis = -np.array(jnt.axis, dtype=float)
        lo, hi = float(jnt.range[0]), float(jnt.range[1])
        jnt.range = np.array([-hi, -lo])
    if late:
        model = spec.compile()
        rep.flipped_joints += late
        rep.notes.append(f"negated {len(late)} more joint axes once the rest pose was "
                         f"aligned (straightening a limb turns every axis below it): "
                         + ", ".join(late))

    if rep.worst_zero_error_deg > _ZERO_TOL_DEG:
        raise ValueError(
            f"{target.key}: shipped and twin still disagree about the q=0 pose by "
            f"{rep.worst_zero_error_deg:.2f} deg after alignment — the same policy "
            f"output would be a different pose on each. Per joint: {residual}")
    if offsets:
        rep.notes.append(
            "rotated the vendor's rest pose onto the twin's (baked into the body "
            "frames, ranges carried with them): "
            + ", ".join(f"{k} {v:+.1f} deg" for k, v in sorted(offsets.items())))
    if blind:
        rep.notes.append(
            "these joints' zeros are not observable from geometry (the link they "
            "swing runs along their own axis), so the vendor's zero is kept: "
            + ", ".join(sorted(blind)))

    # ── 8. stand the model on the floor, facing +x, at its rest pose ─────────
    # An added head marker is excluded from origin alignment, then re-seated.
    origin_shift = align_root_origin(spec, twin, _TORSO,
                                     exclude={"head_link"} if added_head else None)
    if added_head:
        p_rel, q_rel = _rel_to_torso(twin, twin_data, "head_link")
        head = spec.body("head_link")
        head.pos, head.quat = p_rel, q_rel
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
        rep.notes.append(f"root body carried a {yaw:+.1f} deg yaw in the world; removed "
                         f"so the pair faces the same way")

    # ── 8b. end the arm in a blob, then mark the hand from the twin ──────────
    # The twin's arm ends in a sphere, so redraw the vendor hand the same way
    # (`_blob_hands`). Visual only: no collision, mass unchanged.
    PairRenderer._blob_hands(spec)
    model = spec.compile()

    # Hand marker at the blob centre (keeps the vendor's arm length); fallback is
    # the twin's marker carried over through the torso frame.
    _aligned = _neutral(model)
    for side in ("left", "right"):
        centre = _hand_blob_local(model, _aligned, side)
        if centre is not None:
            site = spec.body(f"{side}_hand_link").add_site()
            site.name = f"{side}_hand_sphere"
            site.pos = centre
            site.size = np.array([0.005] * 3)
        elif _transplant_site(spec, model, _aligned, twin, twin_data,
                              f"{side}_hand_sphere", f"{side}_hand_link") is None:
            rep.notes.append(f"no {side} hand blob and no twin marker to copy — the "
                             f"retargeter's wrist target falls back to a body offset")
    model = spec.compile()
    rep.notes.append(
        "hands redrawn as the equal-volume sphere the twin's arm ends in, and the "
        "`*_hand_sphere` marker the retargeter aims at re-seated at that sphere's "
        "centre (it used to be copied in the hand's own frame, which the two models "
        "orient differently — it landed ~112 mm from the palm on G1, 85 mm high)")

    # ── 9. hold the actuator internals equal across the pair ─────────────────
    # As for quadrupeds: vendor values are placeholders, not measurements.
    _match_actuator_internals(spec, model, twin,
                              mujoco.MjSpec.from_file(str(twin_mjcf)), rep)
    model = spec.compile()

    rep.total_mass_kg = float(model.body_mass.sum())

    # Copy meshes alongside and make paths relative (see localise_assets).
    normalise_geom_groups(spec, rep)
    localise_assets(spec, out_dir, rep)
    model = spec.compile()

    robot_xml = out_dir / "humanoid.xml"
    robot_xml.write_text(localised_xml(spec))
    rep.dump(out_dir / "rl_adapter_report.yaml")
    for tmp in (flat, Path(rerooted)):
        tmp.unlink(missing_ok=True)
    return robot_xml, rep
