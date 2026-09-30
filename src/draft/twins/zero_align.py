"""Align a vendor model's rest pose and root frame with its twin's.

``q = 0`` is the zero of the action space, so both robots must stand the same
way there (G1's zero has elbows bent 90°; the tree's has arms hanging). Joint
zeros are compared positionally, by where downstream features land, since frame
orientations can agree while links point apart. Corrections are baked into body
frames (rotating a child about its own joint axis), so ``qpos0 == 0`` is the
aligned pose and link geometry, masses and ranges are unchanged.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

#: A probe counts only if it sits >= 2 cm off the joint axis and at least half
#: its length off-axis; joints with no such probe keep the vendor's zero.
_OBSERVABLE_M = 0.02
_OBSERVABLE_RATIO = 0.5


def signed_angle(v_from: np.ndarray, v_to: np.ndarray, axis: np.ndarray) -> float:
    """Rotation about ``axis`` carrying ``v_from`` onto ``v_to``, in radians.

    Both vectors are first projected perpendicular to the axis.
    """
    axis = np.asarray(axis, float)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    a = np.asarray(v_from, float) - float(np.dot(v_from, axis)) * axis
    b = np.asarray(v_to, float) - float(np.dot(v_to, axis)) * axis
    if np.linalg.norm(a) < 1e-9 or np.linalg.norm(b) < 1e-9:
        return 0.0
    return float(math.atan2(float(np.dot(axis, np.cross(a, b))), float(np.dot(a, b))))


def neutral_data(model: mujoco.MjModel) -> mujoco.MjData:
    """``q = 0`` with an upright, unrotated root — the pose the pair must share."""
    data = mujoco.MjData(model)
    data.qpos[:] = 0.0
    if model.njnt and model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE:
        data.qpos[3] = 1.0
    mujoco.mj_forward(model, data)
    return data


def root_frame(model: mujoco.MjModel, data: mujoco.MjData, root: str = "torso_link"):
    """(R, p) of the root body — every measurement below is taken in this frame."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root)
    if bid < 0:
        bid = 1
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, np.asarray(data.xquat[bid], float))
    return mat.reshape(3, 3), np.asarray(data.xpos[bid], float)


# ─────────────────────────────────────────────────────────────────────────────
# Probes: what each joint's zero is measured against
# ─────────────────────────────────────────────────────────────────────────────

def _actuated(model: mujoco.MjModel) -> list[tuple[int, str]]:
    return [(j, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j))
            for j in range(model.njnt)
            if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE
            and mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)]


def _descendants(model: mujoco.MjModel, body: int) -> list[int]:
    out, stack = [], [body]
    while stack:
        b = stack.pop()
        kids = [c for c in range(model.nbody) if int(model.body_parentid[c]) == b and c != b]
        out += kids
        stack += kids
    return out


def chain_probes(twin: mujoco.MjModel, site_ok=None) -> dict[str, tuple[str, ...]]:
    """Per joint, the downstream features its zero can be measured against.

    Candidates are downstream joint anchors, grouped and averaged by depth (so
    a joint carrying several chains is measured symmetrically), then the normal
    of the plane through the sites below it (a sole's direction, not position).
    ``site_ok`` filters sites; exclude any placed from the twin itself.
    """
    depths = joint_depths(twin)
    probes: dict[str, tuple[str, ...]] = {}
    for jid, name in _actuated(twin):
        body = int(twin.jnt_bodyid[jid])
        kids = _descendants(twin, body)
        by_depth: dict[int, list[str]] = {}
        for other, depth in depths.get(name, {}).items():
            by_depth.setdefault(depth, []).append(other)
        below = [f"joint:{','.join(sorted(by_depth[d]))}" for d in sorted(by_depth)]
        sites = [mujoco.mj_id2name(twin, mujoco.mjtObj.mjOBJ_SITE, s)
                 for s in range(twin.nsite) if int(twin.site_bodyid[s]) in ([body] + kids)]
        sites = [s for s in sites if s and (site_ok is None or site_ok(s))]
        probes[name] = tuple(below + ([f"plane:{','.join(sites)}"] if len(sites) >= 3 else []))
    return probes


def _probe_vector(model: mujoco.MjModel, data: mujoco.MjData, ref: str, R: np.ndarray,
                  anchor_w: np.ndarray) -> np.ndarray | None:
    """The probe as a vector from the joint, in the model's root frame.

    ``joint:`` gives the offset to the (mean) anchor; ``plane:`` the sites'
    best-fit normal, pointing away from the joint, scaled to 0.1 m.
    """
    kind, _, name = ref.partition(":")
    if kind == "joint":
        anchors = []
        for one in name.split(","):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, one)
            if jid < 0:
                return None
            anchors.append(np.asarray(data.xanchor[jid], float))
        return R.T @ (np.mean(anchors, axis=0) - anchor_w)
    pts = []
    for site in name.split(","):
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
        if sid < 0:
            return None                      # the pair must measure the same feature
        pts.append(np.asarray(data.site_xpos[sid], float))
    pts = np.asarray(pts, float)
    if len(pts) < 3:
        return None
    centre = pts.mean(axis=0)
    _u, _s, vt = np.linalg.svd(pts - centre)
    normal = vt[2]
    if float(np.dot(normal, centre - anchor_w)) < 0.0:
        normal = -normal
    return 0.1 * (R.T @ normal)


def joint_depths(model: mujoco.MjModel) -> dict[str, dict[str, int]]:
    """joint -> {joint below it: how many actuated joints deeper it is}.

    Probes are comparable only at equal depth on both robots (e.g. G1's wrist
    axis order differs from the tree's).
    """
    of_body: dict[int, list[tuple[int, str]]] = {}
    for jid, name in _actuated(model):
        of_body.setdefault(int(model.jnt_bodyid[jid]), []).append((jid, name))
    out: dict[str, dict[str, int]] = {}
    for jid, name in _actuated(model):
        below: dict[str, int] = {}
        frontier = [(int(model.jnt_bodyid[jid]), 0)]
        while frontier:
            body, depth = frontier.pop()
            for child in range(model.nbody):
                if int(model.body_parentid[child]) != body or child == body:
                    continue
                step = depth + (1 if child in of_body else 0)
                for _cj, cname in of_body.get(child, []):
                    below[cname] = step
                frontier.append((child, step))
        out[name] = below
    return out


def descendant_sites(model: mujoco.MjModel) -> dict[str, set[str]]:
    """joint -> the names of the sites anywhere below it."""
    out: dict[str, set[str]] = {}
    for jid, name in _actuated(model):
        bodies = {int(model.jnt_bodyid[jid])} | set(
            _descendants(model, int(model.jnt_bodyid[jid])))
        out[name] = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, s)
                     for s in range(model.nsite) if int(model.site_bodyid[s]) in bodies}
    return out


def _comparable(refs, joint: str, depth_t, depth_s, sites_t, sites_s) -> list[str]:
    """Keep only the probes that mean the same thing on both robots."""
    out = []
    for ref in refs:
        kind, _, name = ref.partition(":")
        if kind == "joint":
            d_t, d_s = depth_t.get(joint, {}), depth_s.get(joint, {})
            group = name.split(",")
            if all(n in d_t and n in d_s and d_t[n] == d_s[n] for n in group) and \
                    len({d_t[n] for n in group}) == 1:
                out.append(ref)
        else:
            wanted = set(name.split(","))
            if wanted <= sites_t.get(joint, set()) and wanted <= sites_s.get(joint, set()):
                out.append(ref)
    return out


def zero_pose_deltas(ship: mujoco.MjModel, twin: mujoco.MjModel,
                     probes: dict[str, tuple[str, ...]], root: str = "torso_link",
                     ) -> tuple[dict[str, float], dict[str, str]]:
    """Per joint: the rotation about its own axis from the twin's pose to the shipped's.

    Uses the comparable probe with the longest off-axis lever arm; joints with
    none are reported as unobservable and left at 0.
    """
    d_ship, d_twin = neutral_data(ship), neutral_data(twin)
    R_s, p_s = root_frame(ship, d_ship, root)
    R_t, p_t = root_frame(twin, d_twin, root)
    depth_t, depth_s = joint_depths(twin), joint_depths(ship)
    sites_t, sites_s = descendant_sites(twin), descendant_sites(ship)
    out: dict[str, float] = {}
    how: dict[str, str] = {}
    for jid_t, name in _actuated(twin):
        jid_s = mujoco.mj_name2id(ship, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid_s < 0:
            continue
        axis = R_t.T @ np.asarray(d_twin.xaxis[jid_t], float)
        best = None
        for ref in _comparable(probes.get(name, ()), name, depth_t, depth_s,
                               sites_t, sites_s):
            v_t = _probe_vector(twin, d_twin, ref, R_t,
                                np.asarray(d_twin.xanchor[jid_t], float))
            v_s = _probe_vector(ship, d_ship, ref, R_s,
                                np.asarray(d_ship.xanchor[jid_s], float))
            if v_t is None or v_s is None:
                continue
            perp = [float(np.linalg.norm(v - float(np.dot(v, axis)) * axis))
                    for v in (v_t, v_s)]
            span = [float(np.linalg.norm(v)) for v in (v_t, v_s)]
            if any(pp < max(_OBSERVABLE_M, _OBSERVABLE_RATIO * s)
                   for pp, s in zip(perp, span)):
                continue
            lever = min(perp)
            if best is None or lever > best[0]:
                best = (lever, ref, signed_angle(v_t, v_s, axis))
        if best is not None:
            out[name], how[name] = best[2], best[1]
        else:
            # No off-axis probe (e.g. a wrist roll about its forearm): keep vendor zero.
            out[name] = 0.0
            how[name] = "unobservable — vendor zero kept"
    return out, how


# ─────────────────────────────────────────────────────────────────────────────
# Baking a correction into a body frame
# ─────────────────────────────────────────────────────────────────────────────

def _axis_angle_quat(axis: np.ndarray, angle: float) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, np.asarray(axis, float), float(angle))
    return q


def bake_joint_rotation(spec: mujoco.MjSpec, joint_name: str, theta: float) -> None:
    """Rotate a joint's child body by ``theta`` about that joint's own axis.

    Only the labelling of configurations changes; the range shifts by
    ``-theta`` so it describes the same physical travel.
    """
    if abs(theta) < 1e-12:
        return
    jnt = spec.joint(joint_name)
    body = jnt.parent
    axis = np.asarray(jnt.axis, float)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)

    q_child = np.asarray(body.quat, float).copy()
    if getattr(body, "alt", None) is not None and \
            int(body.alt.type) != int(mujoco.mjtOrientation.mjORIENTATION_QUAT):
        raise ValueError(f"{body.name}: orientation is not declared as a quaternion "
                         f"({body.alt.type}); cannot bake a frame rotation safely")
    p_child = np.asarray(body.pos, float).copy()

    # Axis and pivot in the PARENT's frame.
    axis_p = np.zeros(3)
    mujoco.mju_rotVecQuat(axis_p, axis, q_child)
    anchor_p = np.zeros(3)
    mujoco.mju_rotVecQuat(anchor_p, np.asarray(jnt.pos, float), q_child)
    pivot = p_child + anchor_p

    rot = _axis_angle_quat(axis_p, theta)
    q_new = np.zeros(4)
    mujoco.mju_mulQuat(q_new, rot, q_child)
    moved = np.zeros(3)
    mujoco.mju_rotVecQuat(moved, p_child - pivot, rot)

    body.quat = q_new
    body.pos = pivot + moved
    jnt.range = np.array([float(jnt.range[0]) - theta, float(jnt.range[1]) - theta])


def normalize_joint_refs(spec: mujoco.MjSpec) -> dict[str, float]:
    """Fold any non-zero ``ref`` into the body frame, so ``q = 0`` IS the rest pose."""
    moved: dict[str, float] = {}
    for jnt in list(spec.joints):
        if jnt.type == mujoco.mjtJoint.mjJNT_FREE or not jnt.name:
            continue
        ref = float(jnt.ref)
        if abs(ref) < 1e-12:
            continue
        lo, hi = float(jnt.range[0]), float(jnt.range[1])
        bake_joint_rotation(spec, jnt.name, -ref)
        jnt.ref = 0.0
        jnt.range = np.array([lo, hi])   # bake shifted it; ref removal shifts it back
        moved[jnt.name] = math.degrees(ref)
    return moved


def align_rest_pose(spec: mujoco.MjSpec, twin: mujoco.MjModel,
                    probes: dict[str, tuple[str, ...]], passes: int = 12,
                    tol_deg: float = 0.05, root: str = "torso_link",
                    ) -> tuple[mujoco.MjModel, dict[str, float], dict[str, float], list[str]]:
    """Sweep root-to-tip, baking each joint, until ``q = 0`` is one pose on both.

    Re-measures after each joint, since a deep probe sees errors below it.
    Returns (compiled model, applied offsets in degrees, residual per joint in
    degrees, joints measured by the relative-rotation fallback).
    """
    applied: dict[str, float] = {}
    model = spec.compile()
    order = [n for _j, n in _actuated(twin)]     # MuJoCo orders parents first
    for _ in range(passes):
        deltas, how = zero_pose_deltas(model, twin, probes, root)
        if max((abs(d) for d in deltas.values()), default=0.0) <= math.radians(tol_deg):
            break
        for name in order:
            delta = deltas.get(name, 0.0)
            if abs(delta) <= math.radians(tol_deg):
                continue
            bake_joint_rotation(spec, name, -delta)
            applied[name] = applied.get(name, 0.0) - math.degrees(delta)
            model = spec.compile()
            deltas, how = zero_pose_deltas(model, twin, probes, root)
    deltas, how = zero_pose_deltas(model, twin, probes, root)
    residual = {k: round(math.degrees(v), 3) for k, v in deltas.items()}
    unobservable = sorted(k for k, v in how.items() if v.startswith("unobservable"))
    return model, {k: round(v, 3) for k, v in applied.items()}, residual, unobservable


# ─────────────────────────────────────────────────────────────────────────────
# Root frame
# ─────────────────────────────────────────────────────────────────────────────

def lowest_point(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    """World z of the lowest point of the collidable geometry, meshes exactly."""
    low = float("inf")
    for g in range(model.ngeom):
        if not (model.geom_contype[g] or model.geom_conaffinity[g]):
            continue
        if int(model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_MESH):
            mid = int(model.geom_dataid[g])
            adr, num = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
            verts = np.asarray(model.mesh_vert[adr:adr + num], float).reshape(-1, 3)
            rot = np.asarray(data.geom_xmat[g], float).reshape(3, 3)
            low = min(low, float(((verts @ rot.T) + data.geom_xpos[g])[:, 2].min()))
        elif int(model.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_PLANE):
            low = min(low, float(data.geom_xpos[g][2] - model.geom_rbound[g]))
    return 0.0 if not math.isfinite(low) else low


def align_root_origin(spec: mujoco.MjSpec, twin: mujoco.MjModel,
                      root: str = "torso_link", exclude: set | None = None) -> np.ndarray:
    """Slide the root body's ORIGIN to where the twin keeps its own.

    The tree's origin is on the shoulder line; Unitree's sits lower. The shift
    is the least-squares fit over shared bodies, and everything the root carries
    moves oppositely, so the machine is unchanged.

    Returns the shift applied, in the root's own frame.
    """
    model = spec.compile()
    data = neutral_data(model)
    d_twin = neutral_data(twin)
    R_s, p_s = root_frame(model, data, root)
    R_t, p_t = root_frame(twin, d_twin, root)
    deltas = []
    for b in range(1, twin.nbody):
        name = mujoco.mj_id2name(twin, mujoco.mjtObj.mjOBJ_BODY, b)
        other = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) if name else -1
        if other <= 0 or name == root or name in (exclude or set()):
            continue
        deltas.append(R_t.T @ (np.asarray(d_twin.xpos[b], float) - p_t)
                      - R_s.T @ (np.asarray(data.xpos[other], float) - p_s))
    if not deltas:
        return np.zeros(3)
    shift = -np.mean(deltas, axis=0)          # move the origin, not the robot

    body = spec.body(root)
    body.pos = np.asarray(body.pos, float) + shift
    body.ipos = np.asarray(body.ipos, float) - shift
    for child in body.bodies:
        child.pos = np.asarray(child.pos, float) - shift
    for item in list(body.geoms) + list(body.sites) + list(body.cameras) + \
            list(body.lights) + list(body.frames):
        item.pos = np.asarray(item.pos, float) - shift
    for jnt in body.joints:
        if jnt.type != mujoco.mjtJoint.mjJNT_FREE:
            jnt.pos = np.asarray(jnt.pos, float) - shift
    return shift


def normalize_root_frame(spec: mujoco.MjSpec, root: str = "torso_link") -> tuple[float, float]:
    """Face the model along +x and stand it on the floor at its rest pose.

    Removes any yaw declared on the base body; no relative geometry changes.
    Returns (yaw removed in degrees, rest height in metres).
    """
    body = spec.body(root)
    quat = np.asarray(body.quat, float).copy()
    yaw = math.degrees(2.0 * math.atan2(float(quat[3]), float(quat[0])))
    body.quat = np.array([1.0, 0.0, 0.0, 0.0])
    body.pos = np.array([0.0, 0.0, 0.0])
    model = spec.compile()
    data = neutral_data(model)
    height = -lowest_point(model, data)
    body.pos = np.array([0.0, 0.0, height])
    return round(((yaw + 180.0) % 360.0) - 180.0, 3), round(height, 5)
