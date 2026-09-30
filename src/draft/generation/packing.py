"""Actuator packing: do the drawn actuators fit without interpenetrating?

Each actuator is an oriented box (`L` along its axis, `2r` across) at the
neutral pose of the compiled model; overlap uses the 15-axis separating-axis
test. Conservative: boxes may overlap where cylinders would not, but a real
intersection is never missed. Also measures actuator volume inside the root mesh.
"""

from __future__ import annotations

import math

import numpy as np

_AXIS_INDEX = {'x': 0, 'y': 1, 'z': 2}

#: Absolute overlap tolerance (face-to-face actuators share a plane).
TOLERANCE_M = 5e-4

#: Relative tolerance: 2% of the smaller envelope is within the scatter of the
#: `motor_r`/`motor_L` trends, so it says nothing about buildability.
TOLERANCE_FRACTION = 0.02

#: Monte Carlo noise floor for `structure_overlap` (fraction of actuator volume).
STRUCTURE_NOISE = 0.002

#: Fraction of an actuator's volume inside the root mesh at which the structure gate trips.
STRUCTURE_TOL = 0.01


def _obb(motor: dict) -> tuple:
    """Centre, rotation (columns = box axes) and half-extents.

    Uses `R`/`half` when given, else a world-aligned box from the `axis` letter.
    """
    c = np.asarray(motor['center'], dtype=float)
    if motor.get('R') is not None and motor.get('half') is not None:
        return (c, np.asarray(motor['R'], dtype=float).reshape(3, 3),
                np.asarray(motor['half'], dtype=float))
    ax = _AXIS_INDEX.get(str(motor.get('axis', 'z')).lstrip('+-'), 0)
    half = np.array([motor['r'], motor['r'], motor['r']], dtype=float)
    half[ax] = motor['L'] / 2.0
    return c, np.eye(3), half


def _tolerance(ra: float, rb: float) -> float:
    """Tolerance on one candidate axis, relative to the extents projected on it."""
    return max(TOLERANCE_M, TOLERANCE_FRACTION * 2.0 * min(ra, rb))


def overlap(a: dict, b: dict) -> dict | None:
    """Penetration depth and axis between two actuators, or None when clear."""
    ca, Ra, ha = _obb(a)
    cb, Rb, hb = _obb(b)
    d = cb - ca

    # Face normals of each box, then edge cross-products (parallel edges skipped).
    axes, labels = [], []
    for i in range(3):
        axes.append(Ra[:, i]); labels.append(f'a{"xyz"[i]}')
        axes.append(Rb[:, i]); labels.append(f'b{"xyz"[i]}')
    for i in range(3):
        for j in range(3):
            v = np.cross(Ra[:, i], Rb[:, j])
            n = float(np.linalg.norm(v))
            if n > 1e-9:
                axes.append(v / n)
                labels.append(f'a{"xyz"[i]}xb{"xyz"[j]}')

    depth, axis, tol = None, None, TOLERANCE_M
    for v, label in zip(axes, labels):
        # Projected half-width of each box onto the candidate axis.
        ra = float(np.abs(Ra.T @ v) @ ha)
        rb = float(np.abs(Rb.T @ v) @ hb)
        gap = abs(float(d @ v)) - (ra + rb)
        axis_tol = _tolerance(ra, rb)
        if gap >= -axis_tol:
            return None                      # a separating axis exists
        if depth is None or -gap < depth:
            depth, axis, tol = -gap, label, axis_tol
    return {'a': a['name'], 'b': b['name'], 'depth_m': round(depth, 6),
            'axis': axis, 'tolerance_m': round(tol, 6),
            'a_class': a.get('motor_class'), 'b_class': b.get('motor_class'),
            'a_drives': a.get('drives'), 'b_drives': b.get('drives')}


def _consecutive(a: dict, b: dict, all_motors: list) -> bool:
    """Are these neighbours on one chain with no actuator between?

    Such a pair overlaps at every joint angle; pairs on different limbs are a
    neutral-pose artifact that joints can separate.
    """
    pa, pb = a.get('path') or (), b.get('path') or ()
    if len(pa) > len(pb):
        pa, pb = pb, pa
    if pb[:len(pa)] != pa:
        return False                          # different branches
    between = set(pb[len(pa):-1])
    return not any(m['body'] in between for m in all_motors)


def check(motor_geoms: list, exempt: set | None = None) -> dict:
    """Every interpenetrating actuator pair, split into hard (consecutive) and soft.

    `exempt` names geoms that depict an actuator counted elsewhere (e.g. the
    quadruped's knee pulley).
    """
    exempt = exempt or set()
    live = [m for m in motor_geoms if m['name'] not in exempt]
    by_name = {m['name']: m for m in live}
    hard, soft = [], []
    for i, a in enumerate(live):
        for b in live[i + 1:]:
            if a['body'] == b['body']:
                continue                     # one body draws at most one actuator
            hit = overlap(a, b)
            if not hit:
                continue
            (hard if _consecutive(a, b, live) else soft).append(hit)
    hard.sort(key=lambda p: -p['depth_m'])
    soft.sort(key=lambda p: -p['depth_m'])
    return {
        'n_actuator_geoms': len(live),
        'n_exempt': len(exempt),
        'n_overlapping_pairs': len(hard),
        'n_neutral_pose_pairs': len(soft),
        'worst_depth_m': hard[0]['depth_m'] if hard else 0.0,
        'worst_neutral_pose_depth_m': soft[0]['depth_m'] if soft else 0.0,
        'tolerance_m': TOLERANCE_M,
        'tolerance_fraction': TOLERANCE_FRACTION,
        'pairs': hard[:24],
        'neutral_pose_pairs': soft[:12],
    }


def messages(report: dict) -> list:
    """One line per un-buildable pair, in the form `FeasibilityViolation` prints."""
    out = []
    for p in report['pairs']:
        drives = ''
        if p.get('a_drives') and p.get('b_drives'):
            drives = f" ({p['a_drives']} → {p['b_drives']})"
        out.append(
            f"actuator packing: {p['a']} and {p['b']}{drives} interpenetrate by "
            f"{p['depth_m'] * 1e3:.1f} mm along {p['axis']} "
            f"(tolerance {p['tolerance_m'] * 1e3:.1f} mm). They are consecutive "
            f"on one chain, so no joint angle separates them: the link between "
            f"them is shorter than the two actuator half-envelopes it must hold "
            f"apart.")
    extra = report['n_overlapping_pairs'] - len(report['pairs'])
    if extra > 0:
        out.append(f"actuator packing: {extra} further overlapping pair(s) not listed.")
    return out


def soft_messages(report: dict) -> list:
    """Actuators that touch in the neutral pose but can be driven apart."""
    if not report['n_neutral_pose_pairs']:
        return []
    worst = report['neutral_pose_pairs'][0]
    return [
        f"actuator packing: {report['n_neutral_pose_pairs']} actuator pair(s) "
        f"interpenetrate in the NEUTRAL POSE only — worst {worst['a']} vs "
        f"{worst['b']} at {worst['depth_m'] * 1e3:.0f} mm. Joints separate them, "
        f"so the design is buildable, but that part of the workspace is not "
        f"reachable and contact will be generated there."]


def structure_overlap(model, data, rho: float, samples: int = 8000,
                      role_of=None) -> dict:
    """Volume of drawn cylinders inside the root mesh, and the mass charged twice.

    Monte Carlo over cylinders whose bounds meet the mesh. `structure_violations`
    turns the result into the gate.
    """
    import mujoco
    out = {'mesh_volume_cm3': None, 'inside_cm3': 0.0, 'double_charged_kg': 0.0,
           'actuators': []}
    if model is None or data is None:
        return out
    meshes = [g for g in range(model.ngeom)
              if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]
    if not meshes:
        return out
    try:
        import trimesh
    except ImportError:                                  # noqa: BLE001
        return out
    g = meshes[0]
    mid = model.geom_dataid[g]
    v0, nv = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
    f0, nf = model.mesh_faceadr[mid], model.mesh_facenum[mid]
    R = np.asarray(data.geom_xmat[g]).reshape(3, 3)
    verts = model.mesh_vert[v0:v0 + nv].reshape(-1, 3) @ R.T + data.geom_xpos[g]
    tm = trimesh.Trimesh(vertices=verts,
                         faces=model.mesh_face[f0:f0 + nf].reshape(-1, 3),
                         process=False)
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    out['mesh_volume_cm3'] = round(float(tm.volume) * 1e6, 1)
    rng = np.random.default_rng(0)
    total = 0.0
    for gi in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gi) or ''
        # Every drawn solid cylinder (motors, links, spheres); weightless `_detail` rings skipped.
        if (not name.endswith(('_motor', '_link', '_sphere'))
                or name.endswith('_detail')
                or model.geom_type[gi] != mujoco.mjtGeom.mjGEOM_CYLINDER):
            continue
        c = np.asarray(data.geom_xpos[gi], float)
        r, h = float(model.geom_size[gi][0]), float(model.geom_size[gi][1])
        reach = math.sqrt(r * r + h * h)
        if np.any(c + reach < lo) or np.any(c - reach > hi):
            continue
        Rg = np.asarray(data.geom_xmat[gi]).reshape(3, 3)
        z = rng.uniform(-h, h, samples)
        t = rng.uniform(0, 2 * math.pi, samples)
        rr = r * np.sqrt(rng.uniform(0, 1, samples))
        pts = np.stack([rr * np.cos(t), rr * np.sin(t), z], 1) @ Rg.T + c
        frac = float(tm.contains(pts).mean())
        if frac <= STRUCTURE_NOISE:
            continue
        vol = math.pi * r * r * 2 * h * frac
        total += vol
        out['actuators'].append({'geom': name, 'inside_frac': round(frac, 4),
                                 'drives': (role_of or {}).get(name),
                                 'inside_cm3': round(vol * 1e6, 1)})
    out['inside_cm3'] = round(total * 1e6, 1)
    out['double_charged_kg'] = round(total * float(rho), 4)
    return out


def structure_violations(report: dict, allowed, tol: float = STRUCTURE_TOL) -> list:
    """Solids inside the root mesh beyond `tol` whose role or geom name is not in `allowed`.

    `allowed` is the design's `structure_overlap_allowed`.
    """
    out = []
    ok = set(allowed or ())
    for a in report.get('actuators') or ():
        # Match by role, or by geom name for parts from an attached `module:` MJCF.
        if (float(a.get('inside_frac') or 0.0) <= tol
                or a.get('drives') in ok or a.get('geom') in ok):
            continue
        out.append(
            f"{a['geom']} is drawn {100 * a['inside_frac']:.0f}% inside the root "
            f"mesh ({a['inside_cm3']:.0f} cm3), which no design declared: it "
            f"drives {a.get('drives') or 'an undeclared role'}, and only "
            f"{', '.join(sorted(ok)) or '(nothing)'} may live in the structure. "
            f"That volume holds two solids and is charged at both their "
            f"densities.")
    return out


def structure_messages(report: dict, trunk_mass: float,
                       frac: float = 0.05) -> list:
    """Warn when the trunk is paying for a meaningful volume of actuator."""
    kg = float(report.get('double_charged_kg') or 0.0)
    if trunk_mass <= 0 or kg <= frac * trunk_mass:
        return []
    worst = max(report.get('actuators') or [], key=lambda a: a['inside_cm3'],
                default=None)
    return [f"{report['inside_cm3']:.0f} cm3 of actuator is drawn inside the root "
            f"mesh, so {kg:.2f} kg is charged both at the trunk's density and at "
            f"the motor trend's — {kg / trunk_mass * 100:.0f}% of the trunk"
            + (f", worst {worst['geom']} at {worst['inside_frac'] * 100:.0f}% "
               f"buried" if worst else "") + "."]


def required_separation(a: dict, b: dict, axis: str) -> float:
    """Centre-to-centre distance along world `axis` that would just clear the pair."""
    v = np.zeros(3)
    v[_AXIS_INDEX[axis]] = 1.0
    _, Ra, ha = _obb(a)
    _, Rb, hb = _obb(b)
    return (float(np.abs(Ra.T @ v) @ ha) + float(np.abs(Rb.T @ v) @ hb)
            + TOLERANCE_M)
