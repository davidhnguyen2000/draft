"""Format-independent tree operations: mirroring, motor classes, link sizing, taper."""
from __future__ import annotations

import copy
import math
from typing import Any


# ── Axis helpers ───────────────────────────────────────────────────────────────

_AXIS_VEC: dict[str, tuple] = {
    'x':  (1, 0, 0), '+x': (1, 0, 0), '-x': (-1, 0, 0),
    'y':  (0, 1, 0), '+y': (0, 1, 0), '-y': (0, -1, 0),
    'z':  (0, 0, 1), '+z': (0, 0, 1), '-z': (0, 0, -1),
}


def avec(s: str) -> tuple:
    return _AXIS_VEC[s.strip()]


def euler_for_axis(s: str) -> list | None:
    """Euler angles (rx, ry, 0) that point a Z-default cylinder along axis `s`."""
    base = s.strip().lstrip('+-')
    if base == 'x':
        return [0.0, math.pi / 2, 0.0]
    if base == 'y':
        return [math.pi / 2, 0.0, 0.0]
    return None                      # z-axis: identity


def axis_center(axis: str, half_len: float) -> tuple:
    return tuple(v * half_len for v in avec(axis))


def rgba(color: Any) -> list:
    """Colour as a 4-list, from either a list or a whitespace-separated string."""
    if isinstance(color, (list, tuple)):
        vals = [float(x) for x in color]
    else:
        vals = [float(x) for x in str(color).split()]
    while len(vals) < 4:
        vals.append(1.0)
    return vals[:4]


# ── Mirror expansion ───────────────────────────────────────────────────────────

_PREFIX_PAIRS = [
    ('left_', 'right_'), ('right_', 'left_'),
    ('fl_', 'fr_'),  ('fr_', 'fl_'),
    ('rl_', 'rr_'),  ('rr_', 'rl_'),
    ('bl_', 'br_'),  ('br_', 'bl_'),
]


def _mirror_name(name: str) -> str:
    for a, b in _PREFIX_PAIRS:
        if name.startswith(a):
            return b + name[len(a):]
    return name


def _mirror_axis_str(axis_str: str, mirror: str) -> str:
    """Negate the component of `axis_str` that lies along the mirror axis."""
    av = list(avec(axis_str))
    mv = avec(mirror)
    for i in range(3):
        if mv[i] != 0:
            av[i] = -av[i]
    for i, val in enumerate(av):
        if val != 0:
            return ('' if val > 0 else '-') + 'xyz'[i]
    return axis_str


def _mirror_pos(pos: list, mirror: str) -> list:
    mv = avec(mirror)
    return [(-p if mv[i] != 0 else p) for i, p in enumerate(pos)]


def _mirror_node(node: dict, axis: str) -> None:
    """Mirror a node in place, recursing into children."""
    if 'name' in node:
        node['name'] = _mirror_name(node['name'])
    if 'prefix' in node:
        node['prefix'] = _mirror_name(str(node['prefix']))
    if isinstance(node.get('pos'), list):
        node['pos'] = _mirror_pos(node['pos'], axis)
    if isinstance(node.get('joint'), dict):
        j = node['joint']
        if 'name' in j:
            j['name'] = _mirror_name(j['name'])
        # Joint axis/range are not mirrored: +θ is the same motion on both sides.
    if isinstance(node.get('geometry'), dict):
        g = node['geometry']
        for k in ('link_axis', 'motor_axis'):
            if k in g:
                g[k] = _mirror_axis_str(str(g[k]), axis)
        if isinstance(g.get('motor_shift'), list):
            g['motor_shift'] = _mirror_pos(g['motor_shift'], axis)
    for site in node.get('sites', []):
        if 'name' in site:
            site['name'] = _mirror_name(site['name'])
        if isinstance(site.get('pos'), list):
            site['pos'] = _mirror_pos(site['pos'], axis)
    for child in node.get('children', []):
        _mirror_node(child, axis)


def expand_mirrors(node: dict) -> dict:
    """Replace every `mirror_of` entry with a mirrored copy of what it names."""
    if 'children' not in node:
        return node
    expanded: list[dict] = []
    registry: dict[str, dict] = {}
    for child in node['children']:
        if 'mirror_of' in child:
            orig = registry[child['mirror_of']]
            new = copy.deepcopy(orig)
            _mirror_node(new, child['mirror_axis'])
            expanded.append(new)
            registry[new['name']] = new
        else:
            child = expand_mirrors(child)
            expanded.append(child)
            registry[child['name']] = child
    return {**node, 'children': expanded}


# ── Motor class resolution ─────────────────────────────────────────────────────

def motor_props(motor_class: str, params: dict, overrides: dict | None = None) -> dict:
    """Resolve one motor size class (L / M / S …) into its emitted properties."""
    cls = motor_class.strip()
    props = {
        'r':        float(params[f'{cls}_motor_r']),
        'L':        float(params[f'{cls}_motor_L']),
        # Per-class density overrides the global `motor_rho`.
        'rho':      float(params.get(f'{cls}_motor_rho', params.get('motor_rho', 2700.0))),
        'color':    params.get('motor_color', '0.2 0.2 0.2 1'),
        'detail':   params.get('mot_detail_color', '0.1 0.1 0.1 1'),
        'effort':   float(params[f'{cls}_motor_effort']),
        'velocity': float(params[f'{cls}_motor_velocity']),
        'friction': float(params[f'{cls}_motor_friction']),
        'damping':  float(params[f'{cls}_motor_damping']),
        'armature': float(params[f'{cls}_motor_armature']),
    }
    if overrides:
        props.update({k: float(v) for k, v in overrides.items()})
    return props


# ── Structural vs drawn link radius ────────────────────────────────────────────
#
# The STRUCTURAL radius carries the mass (solved from the λ trend for a
# `link_mass_class` member, else stated). The DRAWN radius may be slimmer; the
# emitted density is scaled by (r_struct/r_draw)² so mass is unchanged.

def drawn_radius(g: dict, params: dict, r_struct: float, mp: dict | None,
                 link_axis: str = 'z', motor_axis: str | None = None,
                 parent_mp: dict | None = None,
                 parent_axis: str | None = None) -> float:
    """Radius the link cylinder is drawn (and collides) at.

    Bounded by the face of each end actuator: `r` if coaxial with the link,
    min(r, L/2) if broadside, times `link_render_radius_scale`. `link_render_r`
    overrides. Mass is preserved via `slim_density`.
    """
    if 'link_render_r' in g:
        return float(g['link_render_r'])
    scale = float(params.get('link_render_radius_scale') or 1.0)
    base = link_axis.strip().lstrip('+-')
    bounds = [r_struct]
    for m, axis in ((mp, motor_axis), (parent_mp, parent_axis)):
        if not m:
            continue
        bounds.append(scale * float(m['r']))
        if axis is None or str(axis).strip().lstrip('+-') != base:
            bounds.append(scale * float(m['L']) / 2.0)
    return min(bounds)


def structural_radius_cap(mp: dict | None, motor_axis: str | None,
                          parent_mp: dict | None, parent_axis: str | None,
                          link_axis: str = 'z') -> float:
    """Widest a structural member may be: `drawn_radius`'s face rule without the scale (inf if no actuator)."""
    base = link_axis.strip().lstrip('+-')
    bounds = []
    for m, axis in ((mp, motor_axis), (parent_mp, parent_axis)):
        if not m:
            continue
        bounds.append(float(m['r']))
        if axis is None or str(axis).strip().lstrip('+-') != base:
            bounds.append(float(m['L']) / 2.0)
    return min(bounds) if bounds else math.inf


#: How far a structural link runs into its actuator's envelope (a flange, not to the joint centre).
LINK_MOTOR_PENETRATION_M = 0.005


def motor_axial_halfextent(mp: dict, link_axis: str, motor_axis: str) -> float:
    """Actuator reach along the link axis: L/2 if coaxial, else r (sign-agnostic)."""
    coaxial = link_axis.lstrip('+-') == motor_axis.lstrip('+-')
    return float(mp['L']) / 2.0 if coaxial else float(mp['r'])


def motor_clears_link(shift, link_axis: str, r_link: float, r_motor: float) -> bool:
    """True when `motor_shift` moves the actuator clear of the link cylinder."""
    if not shift:
        return False
    idx = 'xyz'.index(link_axis.lstrip('+-'))
    perp = math.hypot(*[float(v) for i, v in enumerate(shift) if i != idx])
    return perp >= r_link + r_motor


def proximal_inset(parent_mp: dict | None, parent_axis: str, offset_along: float,
                   link_axis: str,
                   penetration: float = LINK_MOTOR_PENETRATION_M) -> float:
    """How far the link's proximal end is buried in the parent's actuator.

    That actuator drives this body's joint, so the overlap is angle-invariant
    and is trimmed to avoid charging the volume twice. `offset_along` is the
    parent actuator's position along this link axis (non-zero under `motor_shift`).
    """
    if parent_mp is None:
        return 0.0
    h = motor_axial_halfextent(parent_mp, link_axis, parent_axis)
    return max(0.0, offset_along + h - penetration)


def link_span(link_l: float, mot_offset: float, mp: dict | None,
              link_axis: str, motor_axis: str, shift,
              r_link: float, penetration: float = LINK_MOTOR_PENETRATION_M,
              min_frac: float = 0.05, t0: float = 0.0) -> tuple[float, float]:
    """(t0, t1) along the link axis occupied by the structural cylinder.

    The distal end stops at the actuator's housing face so its volume is not
    charged twice; `t0` is the proximal inset (`proximal_inset`). `min_frac`
    keeps a stub of the link length.
    """
    if link_l <= 0:
        return 0.0, max(link_l, 0.0)
    if mp is None or motor_clears_link(shift, link_axis, r_link, float(mp['r'])):
        t1 = link_l
    else:
        h = motor_axial_halfextent(mp, link_axis, motor_axis)
        t1 = min(link_l, mot_offset - h + penetration)
    # Keep a stub; the proximal inset yields if the two would cross.
    t0 = min(max(t0, 0.0), max(link_l - link_l * min_frac, 0.0))
    return t0, max(t1, t0 + link_l * min_frac)


def slim_density(link_rho: float, r_struct: float, r_draw: float) -> float:
    """Density that makes a slimmer cylinder weigh what the structural one did."""
    if r_draw <= 0 or r_struct <= 0:
        return link_rho
    return link_rho * (r_struct / r_draw) ** 2


# ── Distal taper ───────────────────────────────────────────────────────────────
#
# The within-robot λ gradient (`distal_taper`, fit in datasets/robot_descriptions)
# sets each chain's shape; `link_rho` keeps the level. λ = ρ·π·r², so
# `taper_mode: density` scales ρ and `radius` scales r.

def _taper_nodes(node: dict, out: dict) -> None:
    """Collect geometry blocks tagged with a chain and a depth, by (chain, depth)."""
    g = node.get('geometry') or {}
    if 'taper' in g and 'taper_depth' in g:
        out.setdefault((str(g['taper']), int(g['taper_depth'])), []).append(node)
    for c in node.get('children', []) or []:
        _taper_nodes(c, out)


def measured_taper(tree: dict, params: dict) -> dict:
    """Per-step λ gradient per tagged chain, as emitted (read-only; runs without `taper_mode`)."""
    by_slot: dict = {}
    _taper_nodes(tree, by_slot)
    base_rho = float(params.get('link_rho', 500.0))
    out = {}
    for chain in {c for c, _ in by_slot}:
        lam = {}
        for c, d in by_slot:
            if c != chain:
                continue
            g = by_slot[(c, d)][0]['geometry']
            r = float(g.get('link_r', 0.0))
            if r > 0:
                lam[d] = float(g.get('link_rho', base_rho)) * math.pi * r ** 2
        ds = sorted(lam)
        if len(ds) >= 2:
            out[chain] = (lam[ds[-1]] / lam[ds[0]]) ** (1.0 / (len(ds) - 1))
    return out


def apply_distal_taper(tree: dict, params: dict, targets: dict) -> dict | None:
    """Set each tagged chain's λ gradient to `targets` (chain → per-step factor).

    `taper_mode` ``density`` rewrites `link_rho`, ``radius`` rewrites `link_r`.
    Clamps that bind are reported as warnings. Returns a report, or None if off.
    """
    mode = params.get('taper_mode')
    if not mode or not targets:
        return None
    if mode not in ('density', 'radius'):
        raise ValueError(f"taper_mode must be 'density' or 'radius', not {mode!r}")
    by_slot: dict = {}
    _taper_nodes(tree, by_slot)
    if not by_slot:
        return None

    base_rho = float(params.get('link_rho', 500.0))
    rho_lo, rho_hi = params.get('taper_rho_band') or (0.0, float('inf'))
    report: dict = {'mode': mode, 'chains': {}, 'warnings': []}

    for chain, target in targets.items():
        slots = sorted(d for c, d in by_slot if c == chain)
        if len(slots) < 2:
            continue
        # One λ per depth: the mirrored copies of a link are the same link.
        lam, rho0, r0 = {}, {}, {}
        for d in slots:
            nodes = by_slot[(chain, d)]
            rr = [float((n['geometry']).get('link_r', 0.0)) for n in nodes]
            hh = [float((n['geometry']).get('link_rho', base_rho)) for n in nodes]
            if min(rr) <= 0:
                continue
            r0[d] = math.exp(sum(math.log(x) for x in rr) / len(rr))
            rho0[d] = math.exp(sum(math.log(x) for x in hh) / len(hh))
            lam[d] = rho0[d] * math.pi * r0[d] ** 2
        if len(lam) < 2:
            continue

        # Where the chain sits now, and where the trend says it should sit.
        want = {d: target ** (d - slots[0]) for d in lam}
        gm = lambda v: math.exp(sum(math.log(x) for x in v) / len(v))  # noqa: E731
        gm_lam, gm_want = gm(lam.values()), gm(want.values())
        corr = {d: (want[d] / gm_want) / (lam[d] / gm_lam) for d in lam}
        # Rescale so the chain's total structural mass is unchanged (the taper sets shape, not level).
        mass = {d: sum(float((n['geometry']).get('link_l', 0.0))
                       for n in by_slot[(chain, d)]) * lam[d] for d in lam}
        moved = sum(mass[d] * corr[d] for d in lam)
        if moved > 0:
            keep = sum(mass.values()) / moved
            corr = {d: c * keep for d, c in corr.items()}

        applied = {}
        for d in sorted(lam):
            c = corr[d]
            for n in by_slot[(chain, d)]:
                g = n['geometry']
                if mode == 'density':
                    rho = float(g.get('link_rho', base_rho)) * c
                    clamped = min(max(rho, rho_lo), rho_hi)
                    if abs(clamped - rho) > 1e-9:
                        report['warnings'].append(
                            f"{chain} depth {d}: taper wants link_rho "
                            f"{rho:.0f} kg/m³, clamped to {clamped:.0f} by the "
                            f"measured structural band — this link does not "
                            f"reach the population taper.")
                    g['link_rho'] = clamped
                else:
                    r = float(g['link_r']) * math.sqrt(c)
                    cap = g.get('taper_r_max')
                    if cap and r > float(cap):
                        report['warnings'].append(
                            f"{chain} depth {d}: taper wants link_r {r * 1e3:.1f} mm, "
                            f"capped at {float(cap) * 1e3:.1f} mm (a structural "
                            f"member wider than the actuator it houses) — this "
                            f"link does not reach the population taper.")
                        r = float(cap)
                    g['link_r'] = r
            # what the slot actually ended up at, after any clamp
            n0 = by_slot[(chain, d)][0]['geometry']
            applied[d] = (float(n0.get('link_rho', base_rho)) * math.pi
                          * float(n0['link_r']) ** 2)

        ds = sorted(applied)
        steps = len(ds) - 1
        report['chains'][chain] = {
            'target_per_step': round(float(target), 4),
            'before_per_step': round((lam[ds[-1]] / lam[ds[0]]) ** (1.0 / steps), 4),
            'after_per_step': round((applied[ds[-1]] / applied[ds[0]]) ** (1.0 / steps), 4),
            'lambda_before_kg_per_m': {d: round(lam[d], 4) for d in ds},
            'lambda_after_kg_per_m': {d: round(applied[d], 4) for d in ds},
            'link_rho_kg_m3': {d: round(float(by_slot[(chain, d)][0]['geometry']
                                              .get('link_rho', base_rho)), 1) for d in ds},
            'link_r_mm': {d: round(float(by_slot[(chain, d)][0]['geometry']['link_r'])
                                   * 1e3, 2) for d in ds},
        }
    return report or None


# ── Geometric volume ───────────────────────────────────────────────────────────

def geom_volume(gtype: str, size: list, fromto: list | None = None,
                mesh_volume_m3: float | None = None) -> float:
    """Volume (m³) of a geom, from the same size fields MuJoCo reads."""
    if gtype == 'sphere':
        return 4.0 / 3.0 * math.pi * size[0] ** 3
    if gtype in ('cylinder', 'capsule'):
        r = size[0]
        half = (0.5 * math.dist(fromto[:3], fromto[3:])) if fromto else size[1]
        vol = math.pi * r * r * 2 * half
        if gtype == 'capsule':                    # two hemispherical caps
            vol += 4.0 / 3.0 * math.pi * r ** 3
        return vol
    if gtype == 'box':
        return 8.0 * size[0] * size[1] * size[2]
    if gtype == 'mesh':
        return mesh_volume_m3 or 0.0
    return 0.0
