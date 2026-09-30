"""2-D cross-section outlines and placing parts on their edges.

Callable from `${}` expressions (see `expr_eval._SAFE`). A mount attaches to an
outline edge and points along its outward normal, so parts follow a reshaped
outline. The polygon lies in the root's cross-section plane as (y, z); mounts
return x = 0 and a roll about x.
"""
from __future__ import annotations

import math


# ── Outline constructors ───────────────────────────────────────────────────────

def rect_outline(lu: float, lv: float, chamfer_factor: float = 0.15,
                 cu: float = 0.0, cv: float = 0.0) -> list:
    """A `lu` x `lv` rectangle as 8 [u, v] vertices, CCW from the +axis side.

    Corners are cut by `chamfer_factor` of the shorter side; `cu`/`cv` offset the centre.
    """
    hu, hv = lu / 2.0, lv / 2.0
    c = min(lu, lv) * chamfer_factor
    if c <= 0 or c >= hu or c >= hv:
        raise ValueError(f'chamfer {c:.5f} out of range for lu={lu}, lv={lv}')
    return [
        [cu - hu + c, cv - hv],      # 0 bottom edge, near -u corner
        [cu + hu - c, cv - hv],      # 1 bottom edge, near +u corner
        [cu + hu,     cv - hv + c],  # 2 right edge,  near -v corner
        [cu + hu,     cv + hv - c],  # 3 right edge,  near +v corner
        [cu + hu - c, cv + hv],      # 4 top edge,    near +u corner
        [cu - hu + c, cv + hv],      # 5 top edge,    near -u corner
        [cu - hu,     cv + hv - c],  # 6 left edge,   near +v corner
        [cu - hu,     cv - hv + c],  # 7 left edge,   near -v corner
    ]


def edge_outward_normal(polygon: list, edge: int) -> tuple[float, float]:
    """Outward unit normal (ny, nz) of edge `edge`, judged against the centroid (winding-independent)."""
    n = len(polygon)
    if n < 3:
        raise ValueError(f'a mounting polygon needs at least 3 vertices, got {n}')
    y0, z0 = polygon[edge % n]
    y1, z1 = polygon[(edge + 1) % n]
    dy, dz = y1 - y0, z1 - z0
    length = math.hypot(dy, dz)
    if length < 1e-12:
        return (0.0, 1.0)                      # degenerate edge: point "up"
    a = (-dz / length, dy / length)
    cy = sum(v[0] for v in polygon) / n
    cz = sum(v[1] for v in polygon) / n
    inward = (cy - (y0 + y1) / 2, cz - (z0 + z1) / 2)
    return a if a[0] * inward[0] + a[1] * inward[1] <= 0 else (-a[0], -a[1])


def edge_point(polygon: list, edge: int, t: float = 0.5) -> tuple[float, float]:
    """The point `t` of the way along edge `edge`, as (y, z)."""
    n = len(polygon)
    y0, z0 = polygon[edge % n]
    y1, z1 = polygon[(edge + 1) % n]
    return (y0 + t * (y1 - y0), z0 + t * (z1 - z0))


def _anchored(polygon, spec: dict) -> bool:
    return bool(polygon) and 'palm_edge' in spec


def mount_pos(polygon: list, spec: dict) -> list:
    """A part's body `pos`, from ``palm_edge`` (+ ``palm_edge_t``) or explicit ``palm_pos_y``/``palm_pos_z``."""
    if _anchored(polygon, spec):
        y, z = edge_point(polygon, int(spec['palm_edge']),
                          float(spec.get('palm_edge_t', 0.5)))
    else:
        y, z = float(spec.get('palm_pos_y', 0.0)), float(spec.get('palm_pos_z', 0.0))
    return [0.0, y, z]


def mount_euler(polygon: list, spec: dict) -> list:
    """A part's body `euler`: a roll about x taking local +z onto the edge's outward normal."""
    if _anchored(polygon, spec):
        ny, nz = edge_outward_normal(polygon, int(spec['palm_edge']))
        angle = math.atan2(-ny, nz)
    else:
        angle = float(spec.get('exit_angle', 0.0))
    return [angle, 0.0, 0.0]
