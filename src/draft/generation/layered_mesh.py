"""Watertight mesh lofted from parallel polygon cross-sections.

Every plane has the same vertex count; adjacent rings are joined by quad strips
and the end caps are earcut-triangulated (concave polygons are fine).

YAML schema
-----------
mesh:
  name: <str>
  axis: x | y | z       # axis perpendicular to every plane
  density: <float>       # kg/m³ — used for inertial property output
  planes:
    - position: <float>  # coordinate along axis
      vertices:          # 2-D polygon, [u, v] pairs
        - [u, v]
        ...              # same vertex count across ALL planes
    - position: <float>
      vertices:
        ...

Axis conventions (u, v in plane):
  axis x → planes in YZ:  u = Y, v = Z
  axis y → planes in XZ:  u = X, v = Z
  axis z → planes in XY:  u = X, v = Y

Vertices should be CCW viewed from +axis.
"""

import numpy as np
import trimesh
import yaml
from pathlib import Path

from .mounting import rect_outline


def inertial_properties(mesh: trimesh.Trimesh, rho: float) -> dict:
    """Compute mass, CoM, and inertia tensors for a watertight mesh with density rho."""
    if not mesh.is_watertight:
        raise ValueError("Mesh must be watertight to compute volume.")
    volume = mesh.volume
    mass = rho * volume
    com = mesh.center_mass
    # trimesh's moment_inertia is per unit density; scale to rho.
    inertia_com = mesh.moment_inertia * rho
    inertia_origin = (
        inertia_com
        + mass * (np.dot(com, com) * np.eye(3) - np.outer(com, com))
    )
    return {
        "volume": volume,
        "mass": mass,
        "center_of_mass": com,
        "inertia_com": inertia_com,
        "inertia_origin": inertia_origin,
    }


# ── Axis mapping ──────────────────────────────────────────────────────────────

# (u_col, v_col, axis_col) indices into the 3-D vertex array
_AXIS_MAP = {
    'x': (1, 2, 0),
    'y': (0, 2, 1),
    'z': (0, 1, 2),
}


# ── Triangulation helpers ─────────────────────────────────────────────────────

def _signed_area_2d(pts: np.ndarray) -> float:
    """Shoelace formula — positive = CCW, negative = CW."""
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _earcut(pts_2d: np.ndarray) -> np.ndarray:
    """Triangulate a simple 2-D polygon. Returns (M, 3) int array of indices."""
    import mapbox_earcut
    pts = np.asarray(pts_2d, dtype=np.float64)
    rings = np.array([len(pts)])
    try:
        tris = mapbox_earcut.triangulate_float64(pts, rings)
    except AttributeError:
        tris = mapbox_earcut.triangulate_float32(pts.astype(np.float32), rings)
    return tris.reshape(-1, 3).astype(np.int64)


# ── Convenience polygon builders ──────────────────────────────────────────────

#: A chamfered rectangle (defined in `mounting.py` so `${}` expressions can use it).
chamfered_rect_polygon = rect_outline


def inset_polygon(vertices: list, distance: float) -> list:
    """Offset a polygon inward by `distance`, keeping vertex count and order.

    Used for end-face chamfers (an inset layer at each end). Raises if a corner
    collapses, since the loft pairs vertices ring to ring.
    """
    from shapely.geometry import Polygon as _Polygon

    pts = np.asarray(vertices, dtype=float)
    if distance <= 0:
        return [[float(u), float(v)] for u, v in pts]
    poly = _Polygon(pts)
    if not poly.is_valid:
        poly = poly.buffer(0)
    inner = poly.buffer(-float(distance), join_style=2, mitre_limit=10.0)
    if inner.is_empty or not inner.is_valid:
        raise ValueError(
            f"inset of {distance:.4f} m collapses the outline; use a smaller "
            f"chamfer or a larger cross-section")
    ring = np.array(inner.exterior.coords[:-1])
    if len(ring) != len(pts):
        raise ValueError(
            f"inset of {distance:.4f} m left {len(ring)} vertices from {len(pts)}: "
            f"a corner collapsed. Reduce the inset or simplify the outline.")
    # Match the source's winding, or the loft twists into a self-intersecting band.
    if _signed_area_2d(ring) * _signed_area_2d(pts) < 0:
        ring = ring[::-1]
    # Align the start vertex so corresponding corners pair up.
    k = int(np.argmin(np.linalg.norm(ring - pts[0], axis=1)))
    ring = np.roll(ring, -k, axis=0)
    return [[float(u), float(v)] for u, v in ring]


# ── Core builder ──────────────────────────────────────────────────────────────

class LayeredMeshBuilder:
    """Build a watertight mesh from stacked parallel polygon cross-sections.

    Parameters
    ----------
    axis    : 'x', 'y', or 'z' — direction perpendicular to all planes
    planes  : list of {'position': float, 'vertices': [[u,v], ...]}
    density : kg/m³ for inertial property computation
    name    : base name used when saving output files
    """

    def __init__(
        self,
        axis: str = 'z',
        planes: list = None,
        density: float = 1000.0,
        name: str = 'mesh',
    ):
        self.axis = axis
        self.planes = sorted(planes or [], key=lambda p: p['position'])
        self.density = density
        self.name = name

    # ── Construction ─────────────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, yaml_path) -> 'LayeredMeshBuilder':
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)
        m = cfg['mesh']
        return cls(
            axis=m['axis'],
            planes=m['planes'],
            density=m.get('density', 1000.0),
            name=m.get('name', 'mesh'),
        )

    @classmethod
    def from_chamfered_rects(
        cls,
        axis: str,
        layers: list,          # list of (position, lx, ly, cx=0, cy=0)
        chamfer_factor: float,
        density: float = 1000.0,
        name: str = 'mesh',
    ) -> 'LayeredMeshBuilder':
        """Build from a stack of chamfered-rectangle cross-sections.

        Each entry in `layers` is a tuple:
          (position, lx, ly)           or
          (position, lx, ly, cx, cy)
        where cx/cy are optional center offsets in the plane (default 0).
        """
        planes = []
        for entry in layers:
            pos, lx, ly = entry[0], entry[1], entry[2]
            cx = entry[3] if len(entry) > 3 else 0.0
            cy = entry[4] if len(entry) > 4 else 0.0
            planes.append({
                'position': pos,
                'vertices': chamfered_rect_polygon(lx, ly, chamfer_factor, cx, cy),
            })
        return cls(axis=axis, planes=planes, density=density, name=name)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _to_3d(self, uv: np.ndarray, position: float) -> np.ndarray:
        """Map 2-D polygon vertices + axis position to (N, 3) 3-D coords."""
        u_col, v_col, ax_col = _AXIS_MAP[self.axis]
        n = len(uv)
        verts = np.zeros((n, 3), dtype=float)
        verts[:, u_col] = uv[:, 0]
        verts[:, v_col] = uv[:, 1]
        verts[:, ax_col] = position
        return verts

    def _quad_strip(self, base0: int, base1: int, n: int) -> list:
        """Two triangles per quad connecting ring base0 to ring base1."""
        faces = []
        for k in range(n):
            j = (k + 1) % n
            a, b = base0 + k, base0 + j
            c, d = base1 + k, base1 + j
            faces.append([a, b, d])
            faces.append([a, d, c])
        return faces

    # ── Public API ────────────────────────────────────────────────────────────

    def build(self) -> trimesh.Trimesh:
        """Assemble and return a watertight trimesh.Trimesh."""
        if len(self.planes) < 2:
            raise ValueError("At least 2 planes are required.")

        n = len(self.planes[0]['vertices'])
        for p in self.planes:
            if len(p['vertices']) != n:
                raise ValueError(
                    f"Plane at position {p['position']} has {len(p['vertices'])} "
                    f"vertices but expected {n}."
                )

        all_verts: list = []
        all_faces: list = []
        ring_starts: list = []

        for plane in self.planes:
            ring_starts.append(len(all_verts))
            uv = np.asarray(plane['vertices'], dtype=float)
            all_verts.extend(self._to_3d(uv, plane['position']).tolist())

        # Side faces — quad strips between adjacent rings
        for i in range(len(self.planes) - 1):
            all_faces.extend(self._quad_strip(ring_starts[i], ring_starts[i + 1], n))

        # End caps — earcut triangulation of first and last polygon
        bot_pts = np.asarray(self.planes[0]['vertices'], dtype=float)
        top_pts = np.asarray(self.planes[-1]['vertices'], dtype=float)

        bot_tris = _earcut(bot_pts)
        top_tris = _earcut(top_pts)

        # Bottom cap faces -axis: flip winding if polygon is CCW
        if _signed_area_2d(bot_pts) > 0:
            bot_tris = bot_tris[:, ::-1]
        # Top cap faces +axis: flip winding if polygon is CW
        if _signed_area_2d(top_pts) < 0:
            top_tris = top_tris[:, ::-1]

        rb, rt = ring_starts[0], ring_starts[-1]
        for tri in bot_tris:
            all_faces.append([rb + int(tri[0]), rb + int(tri[1]), rb + int(tri[2])])
        for tri in top_tris:
            all_faces.append([rt + int(tri[0]), rt + int(tri[1]), rt + int(tri[2])])

        mesh = trimesh.Trimesh(
            vertices=np.array(all_verts, dtype=float),
            faces=np.array(all_faces, dtype=int),
            process=True,
        )
        mesh.fix_normals()
        return mesh

    def build_and_save(
        self,
        output_dir,
        mesh_filename: str = None,
        props_filename: str = None,
    ) -> tuple:
        """Build, save STL and inertial-properties YAML; return (mesh, props)."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        mesh = self.build()
        props = inertial_properties(mesh, self.density)

        stl_path = output_dir / (mesh_filename or f'{self.name}_mesh.stl')
        yaml_path = output_dir / (props_filename or f'{self.name}_inertial_properties.yaml')

        mesh.export(str(stl_path))
        with open(yaml_path, 'w') as f:
            yaml.dump({
                'mass':  float(props['mass']),
                'com_x': float(props['center_of_mass'][0]),
                'com_y': float(props['center_of_mass'][1]),
                'com_z': float(props['center_of_mass'][2]),
                'I_xx':  float(props['inertia_com'][0, 0]),
                'I_yy':  float(props['inertia_com'][1, 1]),
                'I_zz':  float(props['inertia_com'][2, 2]),
                'I_xy':  float(props['inertia_com'][0, 1]),
                'I_xz':  float(props['inertia_com'][0, 2]),
                'I_yz':  float(props['inertia_com'][1, 2]),
            }, f)

        return mesh, props
