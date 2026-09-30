"""Side-by-side MuJoCo renders of a shipped robot and its generated twin.

Both robots stand in one scene (one floor, light and camera), separated along
the camera's right axis so they appear at the same scale. The camera is fitted
to the pair in true perspective (`_frame_pair`). Quadrupeds are posed by joint
role (`STANDING_POSE`); humanoids keep their zero pose, with the twin's elbows
bent to match (`TWIN_POSE`). Each half is dropped onto the floor by its lowest
point. The shipped humanoid's hand is redrawn as an equal-volume sphere
(`_blob_hands`), matching the twin's.

Writes `twin_render_<key>.png` (titled and labelled) and
`twin_render_<key>_bare.png` (plate only). Targets described by a URDF are
converted first by `urdf_for_mujoco()`.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .targets import GENERATED_HUMANOID_ROLES, GENERATED_QUADRUPED_ROLES

#: name of the floor this module adds, so the bounds can exclude it
FLOOR = "twin_render_floor"

#: Name prefix for the twin's elements when the pair is attached into one scene;
#: it is how the two halves are told apart.
PAIR_PREFIX = "twin/"

#: Frame slack around the pair's projected bounding box (1.00 clips).
FRAME_MARGIN = 1.06

#: Gap between the two robots, as a fraction of their mean on-screen width.
PAIR_GAP = 0.10

#: Vertical field of view (deg). A long lens, so both robots are seen from nearly
#: the same bearing; `_frame_pair` sets the distance from it.
FOVY = 5.0

#: Max size (m) of a geom on an equality anchor that is treated as a marker, not structure.
MARKER_SIZE = 0.02

#: Posture for both halves, by joint role (rad). Quadruped zero poses are not
#: comparable; humanoids keep their own zero (see `TWIN_POSE`).
STANDING_POSE = {
    "quadruped": {"hip_roll": 0.0, "hip_pitch": 0.8, "knee": -1.6},
    "humanoid": {},
}

#: Extra pose for the generated half only: bends the twin's elbows to match the
#: flexion G1 and H2 carry at their own zero. Cosmetic only.
TWIN_POSE = {
    "humanoid": {"elbow": -np.pi / 2},
    "quadruped": {},
}

#: Vendor meshes misplaced by their MJCF conversion (B2's depth-camera shells);
#: massless and non-colliding, so dropping them changes no measurement.
STRAY_MESH_RE = re.compile(r"_dc_link$", re.I)

#: Hand mesh names on a shipped humanoid (matched by mesh, not body).
HAND_MESH_RE = re.compile(r"hand", re.I)


# ── COLLADA: the colours a URDF names but never defines ──────────────────────

def _dae_to_obj(dae: Path, out_dir: Path) -> list[tuple[str, tuple]] | None:
    """Convert one COLLADA file to one OBJ per material, keeping the colours.

    Returns `(obj stem, rgba)` per part, largest first; a merged `<stem>.obj`
    is also written. Cached across runs.
    """
    import trimesh
    merged = out_dir / f"{dae.stem}.obj"
    manifest = out_dir / f"{dae.stem}.parts"
    if merged.exists() and manifest.exists():           # cached from a past run
        out = []
        for line in manifest.read_text().splitlines():
            stem, _, rgba = line.partition(" ")
            out.append((stem, tuple(float(v) for v in rgba.split())))
        return out
    try:
        scene = trimesh.load(dae)
        trimesh.load(dae, force="mesh").export(merged)
        pieces = scene.dump() if hasattr(scene, "dump") else []
    except Exception as exc:                            # noqa: BLE001
        print(f"    ! mesh {dae.name}: {exc}")
        return None

    out = []
    pieces = sorted((p for p in pieces if len(p.faces)),
                    key=lambda m: -len(m.faces))
    for i, piece in enumerate(pieces if len(pieces) > 1 else []):
        stem = f"{dae.stem}__{i}"
        try:
            piece.export(out_dir / f"{stem}.obj")
        except Exception as exc:                        # noqa: BLE001
            print(f"    ! mesh {dae.name} part {i}: {exc}")
            return None
        out.append((stem, _mesh_rgba(piece)))
    if not out:
        out = [(dae.stem,
                _mesh_rgba(pieces[0]) if pieces else (0.7, 0.7, 0.7, 1.0))]
    manifest.write_text("".join(
        f"{s} {' '.join(f'{v:.4f}' for v in c)}\n" for s, c in out))
    return out


def _mesh_rgba(mesh) -> tuple:
    """A COLLADA sub-mesh's colour as URDF rgba (channels floored at 0.15 so black shades)."""
    for probe in ("baseColorFactor", "diffuse"):
        col = getattr(getattr(mesh.visual, "material", None), probe, None)
        if col is None:
            continue
        v = [float(c) for c in col][:4]
        if max(v) > 1.001:              # 0-255 ints
            v = [c / 255.0 for c in v]
        v = [max(c, 0.15) for c in v[:3]] + [v[3] if len(v) > 3 else 1.0]
        return tuple(round(c, 4) for c in v)
    return (0.7, 0.7, 0.7, 1.0)


def _paint_visuals(root: ET.Element, parts: dict) -> None:
    """Expand every `<visual>` whose mesh was split into one visual per part.

    Colours are written inline under unique names; MuJoCo's URDF reader ignores
    `<robot>`-level materials.
    """
    seq = iter(range(1 << 20))

    def paint(vis: ET.Element, rgba: tuple) -> None:
        for old in vis.findall("material"):
            vis.remove(old)
        mat = ET.SubElement(vis, "material", {"name": f"dae_{next(seq)}"})
        ET.SubElement(mat, "color",
                      {"rgba": " ".join(f"{c:.4f}" for c in rgba)})

    for link in root.iter("link"):
        for vis in list(link.findall("visual")):
            mesh = vis.find("geometry/mesh")
            if mesh is None:
                continue
            got = parts.get(Path(mesh.get("filename", "")).stem)
            if not got:
                continue
            at = list(link).index(vis)
            link.remove(vis)
            for j, (stem, rgba) in enumerate(got):
                piece = ET.fromstring(ET.tostring(vis))
                piece.find("geometry/mesh").set("filename",
                                                f"meshes/{stem}.obj")
                paint(piece, rgba)
                link.insert(at + j, piece)


# ── URDF → MuJoCo-loadable URDF ─────────────────────────────────────────────

def urdf_for_mujoco(urdf: Path, out_dir: Path) -> Path | None:
    """Write a copy of a URDF that MuJoCo can compile with its visual meshes.

    Sets `discardvisual="false"`, converts COLLADA meshes to OBJ, and repairs
    repeated `<material>` elements and degenerate inertials. Render-only: all
    measurements come from the original URDF.
    """
    urdf = Path(urdf)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tree = ET.parse(urdf)
    root = tree.getroot()

    src_meshes = next((c for c in (urdf.parent.parent / "meshes",
                                   urdf.parent / "meshes") if c.is_dir()), None)
    if any(Path(m.get("filename", "")).suffix.lower() == ".dae"
           for m in root.iter("mesh")) and src_meshes is not None:
        meshes = out_dir / "meshes"
        meshes.mkdir(parents=True, exist_ok=True)
        parts: dict[str, list[tuple[str, tuple]]] = {}
        for dae in sorted(src_meshes.glob("*.dae")):
            got = _dae_to_obj(dae, meshes)
            if got is None:
                return None
            parts[dae.stem] = got
        for mesh in root.iter("mesh"):
            stem = Path(mesh.get("filename", "")).stem
            if stem:
                mesh.set("filename", f"meshes/{stem}.obj")
        _paint_visuals(root, parts)
    else:
        # The rewritten file lives elsewhere, so make mesh paths absolute.
        base = urdf.parent
        for mesh in root.iter("mesh"):
            fn = mesh.get("filename", "")
            if not fn or Path(fn).is_absolute():
                continue
            rel = fn.split("://", 1)[-1] if "://" in fn else fn
            for cand in (base / rel, base.parent / rel,
                         base.parent.parent / rel):
                if cand.exists():
                    mesh.set("filename", str(cand.resolve()))
                    break

    # MuJoCo rejects repeated `<material>`s: keep the first per element, and
    # reduce re-declared names to name-only references.
    seen: set = set()
    for parent in root.iter():
        mats = parent.findall("material")
        for extra in mats[1:]:
            parent.remove(extra)
        for mat in mats[:1]:
            name = mat.get("name")
            if not name:
                continue
            if name in seen:
                for child in list(mat):
                    mat.remove(child)
            else:
                seen.add(name)
    # Floor near-massless placeholder links to 0.1 g so MuJoCo accepts them.
    for inertial in root.iter("inertial"):
        mass = inertial.find("mass")
        if mass is None or float(mass.get("value", 0)) >= 1e-4:
            continue
        mass.set("value", "1e-4")
        inertia = inertial.find("inertia")
        if inertia is not None:
            for k in ("ixx", "iyy", "izz"):
                inertia.set(k, "1e-6")
            for k in ("ixy", "ixz", "iyz"):
                inertia.set(k, "0")

    # Replace any existing <mujoco> element (MuJoCo rejects two).
    for existing in root.findall("mujoco"):
        root.remove(existing)
    mj = ET.SubElement(root, "mujoco")
    ET.SubElement(mj, "compiler", {"balanceinertia": "true",
                                   "discardvisual": "false",
                                   "fusestatic": "false",
                                   "strippath": "false"})
    path = out_dir / f"{urdf.stem}_mujoco.urdf"
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return path


# ── rendering ─────────────────────────────────────────────────────────────────

class PairRenderer:
    """Renders one (shipped, twin) pair into a single labelled image."""

    #: Viewport (w, h) per body plan, matched to the pair's aspect.
    VIEWPORT = {"humanoid": (880, 1120), "quadruped": (1680, 720)}

    def __init__(self, width: int = 900, height: int = 1180,
                 azimuth: float = 128.0, elevation: float = -12.0):
        self.width, self.height = width, height
        self.azimuth, self.elevation = azimuth, elevation

    # ── one model ────────────────────────────────────────────────────────────
    def _prepared(self, model_path: Path, blob_hands: bool = False,
                  role_map=None, driven_joints=None):
        """Load a model as an `MjSpec`, give it a floating base, and strip it to
        its own solids (vendor lights and skybox removed; the scene adds its own)."""
        import mujoco
        # Not `from_file`: MuJoCo's path-keyed mesh cache misses rebuilt files.
        from .measure import spec_from_file
        spec = spec_from_file(model_path)

        # URDF roots are welded to the world; add a free joint so it can be dropped.
        roots = list(spec.worldbody.bodies)
        if roots and not any(j.type == mujoco.mjtJoint.mjJNT_FREE
                             for j in roots[0].joints):
            roots[0].add_freejoint()

        # Shared with every viewer of a pair (`prepare`).
        self.hidden_linkage = self.prepare(
            spec, blob_hands=blob_hands, role_map=role_map,
            driven_joints=driven_joints)

        # Drop vendor lights and skybox, on both halves, before attaching.
        for light in list(spec.lights):
            spec.delete(light)
        for tex in list(spec.textures):
            if tex.type == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
                spec.delete(tex)
        return spec

    # ── the scene both halves stand in ───────────────────────────────────────
    def _dress(self, spec, half: float = 3.0, grid: float | None = None) -> None:
        """Add the floor, grid, key light and visual settings, once per scene.

        `half` is the floor's half-width and `grid` the grid's (m).
        """
        grid = half if grid is None else grid
        import mujoco
        spec.add_texture(name="twin_sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                         builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT,
                         rgb1=[0.988, 0.988, 0.984], rgb2=[0.988, 0.988, 0.984],
                         width=256, height=256)
        # Floor and grid are flat boxes with plain rgba: planes and materials
        # render differently depending on the host model.
        spec.worldbody.add_geom(
            name=FLOOR, type=mujoco.mjtGeom.mjGEOM_BOX, size=[half, half, 0.01],
            pos=[0, 0, -0.01], rgba=[0.886, 0.882, 0.870, 1.0],
            contype=0, conaffinity=0, group=0)
        step, w = 0.25, 0.004
        for i in range(1, int(grid / step) + 1):
            for s in (-1, 1):
                off = s * i * step
                major = abs(off * 4 - round(off * 4)) < 1e-6 and (i % 4 == 0)
                grey = 0.72 if major else 0.83
                for axis in (0, 1):
                    size = [w, grid, 0.001] if axis == 0 else [grid, w, 0.001]
                    pos = [off, 0, 0.001] if axis == 0 else [0, off, 0.001]
                    spec.worldbody.add_geom(
                        name=f"{FLOOR}_grid_{axis}_{i}_{s}",
                        type=mujoco.mjtGeom.mjGEOM_BOX, size=size, pos=pos,
                        rgba=[grey, grey, grey * 0.99, 1.0],
                        contype=0, conaffinity=0, group=0)
        spec.worldbody.add_light(
            name="twin_key", pos=[3.0, -2.6, 6.0], dir=[-0.45, 0.38, -1.0],
            type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
            diffuse=[0.42, 0.42, 0.42], specular=[0.05, 0.05, 0.05],
            castshadow=1)

        spec.visual.headlight.diffuse = [0.60, 0.60, 0.60]
        spec.visual.headlight.ambient = [0.32, 0.32, 0.32]
        spec.visual.headlight.specular = [0.10, 0.10, 0.10]
        spec.visual.quality.shadowsize = 4096
        spec.visual.quality.offsamples = 8
        # znear/zfar are fractions of stat.extent; pin it to robot scale so the
        # large floor does not set the clipping planes.
        spec.stat.extent = 2.0
        spec.stat.center = [0.0, 0.0, 0.4]
        spec.visual.map.znear = 0.02
        # Far enough for the long-lens camera distance.
        spec.visual.map.zfar = 60
        spec.visual.global_.fovy = FOVY
        # the offscreen buffer is sized in the model, not by the Renderer
        spec.visual.global_.offwidth = max(spec.visual.global_.offwidth, self.width)
        spec.visual.global_.offheight = max(spec.visual.global_.offheight, self.height)

    # ── both models, one scene ───────────────────────────────────────────────
    def _load_pair(self, shipped: Path, twin: Path, shipped_roles, twin_roles,
                   category: str, shipped_joints=None, twin_joints=None):
        """Both robots attached into one compiled model sharing floor, light and camera.

        The twin is attached under `PAIR_PREFIX`.
        """
        spec = self._prepared(shipped, blob_hands=(category == "humanoid"),
                              role_map=shipped_roles,
                              driven_joints=shipped_joints)
        other = self._prepared(twin, blob_hands=False, role_map=twin_roles,
                               driven_joints=twin_joints)
        # Geom frames per half before attaching (attach consumes `other`).
        solo_ship, solo_twin = self._solid_frames(spec), self._solid_frames(other)
        # Identity frame; separation is applied later via each half's free joint.
        frame = spec.worldbody.add_frame(pos=[0, 0, 0])
        spec.attach(other, prefix=PAIR_PREFIX, frame=frame)
        # Floor and grid large enough that their edges stay out of frame.
        self._dress(spec, half=60.0, grid=30.0)
        pair = spec.compile()
        self._check_solids_moved_nowhere(pair, {"": solo_ship, PAIR_PREFIX: solo_twin})
        return pair

    # ── the pair is the two halves, and nothing else ─────────────────────────
    @staticmethod
    def _solid_frames(spec) -> dict:
        """Geom positions per body name when this half is compiled alone."""
        import mujoco
        model = spec.copy().compile()
        out = {}
        for g in range(model.ngeom):
            body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                     model.geom_bodyid[g]) or ""
            out.setdefault(body, []).append(np.asarray(model.geom_pos[g], float))
        return out

    @staticmethod
    def _check_solids_moved_nowhere(pair, solos: dict, tol: float = 1e-6) -> None:
        """Raise if attaching moved any geom within its body (seen intermittently)."""
        import mujoco
        moved = []
        for g in range(pair.ngeom):
            body = mujoco.mj_id2name(pair, mujoco.mjtObj.mjOBJ_BODY,
                                     pair.geom_bodyid[g]) or ""
            for prefix, solo in solos.items():
                if prefix and not body.startswith(prefix):
                    continue
                want = solo.get(body[len(prefix):] if prefix else body)
                if want is None:
                    continue
                i = g - int(np.argmax(pair.geom_bodyid == pair.geom_bodyid[g]))
                if i < len(want) and np.max(np.abs(want[i] - pair.geom_pos[g])) > tol:
                    mid = pair.geom_dataid[g]
                    what = (mujoco.mj_id2name(pair, mujoco.mjtObj.mjOBJ_MESH, mid)
                            if mid >= 0 else f"geom {g}")
                    moved.append(f"{body}/{what} by "
                                 f"{np.max(np.abs(want[i] - pair.geom_pos[g])):.3f} m")
                break
        if moved:
            raise RuntimeError(
                "attaching the two halves moved solids inside their own bodies, "
                "so this plate would draw a robot in pieces: "
                + "; ".join(moved[:8])
                + (f" (+{len(moved) - 8} more)" if len(moved) > 8 else "")
                + ". Re-run; this has been seen to come and go on identical "
                  "inputs.")

    @staticmethod
    def _halves(model) -> list:
        """Geoms, joints and free-joint address of each half ([shipped, twin]),
        decided by root body; world-body geoms belong to neither."""
        import mujoco
        root = np.zeros(model.nbody, dtype=int)
        for b in range(1, model.nbody):
            p = int(model.body_parentid[b])
            root[b] = b if p == 0 else root[p]

        def side(b: int) -> int:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                     int(root[b])) or ""
            return 1 if name.startswith(PAIR_PREFIX) else 0

        halves = [{"geoms": [], "joints": [], "free": None} for _ in range(2)]
        for g in range(model.ngeom):
            b = int(model.geom_bodyid[g])
            if b == 0:                      # floor and grid — neither robot
                continue
            halves[side(b)]["geoms"].append(g)
        for j in range(model.njnt):
            h = halves[side(int(model.jnt_bodyid[j]))]
            h["joints"].append(j)
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                h["free"] = int(model.jnt_qposadr[j])
        return halves

    @staticmethod
    def _extents(model, data, geoms) -> tuple:
        """(centre, half-extent) per geom in world axes: meshes from their
        vertices, primitives via `_half_extent`."""
        import mujoco
        c, h = [], []
        for g in geoms:
            pos = np.asarray(data.geom_xpos[g], dtype=float)
            if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mid = model.geom_dataid[g]
                v0, nv = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
                v = model.mesh_vert[v0:v0 + nv].reshape(-1, 3)
                w = v @ np.asarray(data.geom_xmat[g]).reshape(3, 3).T + pos
                c.append(0.5 * (w.min(0) + w.max(0)))
                h.append(0.5 * (w.max(0) - w.min(0)))
            else:
                c.append(pos)
                h.append(_half_extent(model.geom_type[g], model.geom_size[g],
                                      np.asarray(data.geom_xmat[g]).reshape(3, 3)))
        return np.asarray(c), np.asarray(h)

    @staticmethod
    def _span(c, h, axis) -> tuple:
        """(lo, hi) of a set of world AABBs projected onto one unit axis."""
        m, r = c @ axis, h @ np.abs(axis)
        return float((m - r).min()), float((m + r).max())

    def _basis(self) -> tuple:
        """The camera's (forward, right, up) from its azimuth and elevation."""
        az, el = np.radians(self.azimuth), np.radians(self.elevation)
        f = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az),
                      np.sin(el)])
        r = np.cross(f, [0.0, 0.0, 1.0])
        r /= np.linalg.norm(r)
        return f, r, np.cross(r, f)

    # ── what the model IS, as opposed to how it is lit ───────────────────────
    @classmethod
    def prepare(cls, spec, blob_hands: bool = False, role_map=None,
                driven_joints=None) -> list:
        """Edit a loaded spec down to the solids worth comparing.

        Drops vendor ground planes, duplicate collision proxies, misplaced
        dressing and closed-loop linkage; optionally blobs the hands. Every viewer
        of a pair should call this. Returns the hidden linkage bodies.
        """
        import mujoco
        for geom in list(spec.worldbody.geoms):
            if geom.type == mujoco.mjtGeom.mjGEOM_PLANE:
                spec.delete(geom)
        cls._drop_collision_proxies(spec)
        cls._drop_stray_dressing(spec)
        # No-op on models without equality constraints (every twin).
        hidden = cls._hide_loop_linkage(spec, role_map, driven_joints)
        if blob_hands:
            cls._blob_hands(spec)
        return hidden

    # ── collision proxies ────────────────────────────────────────────────────
    @staticmethod
    def _drop_collision_proxies(spec) -> int:
        """Delete colliding geoms from bodies that also have a visual geom.

        Avoids z-fighting and keeps proxies out of the framing; bodies with no
        visual geom keep their colliders.
        """
        n = 0
        for body in spec.bodies:
            geoms = list(body.geoms)
            if not any(g.contype == 0 and g.conaffinity == 0 for g in geoms):
                continue
            for geom in geoms:
                if geom.contype or geom.conaffinity:
                    spec.delete(geom)
                    n += 1
        return n

    # ── set-dressing the vendor mis-placed ───────────────────────────────────
    @staticmethod
    def _drop_stray_dressing(spec) -> int:
        """Delete geoms whose mesh/name matches `STRAY_MESH_RE` (no-op on twins)."""
        n = 0
        for body in spec.bodies:
            for geom in list(body.geoms):
                if STRAY_MESH_RE.search(geom.meshname or geom.name or ""):
                    spec.delete(geom)
                    n += 1
        return n

    # ── closed-loop linkages ─────────────────────────────────────────────────
    @staticmethod
    def _hide_loop_linkage(spec, role_map=None, driven_joints=None) -> list:
        """Hide closed-loop linkage members, which `mj_forward` cannot pose.

        A body is linkage when its subtree drives no role-mapped joint and
        contains a `connect` equality endpoint. Returns the hidden body names.
        """
        import mujoco
        probe = spec.compile()
        if probe.neq == 0:
            return []
        endpoints = set()
        for e in range(probe.neq):
            if int(probe.eq_type[e]) != int(mujoco.mjtEq.mjEQ_CONNECT):
                continue
            endpoints.update((int(probe.eq_obj1id[e]), int(probe.eq_obj2id[e])))
        if not endpoints:
            return []
        # Driven joints: `driven_joints` from `measure.py` when given (role
        # patterns are loose and can match linkage cranks); patterns otherwise.
        compiled = [(re.compile(p), r) for p, r in (role_map or ())]
        driven = set()
        for j in range(probe.njnt):
            jn = mujoco.mj_id2name(probe, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
            hit = (jn in driven_joints) if driven_joints else \
                any(pat.search(jn) for pat, _ in compiled)
            if hit:
                driven.add(int(probe.jnt_bodyid[j]))
        children: dict[int, list] = {}
        for b in range(1, probe.nbody):
            children.setdefault(int(probe.body_parentid[b]), []).append(b)

        def subtree(b):
            out, stack = [], [b]
            while stack:
                x = stack.pop()
                out.append(x)
                stack.extend(children.get(x, ()))
            return out

        marked: set = set()
        for b in range(1, probe.nbody):
            if b in marked:
                continue
            sub = subtree(b)
            if any(x in driven for x in sub) or not any(x in endpoints for x in sub):
                continue
            marked.update(sub)
        names = {mujoco.mj_id2name(probe, mujoco.mjtObj.mjOBJ_BODY, b) or ""
                 for b in marked}
        for body in spec.bodies:
            if body.name in names:
                for geom in list(body.geoms):
                    spec.delete(geom)

        # Also drop small marker geoms sitting on the remaining anchor points.
        anchors: dict[str, list] = {}
        for e in range(probe.neq):
            if int(probe.eq_type[e]) != int(mujoco.mjtEq.mjEQ_CONNECT):
                continue
            if int(probe.eq_objtype[e]) != int(mujoco.mjtObj.mjOBJ_BODY):
                continue                       # anchored on sites: nothing to draw
            for oid, sl in ((int(probe.eq_obj1id[e]), slice(0, 3)),
                            (int(probe.eq_obj2id[e]), slice(3, 6))):
                bn = mujoco.mj_id2name(probe, mujoco.mjtObj.mjOBJ_BODY, oid) or ""
                anchors.setdefault(bn, []).append(
                    np.asarray(probe.eq_data[e][sl], float))
        for body in spec.bodies:
            pts = anchors.get(body.name)
            if not pts:
                continue
            for geom in [g for g in list(body.geoms)
                         if max(np.abs(np.asarray(g.size, float))) <= MARKER_SIZE
                         and any(np.linalg.norm(np.asarray(g.pos, float) - a)
                                 <= MARKER_SIZE for a in pts)]:
                spec.delete(geom)
        return sorted(n for n in names if n)

    # ── end effectors ────────────────────────────────────────────────────────
    @staticmethod
    def _blob_hands(spec) -> None:
        """Replace a shipped humanoid's hand meshes with equal-volume spheres.

        The twin's arm ends in a sphere. Visual only: mass and measurements are
        unaffected. Blobs are named `<body>_hand_blob_<group>`.
        """
        import mujoco
        probe = spec.compile()
        blobs: dict[tuple[str, str], tuple[np.ndarray, float]] = {}
        for g in range(probe.ngeom):
            if probe.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mid = probe.geom_dataid[g]
            mesh = mujoco.mj_id2name(probe, mujoco.mjtObj.mjOBJ_MESH, mid) or ""
            if not HAND_MESH_RE.search(mesh):
                continue
            body = mujoco.mj_id2name(probe, mujoco.mjtObj.mjOBJ_BODY,
                                     probe.geom_bodyid[g]) or ""
            # Mesh vertices in the body frame (the frame of an MjSpec geom's pos).
            v0, nv = probe.mesh_vertadr[mid], probe.mesh_vertnum[mid]
            f0, nf = probe.mesh_faceadr[mid], probe.mesh_facenum[mid]
            v = probe.mesh_vert[v0:v0 + nv].reshape(-1, 3)
            faces = probe.mesh_face[f0:f0 + nf].reshape(-1, 3)
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, probe.geom_quat[g])
            v = v @ R.reshape(3, 3).T + probe.geom_pos[g]
            centre, radius = _equal_volume_sphere(v, faces)
            # Slide the blob back so its near surface meets the wrist end of the hand.
            reach = float(np.linalg.norm(centre))
            if reach > 1e-9:
                u = centre / reach
                centre = u * (float((v @ u).min()) + radius)
            blobs[(body, mesh)] = (centre, radius)

        for body in spec.bodies:
            for geom in list(body.geoms):
                key = (body.name, geom.meshname)
                if key not in blobs:
                    continue
                centre, radius = blobs[key]
                body.add_geom(name=f"{body.name}_hand_blob_{geom.group}",
                              type=mujoco.mjtGeom.mjGEOM_SPHERE,
                              size=[radius, 0, 0], pos=centre.tolist(),
                              rgba=geom.rgba, material=geom.material,
                              group=geom.group, contype=0, conaffinity=0)
                spec.delete(geom)

    @staticmethod
    def _pose(model, data, role_map, category: str, twin: bool = False,
              joints=None) -> None:
        """Pose the model by joint role (`STANDING_POSE`, plus `TWIN_POSE` if `twin`).

        Each target is negated if only the opposite sign is in range, which
        handles mirrored limbs without a per-vendor table.
        """
        import mujoco
        targets = dict(STANDING_POSE.get(category) or {})
        if twin:
            targets.update(TWIN_POSE.get(category) or {})
        if not targets:
            return
        compiled = [(re.compile(p), r) for p, r in role_map]
        # `joints` restricts the sweep to one half of the pair.
        for j in (range(model.njnt) if joints is None else joints):
            if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
                continue
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
            role = next((r for pat, r in compiled if pat.search(name)), None)
            if role not in targets:
                continue
            want = targets[role]
            lo, hi = (float(x) for x in model.jnt_range[j])
            if not model.jnt_limited[j]:
                lo, hi = -np.pi, np.pi
            if not lo <= want <= hi:
                want = -want if lo <= -want <= hi else float(np.clip(want, lo, hi))
            data.qpos[model.jnt_qposadr[j]] = want

    # ── the pair, standing and framed ────────────────────────────────────────
    def _stand_pair(self, model, data, halves, role_maps, category: str) -> None:
        """Pose both halves, drop each by its own lowest point, then separate
        them along the camera's right axis by their on-screen widths."""
        import mujoco
        data.qpos[:] = model.qpos0
        for i, (half, rm) in enumerate(zip(halves, role_maps)):
            self._pose(model, data, rm, category, twin=bool(i),
                       joints=half["joints"])
        mujoco.mj_forward(model, data)

        for half in halves:
            c, h = self._extents(model, data, half["geoms"])
            lo, _ = self._span(c, h, np.array([0.0, 0.0, 1.0]))
            if half["free"] is not None:
                data.qpos[half["free"] + 2] += -lo + 0.002
        mujoco.mj_forward(model, data)

        _, right, _ = self._basis()
        widths, mids = [], []
        for half in halves:
            c, h = self._extents(model, data, half["geoms"])
            lo, hi = self._span(c, h, right)
            widths.append(hi - lo)
            mids.append(0.5 * (lo + hi))
        gap = PAIR_GAP * float(np.mean(widths))
        total = widths[0] + gap + widths[1]
        # Shipped on the left of the frame, twin on the right.
        want = [-0.5 * total + 0.5 * widths[0],
                +0.5 * total - 0.5 * widths[1]]
        for half, w, m in zip(halves, want, mids):
            if half["free"] is not None:
                data.qpos[half["free"]:half["free"] + 2] += (w - m) * right[:2]
        mujoco.mj_forward(model, data)

    def _frame_pair(self, model, data, halves):
        """A perspective camera that just contains the pair.

        A corner `(a, b, c)` (camera axes, relative to lookat) is in frame when
        `|a| <= A*(c + d)` and `|b| <= T*(c + d)`, with `T = tan(fovy/2)` and
        `A = aspect*T`; `d` is the largest resulting lower bound, and the lookat
        is re-centred a few times. Returns `(camera, x_of_half)`, the latter each
        half's centre as a fraction of image width.
        """
        import mujoco
        f, right, up = self._basis()
        pts = self._corners(model, data, halves[0]["geoms"] + halves[1]["geoms"])
        # Margin applied to the field of view, which keeps the pair centred.
        tan_v = np.tan(np.radians(model.vis.global_.fovy) / 2.0) / FRAME_MARGIN
        tan_h = tan_v * (self.width / self.height)
        cam_pts = np.column_stack([pts @ right, pts @ up, pts @ f])

        look = cam_pts.mean(0)
        for _ in range(3):
            a, b, c = (cam_pts - look).T
            dist = float(np.max(np.maximum(np.abs(a) / tan_h,
                                           np.abs(b) / tan_v) - c))
            sr, su = a / (c + dist), b / (c + dist)
            look = look + np.array([0.5 * (sr.max() + sr.min()) * dist,
                                    0.5 * (su.max() + su.min()) * dist, 0.0])

        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = look[0] * right + look[1] * up + look[2] * f
        cam.distance = float(dist)
        cam.azimuth, cam.elevation = self.azimuth, self.elevation

        xs = []
        for half in halves:
            hp = self._corners(model, data, half["geoms"])
            a = hp @ right - look[0]
            c = hp @ f - look[2]
            mid = 0.5 * (float((a / (c + dist)).min())
                         + float((a / (c + dist)).max()))
            xs.append(0.5 + 0.5 * mid / tan_h)
        return cam, xs

    def _corners(self, model, data, geoms) -> np.ndarray:
        """The 8 corners of every geom's world AABB, stacked."""
        c, h = self._extents(model, data, geoms)
        sign = np.array([[sx, sy, sz] for sx in (-1, 1)
                         for sy in (-1, 1) for sz in (-1, 1)], dtype=float)
        return (c[:, None, :] + h[:, None, :] * sign[None, :, :]).reshape(-1, 3)

    # ── the pair ─────────────────────────────────────────────────────────────
    def pair(self, shipped: Path, twin: Path, out: Path, title: str,
             left_label: str, right_label: str, shipped_roles, twin_roles,
             category: str, shipped_joints=None, twin_joints=None) -> Path:
        """Both robots, one scene, one camera, one image."""
        import mujoco
        self.width, self.height = self.VIEWPORT.get(category, (1420, 1120))
        model = self._load_pair(shipped, twin, shipped_roles, twin_roles,
                                category, shipped_joints, twin_joints)
        data = mujoco.MjData(model)
        halves = self._halves(model)
        self._stand_pair(model, data, halves, (shipped_roles, twin_roles),
                         category)
        cam, xs = self._frame_pair(model, data, halves)
        opt = mujoco.MjvOption()
        opt.geomgroup[3:] = 0            # collision-only groups off
        with mujoco.Renderer(model, self.height, self.width) as r:
            r.update_scene(data, camera=cam, scene_option=opt)
            img = r.render().copy()
        # Bare plate too, for figures that add their own title.
        from PIL import Image
        bare = out.with_name(f"{out.stem}_bare{out.suffix}")
        bare.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(img).save(bare)
        return _compose(img, out, title, [(xs[0], left_label, (42, 120, 214)),
                                          (xs[1], right_label, (235, 104, 52))])


def _half_extent(gtype, size, R: np.ndarray) -> np.ndarray:
    """Exact world-axis half-extents of one primitive geom with rotation `R`."""
    import mujoco
    s = np.asarray(size, dtype=float)
    A = np.abs(R)
    if gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
        return np.full(3, s[0])
    if gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:       # radius s0, half-length s1
        return A[:, 2] * s[1] + s[0]
    if gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:      # flat caps, so a disc + axis
        return A[:, 2] * s[1] + s[0] * np.hypot(A[:, 0], A[:, 1])
    if gtype == mujoco.mjtGeom.mjGEOM_BOX:
        return A @ s
    if gtype == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        return np.sqrt((R * s) ** 2 @ np.ones(3))
    return np.full(3, float(np.max(s)))              # plane, hfield, sdf: bounding sphere


def _equal_volume_sphere(v: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, float]:
    """(centre, radius) of the sphere with the same volume as a closed mesh.

    Falls back to the bounding box when the signed volume is zero.
    """
    a, b, c = v[faces[:, 0]], v[faces[:, 1]], v[faces[:, 2]]
    vol6 = np.einsum("ij,ij->i", a, np.cross(b, c))
    total = float(vol6.sum())
    if abs(total) < 1e-12:
        centre = 0.5 * (v.min(0) + v.max(0))
        return centre, float(np.mean(v.max(0) - v.min(0)) * 0.5)
    centre = (vol6 @ (a + b + c) / 4.0) / total
    volume = abs(total) / 6.0
    return centre, float((3.0 * volume / (4.0 * np.pi)) ** (1.0 / 3.0))


def _compose(img: np.ndarray, out: Path, title: str, labels: list) -> Path:
    """Add a title and per-robot labels above `img` and save to `out`.

    `labels` is `(x, text, colour)` with x a fraction of image width; close
    labels are pushed apart. The title shrinks or wraps to fit.
    """
    from PIL import Image, ImageDraw

    pad, gap = 16, 10
    h, w = img.shape[0], img.shape[1]
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))

    # Shrink the title to fit, and break it at the em-dash before shrinking it
    # past legible.
    lines, ftitle = [title], _font(30)
    for size in (30, 27, 24, 21):
        ftitle = _font(size)
        if probe.textlength(title, font=ftitle) <= w:
            lines = [title]
            break
        head, sep, tail = title.partition("  —  ")
        if sep and max(probe.textlength(x, font=ftitle) for x in (head, tail)) <= w:
            lines, ftitle = [head, tail], _font(size)
            break

    flabel = _font(22)
    step = ftitle.size + 5
    band = 13 + step * len(lines) + gap + flabel.size + 6

    canvas = Image.new("RGB", (w + pad * 2, h + band + pad), (252, 252, 251))
    canvas.paste(Image.fromarray(img), (pad, band))
    d = ImageDraw.Draw(canvas)
    for i, line in enumerate(lines):
        d.text((pad, 13 + i * step), line, fill=(11, 11, 11), font=ftitle)

    xs = [pad + x * w for x, _, _ in labels]
    widths = [d.textlength(t, font=flabel) for _, t, _ in labels]
    if len(xs) == 2:
        short = (0.5 * (widths[0] + widths[1]) + 24) - (xs[1] - xs[0])
        if short > 0:
            xs[0] -= short / 2
            xs[1] += short / 2
    y = 13 + step * len(lines) + gap
    for x, (_, text, colour), tw in zip(xs, labels, widths):
        x = min(max(x, pad + tw / 2), pad + w - tw / 2)
        d.text((x, y), text, fill=colour, font=flabel, anchor="ma")
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    print(f"  wrote {out}")
    return out


def _font(size: int):
    """A scalable TTF (matplotlib's DejaVu first); PIL's default as last resort."""
    from PIL import ImageFont
    cands = ["DejaVuSans.ttf",
             "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
             "/System/Library/Fonts/Supplemental/Arial.ttf"]
    try:
        import matplotlib
        cands.insert(0, str(Path(matplotlib.__file__).parent
                            / "mpl-data" / "fonts" / "ttf" / "DejaVuSans.ttf"))
    except Exception:
        pass
    for name in cands:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_all(root: Path, targets: list, out: Path) -> list:
    """Render every pair in `targets` with MuJoCo; returns the written paths."""
    import json
    written = []
    for t in targets:
        key = t.key
        meas = root / key / "measurements.json"
        if not meas.exists():
            continue
        d = json.loads(meas.read_text())
        twin_mjcf = Path(d["summary"]["mjcf"])
        shipped = t.mjcf
        if shipped is None or Path(shipped).suffix.lower() == ".urdf":
            shipped = urdf_for_mujoco(Path(shipped or t.urdf),
                                      root / key / "shipped_mjcf")
            if shipped is None:
                print(f"  SKIP {t.label} — could not build a MuJoCo model")
                continue
        twin_roles = (GENERATED_HUMANOID_ROLES if t.category == "humanoid"
                      else GENERATED_QUADRUPED_ROLES)
        title = (f"{t.label}  —  shipped {d['target']['total_mass_kg']:.1f} kg / "
                 f"{d['target']['n_dof']} DOF     "
                 f"twin {d['twin']['total_mass_kg']:.1f} kg / "
                 f"{d['twin']['n_dof']} DOF")
        role_joints = lambda side: {j["name"] for j in d[side]["joints"]   # noqa: E731
                                    if j.get("role")}
        written.append(PairRenderer().pair(
            shipped, twin_mjcf, out / f"twin_render_{key}.png", title,
            f"{t.label} — as shipped", "generated twin — trend-derived",
            t.role_map, twin_roles, t.category,
            shipped_joints=role_joints("target"),
            twin_joints=role_joints("twin")))
    return written
