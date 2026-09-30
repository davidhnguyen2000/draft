"""Render a twin pair through viser (three.js) instead of MuJoCo's renderer.

viser renders in the browser, so this launches headless Chrome as the client.
Scene, pose and camera come from `render.PairRenderer`; only the renderer
differs. Two non-obvious details:
  - `camera_tracking_enabled` must be off, or mjviser translates the whole
    scene and a world-frame camera misses the robots.
  - viser's camera uses the OpenCV convention (+Z forward, +Y down).
"""

from __future__ import annotations

import contextlib
import logging
import shutil
import socket
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

#: Candidate browsers. Any Chromium with `--headless=new` (needed for WebGL) works.
BROWSERS = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    shutil.which("google-chrome") or "",
    shutil.which("chromium") or "",
)

#: Seconds to wait for the browser to connect before giving up.
CONNECT_TIMEOUT = 60.0

#: Seconds to let a new scene stream to the client before `get_render`.
SETTLE = 4.0

#: Supersampling factor: render this much larger, then LANCZOS-downsample.
#: Drop to 2 if the browser runs out of memory.
SUPERSAMPLE = 3

#: Primitive tessellation for stills; mjviser's defaults show facets.
SPHERE_SUBDIVISIONS = 4         # 5120 faces, against mjviser's 320
CYLINDER_SECTIONS = 128         # against trimesh's default 32
CAPSULE_COUNT = (64, 64)

#: Floor colour (white, unlike the MuJoCo plates) and how far the grid fades into it.
GROUND_RGBA = (1.0, 1.0, 1.0, 1.0)
GRID_BLEND = 0.62               # 0 = keep the line's own grey, 1 = invisible

#: Max camera distance (m) that still gets ground shadows (viser's shadow cascades
#: end at 20 m). Farther cameras render a uniformly shrunk scene from closer.
SHADOW_REACH = 17.5


def scaled_copy(model, data, k: float):
    """`model` and `data` shrunk uniformly by `k` about the origin, for drawing only."""
    import copy

    import mujoco

    m = copy.deepcopy(model)
    for field in ("geom_size", "geom_pos", "site_size", "site_pos",
                  "body_pos", "mesh_vert"):
        getattr(m, field)[:] *= k
    d = mujoco.MjData(m)
    d.qpos[:] = data.qpos
    mujoco.mj_kinematics(m, d)
    d.xpos[:] = data.xpos * k
    d.xmat[:] = data.xmat
    d.xquat[:] = data.xquat
    if m.nmocap:
        d.mocap_pos[:] = data.mocap_pos * k
        d.mocap_quat[:] = data.mocap_quat
    return m, d


@contextlib.contextmanager
def full_detail_meshes():
    """Disable viser's distance LOD so meshes are not decimated at plate distances."""
    import viser

    originals = {}
    for name in ("add_batched_meshes_trimesh", "add_batched_meshes_simple"):
        original = getattr(viser.SceneApi, name)
        originals[name] = original

        def forced(self, *args, __original=original, **kwargs):
            kwargs["lod"] = "off"
            return __original(self, *args, **kwargs)

        setattr(viser.SceneApi, name, forced)
    try:
        yield
    finally:
        for name, original in originals.items():
            setattr(viser.SceneApi, name, original)


@contextlib.contextmanager
def high_resolution_primitives():
    """Temporarily patch mjviser's `_create_shape_mesh` with finer tessellation."""
    import trimesh
    from mjviser import conversions
    from mujoco import mjtGeom

    original = conversions._create_shape_mesh

    def hi_res(geom_type: int, size) -> "trimesh.Trimesh":
        if geom_type == mjtGeom.mjGEOM_SPHERE:
            return trimesh.creation.icosphere(radius=size[0],
                                              subdivisions=SPHERE_SUBDIVISIONS)
        if geom_type == mjtGeom.mjGEOM_CYLINDER:
            return trimesh.creation.cylinder(radius=size[0], height=2.0 * size[1],
                                             sections=CYLINDER_SECTIONS)
        if geom_type == mjtGeom.mjGEOM_CAPSULE:
            return trimesh.creation.capsule(radius=size[0], height=2.0 * size[1],
                                            count=CAPSULE_COUNT)
        if geom_type == mjtGeom.mjGEOM_ELLIPSOID:
            mesh = trimesh.creation.icosphere(subdivisions=SPHERE_SUBDIVISIONS,
                                              radius=1.0)
            mesh.apply_scale(size)
            return mesh
        return original(geom_type, size)     # box, plane, hfield — unchanged

    conversions._create_shape_mesh = hi_res
    try:
        yield
    finally:
        conversions._create_shape_mesh = original


def whiten_ground(model) -> None:
    """Repaint the floor white and fade its grid in the compiled model (viser only)."""
    import mujoco

    from .render import FLOOR

    ground = np.asarray(GROUND_RGBA, dtype=float)
    for g in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if not name.startswith(FLOOR):
            continue
        if name == FLOOR:
            model.geom_rgba[g] = ground
        else:                                # the metre grid
            model.geom_rgba[g] = (model.geom_rgba[g]
                                  + GRID_BLEND * (ground - model.geom_rgba[g]))


def _quiet_websockets() -> None:
    """Silence harmless `websockets` tracebacks from Chrome's speculative probes."""
    for name in ("websockets", "websockets.server", "websockets.asyncio.server"):
        logging.getLogger(name).setLevel(logging.CRITICAL)


def free_port(preferred: int) -> int:
    """`preferred` if free, else an OS-chosen port (avoids a stale viser server)."""
    with contextlib.closing(socket.socket()) as s:
        try:
            s.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            pass
    with contextlib.closing(socket.socket()) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])

def find_browser() -> str | None:
    """The first browser on this machine that can host a viser client."""
    return next((b for b in BROWSERS if b and Path(b).exists()), None)


class ViserPairRenderer:
    """A viser server plus a headless browser; a context manager reused across pairs."""

    #: Software GL (SwiftShader): slow but identical across machines.
    gl_flags: tuple[str, ...] = ("--use-angle=swiftshader",)

    def __init__(self, port: int = 8099, verbose: bool = False,
                 max_render: tuple[int, int] = (1720, 1120)):
        self.port, self.verbose = port, verbose
        #: Largest plate before supersampling; sizes the browser window, since
        #: `get_render` silently crops to the client's size.
        self.max_render = max_render
        self.server = self.client = self._proc = self._profile = None

    # ── lifecycle ────────────────────────────────────────────────────────────
    def __enter__(self):
        import viser
        browser = find_browser()
        if browser is None:
            raise RuntimeError(
                "no Chrome/Chromium found — viser renders in a browser, so one "
                "has to be installed. Use the MuJoCo renderer instead.")
        _quiet_websockets()
        self.port = free_port(self.port)
        self.server = viser.ViserServer(port=self.port, verbose=self.verbose)
        self._profile = tempfile.mkdtemp(prefix="viser_render_")
        self._proc = subprocess.Popen(
            [browser, "--headless=new", *self.gl_flags,
             "--disable-gpu-sandbox", "--no-first-run", "--no-default-browser-check",
             # Window must be at least as big as any requested frame.
             f"--window-size={self.max_render[0] * SUPERSAMPLE + 120},"
             f"{self.max_render[1] * SUPERSAMPLE + 120}",
             f"--user-data-dir={self._profile}",
             f"http://localhost:{self.server.get_port()}"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        print(f"  waiting for a headless browser on :{self.port} …",
              end="", flush=True)
        deadline = time.time() + CONNECT_TIMEOUT
        while time.time() < deadline and not self.server.get_clients():
            if self._proc.poll() is not None:
                print()
                self.__exit__(None, None, None)
                raise RuntimeError(
                    f"the browser exited immediately (code {self._proc.returncode}). "
                    "If Chrome is already running, this can happen when the launch "
                    "is forwarded to it; try again, or pass a different --port.")
            time.sleep(0.25)
        clients = self.server.get_clients()
        if not clients:
            print()
            self.__exit__(None, None, None)
            raise RuntimeError(
                f"no browser connected within {CONNECT_TIMEOUT:.0f}s. viser renders "
                "in a browser, so one has to reach the server; check that nothing "
                "blocks localhost:%d." % self.port)
        print(" connected", flush=True)
        self.client = next(iter(clients.values()))
        return self

    def __exit__(self, *exc) -> None:
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self.server is not None:
            self.server.stop()
        if self._profile:
            shutil.rmtree(self._profile, ignore_errors=True)
        self._proc = self.server = self.client = self._profile = None

    # ── one frame ────────────────────────────────────────────────────────────
    def render(self, model, data, cam, width: int, height: int) -> np.ndarray:
        """One frame of `model` posed by `data`, from the `mjvCamera` `cam`."""
        import viser.transforms as vtf
        from mjviser.scene import ViserMujocoScene

        # Beyond the shadow cascades, shrink the scene and the camera distance
        # together; see `SHADOW_REACH`.
        k = min(1.0, SHADOW_REACH / float(cam.distance))
        if k < 1.0:
            model, data = scaled_copy(model, data, k)

        self.server.scene.reset()
        # White background: headless Chrome's default is black past the floor edge.
        self.server.scene.set_background_image(
            np.full((2, 2, 3), 255, dtype=np.uint8))
        with high_resolution_primitives(), full_detail_meshes():
            scene = ViserMujocoScene(self.server, model, num_envs=1)
        # See the module docstring.
        scene.camera_tracking_enabled = False
        scene.update_from_mjdata(data)

        f, right, up = _basis(cam)
        pos = k * (np.asarray(cam.lookat) - cam.distance * f)
        # OpenCV convention: the camera looks along +Z with +Y down.
        wxyz = vtf.SO3.from_matrix(np.column_stack([right, -up, f])).wxyz
        fov = float(np.radians(model.vis.global_.fovy))

        # Set the client camera (for the settle wait) and pass the same pose to
        # `get_render`.
        self.client.camera.position = pos
        self.client.camera.wxyz = wxyz
        self.client.camera.fov = fov
        deadline = time.time() + 10.0
        while time.time() < deadline:
            time.sleep(0.1)
            if np.allclose(self.client.camera.position, pos, atol=1e-3):
                break
        time.sleep(SETTLE)

        from PIL import Image
        img = self.client.get_render(height=height * SUPERSAMPLE,
                                     width=width * SUPERSAMPLE,
                                     position=pos, wxyz=wxyz, fov=fov,
                                     transport_format="png")
        rgb = np.asarray(img)[..., :3]
        if SUPERSAMPLE > 1:
            rgb = np.asarray(Image.fromarray(rgb).resize((width, height),
                                                         Image.LANCZOS))
        return rgb.copy()


def _basis(cam) -> tuple:
    """(forward, right, up) from the camera's azimuth and elevation."""
    az, el = np.radians(cam.azimuth), np.radians(cam.elevation)
    f = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    r = np.cross(f, [0.0, 0.0, 1.0])
    r /= np.linalg.norm(r)
    return f, r, np.cross(r, f)


def render_all(root: Path, targets: list, out: Path, port: int = 8099) -> list:
    """Render every pair in `targets` via viser, with the same filenames as `render.render_all`."""
    import json

    import mujoco
    from PIL import Image

    from .render import PairRenderer, _compose, urdf_for_mujoco
    from .targets import GENERATED_HUMANOID_ROLES, GENERATED_QUADRUPED_ROLES

    written = []
    # Size the browser for the biggest plate any of these targets will want.
    sizes = [PairRenderer.VIEWPORT.get(t.category, (1420, 1120)) for t in targets]
    biggest = (max(w for w, _ in sizes), max(h for _, h in sizes)) if sizes else (1720, 1120)
    with ViserPairRenderer(port=port, max_render=biggest) as viser_r:
        for t in targets:
            meas = root / t.key / "measurements.json"
            if not meas.exists():
                continue
            d = json.loads(meas.read_text())
            twin_mjcf = Path(d["summary"]["mjcf"])
            shipped = t.mjcf
            if shipped is None or Path(shipped).suffix.lower() == ".urdf":
                shipped = urdf_for_mujoco(Path(shipped or t.urdf),
                                          root / t.key / "shipped_mjcf")
                if shipped is None:
                    print(f"  SKIP {t.label} — could not build a MuJoCo model")
                    continue
            twin_roles = (GENERATED_HUMANOID_ROLES if t.category == "humanoid"
                          else GENERATED_QUADRUPED_ROLES)
            role_joints = {side: {j["name"] for j in d[side]["joints"] if j.get("role")}
                           for side in ("target", "twin")}

            r = PairRenderer()
            r.width, r.height = r.VIEWPORT.get(t.category, (1420, 1120))
            model = r._load_pair(Path(shipped), twin_mjcf, t.role_map, twin_roles,
                                 t.category, role_joints["target"],
                                 role_joints["twin"])
            data = mujoco.MjData(model)
            halves = r._halves(model)
            r._stand_pair(model, data, halves, (t.role_map, twin_roles), t.category)
            cam, xs = r._frame_pair(model, data, halves)
            whiten_ground(model)

            img = viser_r.render(model, data, cam, r.width, r.height)

            plate = out / f"twin_render_{t.key}.png"
            bare = plate.with_name(f"{plate.stem}_bare{plate.suffix}")
            bare.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(img).save(bare)
            title = (f"{t.label}  —  shipped {d['target']['total_mass_kg']:.1f} kg / "
                     f"{d['target']['n_dof']} DOF     "
                     f"twin {d['twin']['total_mass_kg']:.1f} kg / "
                     f"{d['twin']['n_dof']} DOF")
            written.append(_compose(
                img, plate, title,
                [(xs[0], f"{t.label} — as shipped", (42, 120, 214)),
                 (xs[1], "generated twin — trend-derived", (235, 104, 52))]))
    return written
