#!/usr/bin/env python3
"""Paper Fig. 2 — the four link primitives, rendered with viser.

    python scripts/figures/fig2_link_primitives.py [--labels]

Each panel is geometry from generated humanoid/quadruped models, rendered in a
headless browser. Writes `figures/fig2_link_primitives.{pdf,png}`; bare by
default, `--labels` adds panel names. Panels are scaled independently, and a
narrow (12 deg) lens avoids perspective distortion.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import paper_style as S                                          # noqa: E402

OUT = _REPO / "figures"
#: Source models; the root panel uses the quadruped trunk, which shows the loft.
MODELS = {
    "humanoid": _REPO / "generated" / "figlinks" / "humanoid.xml",
    "quadruped": _REPO / "generated" / "figlinks_quad" / "quadruped.xml",
}

#: Tessellation, finer than the viewer's for close-up stills.
SPHERE_SUBDIVISIONS = 5
CYLINDER_SECTIONS = 192

#: One Okabe-Ito hue per panel; parts differ by lightness. Value = mix toward
#: black (negative: toward white).
HUES = {
    "structure": -0.55,     # mixed this far toward WHITE
    "riser": -0.24,
    "motor": 0.00,          # the hue itself
    "detail": 0.42,         # mixed this far toward black
    "solid": 0.00,
}
#: Azimuth from +x (along a limb after `_align`); kept off -90 so the actuator
#: axis is not viewed end-on.
PANELS = [
    dict(model="humanoid", body="left_upper_leg_link", label="Basic link",
         hue="#0072B2", align="horizontal", azim=-124.0, elev=20.0),
    dict(model="humanoid", body="left_hand_link", label="Sphere link",
         hue="#D55E00", align="horizontal", azim=-103.0, elev=18.0),
    dict(model="humanoid", body="left_foot_link", label="Foot link",
         hue="#009E73", align="upright", azim=-122.0, elev=20.0),
    dict(model="quadruped", body="torso_link", label="Root link",
         hue="#CC79A7", align="upright", azim=-125.0, elev=16.0),
]

#: Shade per geom-name suffix (from `spec_builder`); the riser gets its own so
#: it does not merge with the plate.
SHADE = {"_motor": "motor", "_detail": "detail", "_sphere": "motor",
         "_box": "structure", "_riser": "riser", "_link": "structure",
         "_mesh": "solid"}

FOV_DEG = 12.0
PLATE = (1000, 1000)            # per panel, before supersampling
SUPERSAMPLE = 3
SETTLE = 2.5


# ── geometry ──────────────────────────────────────────────────────────────────

def _shade(hex_color: str, k: float) -> tuple[int, int, int]:
    """`hex_color` mixed `k` toward black, or toward white when `k` is negative."""
    rgb = np.array([int(hex_color[i:i + 2], 16) for i in (1, 3, 5)], float)
    mixed = rgb * (1.0 - k) if k >= 0 else rgb + (255.0 - rgb) * (-k)
    return tuple(int(round(v)) for v in mixed)


def _primitive(geom_type, size, model, dataid):
    """One MuJoCo geom as a finely tessellated trimesh."""
    import trimesh
    from mujoco import mjtGeom

    if geom_type == mjtGeom.mjGEOM_SPHERE:
        return trimesh.creation.icosphere(radius=size[0],
                                          subdivisions=SPHERE_SUBDIVISIONS)
    if geom_type == mjtGeom.mjGEOM_CYLINDER:
        return trimesh.creation.cylinder(radius=size[0], height=2.0 * size[1],
                                         sections=CYLINDER_SECTIONS)
    if geom_type == mjtGeom.mjGEOM_CAPSULE:
        return trimesh.creation.capsule(radius=size[0], height=2.0 * size[1],
                                        count=(96, 96))
    if geom_type == mjtGeom.mjGEOM_BOX:
        return trimesh.creation.box(extents=2.0 * np.asarray(size))
    if geom_type == mjtGeom.mjGEOM_MESH:
        # Use the compiled mesh: MuJoCo re-centres meshes on their inertial frame.
        v0, nv = model.mesh_vertadr[dataid], model.mesh_vertnum[dataid]
        f0, nf = model.mesh_faceadr[dataid], model.mesh_facenum[dataid]
        return trimesh.Trimesh(
            vertices=np.array(model.mesh_vert[v0:v0 + nv]).reshape(-1, 3),
            faces=np.array(model.mesh_face[f0:f0 + nf]).reshape(-1, 3),
            process=False)
    raise ValueError(f"no still-quality build for geom type {geom_type}")


def _align(kind: str) -> np.ndarray:
    """World rotation for a panel: limbs (built along -z) laid along -x;
    "upright" parts (foot, trunk) unchanged."""
    if kind == "upright":
        return np.eye(3)
    c, s = 0.0, 1.0                                   # R_y(+90 deg): -z -> -x
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def parts(model, body: str, hue: str, align: str) -> list:
    """Every geom of `body` (not sites), posed, coloured and centred.

    Returns (list of (mesh, rgb), bounding-box diagonal).
    """
    import mujoco

    R = _align(align)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
    if bid < 0:
        raise KeyError(f"no body {body} in this model")

    out, corners = [], []
    for g in range(model.ngeom):
        if model.geom_bodyid[g] != bid:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        mesh = _primitive(model.geom_type[g], model.geom_size[g], model,
                          model.geom_dataid[g])
        q = np.zeros(9)
        mujoco.mju_quat2Mat(q, model.geom_quat[g])
        M = np.eye(4)
        M[:3, :3] = R @ q.reshape(3, 3)
        M[:3, 3] = R @ model.geom_pos[g]
        mesh.apply_transform(M)
        shade = next((v for k, v in SHADE.items() if name.endswith(k)), "structure")
        out.append((mesh, _shade(hue, HUES[shade])))
        corners.append(mesh.bounds)

    lo = np.min([c[0] for c in corners], axis=0)
    hi = np.max([c[1] for c in corners], axis=0)
    centre = 0.5 * (lo + hi)
    for mesh, _ in out:
        mesh.apply_translation(-centre)
    return out, float(np.linalg.norm(hi - lo))


# ── camera ────────────────────────────────────────────────────────────────────

def pose(azim_deg: float, elev_deg: float, distance: float):
    """Camera position and `wxyz` for an orbit angle, looking at the origin.

    `elev_deg` > 0 is above the horizon (opposite of `mjvCamera.elevation`).
    viser's `wxyz` uses the OpenCV convention: camera looks along +z, +y down.
    """
    import viser.transforms as vtf

    az, el = np.radians(azim_deg), np.radians(-elev_deg)
    f = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    right = np.cross(f, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, f)
    wxyz = vtf.SO3.from_matrix(np.column_stack([right, -up, f])).wxyz
    return -distance * f, wxyz


def fit_distance(extent: float) -> float:
    """Camera distance at which a sphere of `extent` fits the frame with margin."""
    return 0.62 * extent / np.tan(np.radians(FOV_DEG) / 2.0)


# ── rendering ─────────────────────────────────────────────────────────────────

class PanelRenderer:
    """A viser server plus one headless browser (the renderer), reused per panel."""

    def __init__(self, port: int = 8107):
        from draft.twins.viser_render import find_browser, free_port, _quiet_websockets
        import subprocess
        import tempfile
        import viser

        browser = find_browser()
        if browser is None:
            raise RuntimeError("no Chrome/Chromium found — viser draws in a browser.")
        _quiet_websockets()
        self.server = viser.ViserServer(port=free_port(port), verbose=False)
        self._profile = tempfile.mkdtemp(prefix="link_primitives_")
        w, h = PLATE[0] * SUPERSAMPLE + 120, PLATE[1] * SUPERSAMPLE + 120
        self._proc = subprocess.Popen(
            [browser, "--headless=new", "--use-angle=swiftshader",
             "--disable-gpu-sandbox", "--no-first-run",
             "--no-default-browser-check", f"--window-size={w},{h}",
             f"--user-data-dir={self._profile}",
             f"http://localhost:{self.server.get_port()}"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        print("  waiting for a headless browser …", end="", flush=True)
        deadline = time.time() + 60.0
        while time.time() < deadline and not self.server.get_clients():
            time.sleep(0.25)
        clients = self.server.get_clients()
        if not clients:
            self.close()
            raise RuntimeError("no browser connected within 60 s.")
        print(" connected", flush=True)
        self.client = next(iter(clients.values()))

    def close(self) -> None:
        import shutil
        if getattr(self, "_proc", None) is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except Exception:
                self._proc.kill()
        if getattr(self, "server", None) is not None:
            self.server.stop()
        if getattr(self, "_profile", None):
            shutil.rmtree(self._profile, ignore_errors=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def render(self, meshes: list, extent: float, azim: float,
               elev: float) -> np.ndarray:
        from PIL import Image

        scene = self.server.scene
        scene.reset()
        scene.set_background_image(np.full((2, 2, 3), 255, dtype=np.uint8))
        # Explicit light rig instead of viser's HDRI map, for stable, moderate
        # contrast.
        scene.configure_environment_map(None)
        scene.configure_default_lights(False)
        # Key/ambient ratio chosen so face angle sets shade without crushing to black.
        scene.add_light_ambient("/ambient", intensity=1.25)
        scene.add_light_hemisphere("/hemisphere", sky_color=(255, 255, 255),
                                   ground_color=(150, 150, 160), intensity=1.0)
        scene.add_light_directional("/key", intensity=2.0,
                                    position=(-2.0, -3.0, 4.0), cast_shadow=True)
        scene.add_light_directional("/fill", intensity=0.55,
                                    position=(3.0, -1.5, 1.0))
        for i, (mesh, rgb) in enumerate(meshes):
            scene.add_mesh_simple(f"/part_{i}", mesh.vertices, mesh.faces,
                                  color=rgb, flat_shading=False,
                                  cast_shadow=True, receive_shadow=True)

        pos, wxyz = pose(azim, elev, fit_distance(extent))
        fov = float(np.radians(FOV_DEG))
        self.client.camera.position, self.client.camera.wxyz = pos, wxyz
        self.client.camera.fov = fov
        deadline = time.time() + 10.0
        while time.time() < deadline:
            time.sleep(0.1)
            if np.allclose(self.client.camera.position, pos, atol=1e-3):
                break
        time.sleep(SETTLE)

        img = self.client.get_render(height=PLATE[1] * SUPERSAMPLE,
                                     width=PLATE[0] * SUPERSAMPLE,
                                     position=pos, wxyz=wxyz, fov=fov,
                                     transport_format="png")
        rgb = np.asarray(img)[..., :3]
        return np.asarray(Image.fromarray(rgb).resize(PLATE, Image.LANCZOS))


def crop(rgb: np.ndarray, pad: int = 6) -> np.ndarray:
    """Tight crop to the part, as RGBA with the white background transparent."""
    ink = rgb.min(axis=2) < 246
    if not ink.any():
        raise RuntimeError("the frame came back empty — check the camera pose.")
    ys, xs = np.where(ink)
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, rgb.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, rgb.shape[1])
    sub = rgb[y0:y1, x0:x1]
    # Alpha from distance to white, so antialiased edges have no white fringe.
    alpha = np.clip((255 - sub.min(axis=2).astype(float)) / 9.0, 0.0, 1.0)
    return np.dstack([sub, (alpha * 255).astype(np.uint8)])


# ── the plate ─────────────────────────────────────────────────────────────────

#: Max art height and label band height (inches); the layout is done in inches.
MAX_ART_H, LABEL_H = 0.86, 0.20


def compose(panels: list, out_stem: Path, labels: bool = False) -> list[Path]:
    """Lay the cropped panels in a row at one IEEEtran column; returns paths.

    Each panel is scaled to its own cell; the art height is the tallest needed.
    `labels` adds a name band beneath.
    """
    import matplotlib.pyplot as plt

    S.use_style("paper")
    n = len(panels)
    fig_w = S.COL1
    cell_w = fig_w / n
    # Inches per pixel per panel: width-limited, capped by MAX_ART_H.
    ks = [min(0.96 * cell_w / img.shape[1], 0.98 * MAX_ART_H / img.shape[0])
          for img, _ in panels]
    art_h = max(k * img.shape[0] for k, (img, _) in zip(ks, panels))
    label_h = LABEL_H if labels else 0.0
    fig_h = art_h + label_h
    fig = plt.figure(figsize=(fig_w, fig_h))

    for i, ((img, label), k) in enumerate(zip(panels, ks)):
        h, w = img.shape[:2]
        pw, ph = k * w, k * h
        x = i * cell_w + (cell_w - pw) / 2
        y = label_h + (art_h - ph) / 2
        ax = fig.add_axes([x / fig_w, y / fig_h, pw / fig_w, ph / fig_h])
        ax.imshow(img, interpolation="antialiased")
        ax.set_axis_off()
        if labels:
            fig.text((i + 0.5) / n, 0.5 * label_h / fig_h, label, ha="center",
                     va="center", fontsize=S.M("pt"), color=S.INK)

    written = []
    for ext in ("pdf", "png"):
        p = out_stem.with_suffix(f".{ext}")
        fig.savefig(p, transparent=False, facecolor="white",
                    bbox_inches=None, pad_inches=0)
        written.append(p)
    plt.close(fig)
    return written


def main() -> int:
    import argparse

    import mujoco

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--labels", action="store_true",
                    help="name each panel beneath it (default: bare plate)")
    args = ap.parse_args()

    needed = {p["model"] for p in PANELS}
    missing = [k for k in needed if not MODELS[k].exists()]
    if missing:
        for k in missing:
            print(f"missing {MODELS[k]}\n"
                  f"  python scripts/generate_robot.py --robot {k} --output-dir "
                  f"generated/{MODELS[k].parent.name} --fixed-output-dir",
                  file=sys.stderr)
        return 1

    loaded = {k: mujoco.MjModel.from_xml_path(str(MODELS[k])) for k in needed}
    built = [(p, *parts(loaded[p["model"]], p["body"], p["hue"], p["align"]))
             for p in PANELS]

    OUT.mkdir(parents=True, exist_ok=True)

    plates = []
    with PanelRenderer() as r:
        for spec, meshes, extent in built:
            print(f"  rendering {spec['label']} "
                  f"({spec['model']}/{spec['body']}) …", flush=True)
            img = crop(r.render(meshes, extent, spec["azim"], spec["elev"]))
            plates.append((img, spec["label"]))

    for p in compose(plates, OUT / "fig2_link_primitives", args.labels):
        print(f"  wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
