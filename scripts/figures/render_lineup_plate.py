#!/usr/bin/env python3
"""Render the quadruped lineup in one scene (shared floor, light and camera).

  python scripts/figures/render_lineup_plate.py --out generated/plots/paper_lineup.png   # paper Fig. 6's plate
  python scripts/figures/render_lineup_plate.py --designs cheetah bear   # any subset, left to right

Reads `generated/<name>/` (run `scripts/generate_quadrupeds.py` first). One
scene keeps relative size honest. Extends `twins.render.PairRenderer` from two
robots to N.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
from draft.twins.render import (FRAME_MARGIN, PAIR_GAP, PairRenderer,  # noqa: E402
                          _compose)
from draft.twins.targets import GENERATED_QUADRUPED_ROLES              # noqa: E402
from draft.twins.viser_render import ViserPairRenderer, whiten_ground  # noqa: E402

OUT_ROOT = _REPO / "generated"

#: Label colours (seaborn "deep"), by actuator family: QDD blue, MidGear
#: orange, Harmonic green.
LABEL_RGB = {"cheetah": (76, 114, 176), "bear": (221, 132, 82),
             "giraffe": (85, 168, 104)}
DEFAULT_RGB = (60, 60, 60)

#: Other designs take their label colour from `motor_color` in their config.
VARIANTS = _REPO / "experiments" / "quadruped_variants"


def label_rgb(name: str) -> tuple[int, int, int]:
    """Paint of the design called `name`, as an 8-bit label colour."""
    if name in LABEL_RGB:
        return LABEL_RGB[name]
    cfg = VARIANTS / f"{name}.yaml"
    if not cfg.exists():
        return DEFAULT_RGB
    import yaml
    raw = (yaml.safe_load(cfg.read_text()) or {}).get("motor_color")
    if not raw:
        return DEFAULT_RGB
    return tuple(int(round(255 * float(v))) for v in str(raw).split()[:3])


#: The three designs, in the paper plate's left-to-right order.
DESIGNS = ["cheetah", "giraffe", "bear"]

#: The paper plate's fixed framing.
PLATE_HEIGHT = 860
PX_PER_DESIGN = 700
CAM_AZIMUTH = 128.0
CAM_ELEVATION = -12.0
PLATE_TITLE = "Quadruped design lineup"


class LineupRenderer(PairRenderer):
    """`PairRenderer` for N robots instead of 2."""

    #: Attach prefix for each design after the first (cf. `PAIR_PREFIX`).
    @staticmethod
    def prefix(i: int) -> str:
        return f"lineup{i}/"

    # ── the scene ────────────────────────────────────────────────────────────
    def _load(self, paths: list[Path]):
        """Every design attached into ONE model, sharing floor, light, camera."""
        spec = self._prepared(paths[0], role_map=GENERATED_QUADRUPED_ROLES)
        for i, path in enumerate(paths[1:], start=1):
            other = self._prepared(path, role_map=GENERATED_QUADRUPED_ROLES)
            # Identity frame; lateral spacing is set later via the free joints.
            frame = spec.worldbody.add_frame(pos=[0, 0, 0])
            spec.attach(other, prefix=self.prefix(i), frame=frame)
        # Floor and grid sized so their edges stay out of frame; scale with count.
        k = max(1.0, len(paths) / 3.0)
        self._dress(spec, half=140.0 * k, grid=70.0 * k)
        return spec.compile()

    def _groups(self, model, n: int) -> list:
        """Geoms and joints per design, grouped by root body (as `_halves`)."""
        import mujoco
        root = np.zeros(model.nbody, dtype=int)
        for b in range(1, model.nbody):
            p = int(model.body_parentid[b])
            root[b] = b if p == 0 else root[p]
        prefixes = [self.prefix(i) for i in range(1, n)]

        def which(b: int) -> int:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                     int(root[b])) or ""
            return next((i for i, p in enumerate(prefixes, start=1)
                         if name.startswith(p)), 0)

        groups = [{"geoms": [], "joints": [], "free": None} for _ in range(n)]
        for g in range(model.ngeom):
            b = int(model.geom_bodyid[g])
            if b == 0:                       # floor and grid — nobody's robot
                continue
            groups[which(b)]["geoms"].append(g)
        for j in range(model.njnt):
            grp = groups[which(int(model.jnt_bodyid[j]))]
            grp["joints"].append(j)
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                grp["free"] = int(model.jnt_qposadr[j])
        return groups

    # ── standing them up ─────────────────────────────────────────────────────
    def _stand(self, model, data, groups) -> None:
        """Pose each design, drop it by its own lowest point, and space them
        along the camera's right axis by on-screen width."""
        import mujoco
        data.qpos[:] = model.qpos0
        for grp in groups:
            self._pose(model, data, GENERATED_QUADRUPED_ROLES, "quadruped",
                       joints=grp["joints"])
        mujoco.mj_forward(model, data)

        for grp in groups:
            c, h = self._extents(model, data, grp["geoms"])
            lo, _ = self._span(c, h, np.array([0.0, 0.0, 1.0]))
            if grp["free"] is not None:
                data.qpos[grp["free"] + 2] += -lo + 0.002
        mujoco.mj_forward(model, data)

        _, right, _ = self._basis()
        widths, mids = [], []
        for grp in groups:
            c, h = self._extents(model, data, grp["geoms"])
            lo, hi = self._span(c, h, right)
            widths.append(hi - lo)
            mids.append(0.5 * (lo + hi))
        gap = PAIR_GAP * float(np.mean(widths))
        total = sum(widths) + gap * (len(widths) - 1)
        # Left to right in argument order.
        want, cursor = [], -0.5 * total
        for w in widths:
            want.append(cursor + 0.5 * w)
            cursor += w + gap
        for grp, w, m in zip(groups, want, mids):
            if grp["free"] is not None:
                data.qpos[grp["free"]:grp["free"] + 2] += (w - m) * right[:2]
        mujoco.mj_forward(model, data)

    def _frame(self, model, data, groups):
        """Fit a camera around all designs (as `_frame_pair`).

        Returns (camera, each robot's on-screen x-centre as a width fraction).
        """
        import mujoco
        f, right, up = self._basis()
        every = [g for grp in groups for g in grp["geoms"]]
        pts = self._corners(model, data, every)
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
        for grp in groups:
            hp = self._corners(model, data, grp["geoms"])
            a, c = hp @ right - look[0], hp @ f - look[2]
            s = a / (c + dist)
            xs.append(0.5 + 0.5 * (0.5 * (float(s.min()) + float(s.max()))) / tan_h)
        return cam, xs

    # ── the picture ──────────────────────────────────────────────────────────
    def lineup(self, paths: list[Path], labels: list[str], out: Path,
               title: str, viser: bool = True, port: int = 8099) -> Path:
        """Stand, frame, render (viser or MuJoCo, same camera) and label.

        Writes `<out>_bare.png`, `<out>.json` (label positions) and `<out>`.
        """
        import mujoco
        model = self._load(paths)
        data = mujoco.MjData(model)
        groups = self._groups(model, len(paths))
        self._stand(model, data, groups)
        cam, xs = self._frame(model, data, groups)
        if viser:
            whiten_ground(model)             # the warm grey is a MuJoCo-only fix
            with ViserPairRenderer(port=port,
                                   max_render=(self.width, self.height)) as vr:
                img = vr.render(model, data, cam, self.width, self.height)
        else:
            opt = mujoco.MjvOption()
            opt.geomgroup[3:] = 0            # collision-only groups off
            with mujoco.Renderer(model, self.height, self.width) as r:
                r.update_scene(data, camera=cam, scene_option=opt)
                img = r.render().copy()

        from PIL import Image
        bare = out.with_name(f"{out.stem}_bare{out.suffix}")
        bare.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(img).save(bare)
        print(f"  wrote {bare}")

        # Label positions, so fig6 can re-letter the bare plate in LaTeX.
        import json
        (out.with_suffix(".json")).write_text(json.dumps(
            {"designs": [p.parent.name for p in paths],
             "labels": labels,
             "x_fraction": [round(x, 5) for x in xs],
             "plate": bare.name,
             "size": [self.width, self.height],
             "azimuth": self.azimuth, "elevation": self.elevation}, indent=1))

        colours = [label_rgb(p.parent.name) for p in paths]
        return _compose(img, out, title,
                        [(x, t, c) for x, t, c in zip(xs, labels, colours)])


def mass_of(xml: Path) -> float:
    import mujoco
    from draft.twins.measure import spec_from_file
    return float(np.sum(spec_from_file(xml).compile().body_mass))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--designs", nargs="+", default=DESIGNS,
                    help="generated/<name>/ folders, left to right. Default is "
                         "the paper's three designs.")
    ap.add_argument("--out", default=str(OUT_ROOT / "plots" / "quadruped_lineup.png"),
                    help="Default is generated/plots/, beside the twins' renders, "
                         "where fig6_quadruped_lineup.py reads it.")
    ap.add_argument("--mujoco", action="store_true",
                    help="Draw with MuJoCo's offscreen renderer instead of viser. "
                         "viser is the default: it draws through three.js, so the "
                         "shading is smooth and the ground shadow is soft rather "
                         "than torn across every stacked cylinder.")
    ap.add_argument("--port", type=int, default=8099,
                    help="Port for the viser server the headless browser attaches to.")
    args = ap.parse_args()
    designs = args.designs

    paths = [OUT_ROOT / n / "quadruped.xml" for n in designs]
    missing = [p for p in paths if not p.exists()]
    if missing:
        ap.error("no such design(s): " + ", ".join(str(p.parent) for p in missing)
                 + " — run scripts/generate_quadrupeds.py first")

    labels = [f"{n.capitalize()}   {mass_of(p):.1f} kg"
              for n, p in zip(designs, paths)]
    width = min(9000, max(2000, PX_PER_DESIGN * len(designs)))
    r = LineupRenderer(width=width, height=PLATE_HEIGHT,
                       azimuth=CAM_AZIMUTH, elevation=CAM_ELEVATION)
    r.lineup(paths, labels, Path(args.out), PLATE_TITLE,
             viser=not args.mujoco, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
