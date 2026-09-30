#!/usr/bin/env python3
"""Paper Fig. 6 — the three quadruped designs in one scene, single column.

    python scripts/figures/fig6_quadruped_lineup.py [--labels]

Bare plate by default; `--labels` adds mass and knee actuator per design, read
from each design's `feasibility_report.yaml`.

Prerequisites:
    python scripts/generate_quadrupeds.py
    python scripts/figures/render_lineup_plate.py --out generated/plots/paper_lineup.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_style as S  # noqa: E402

SRC = S.ROOT / "generated" / "plots"
PLATE = SRC / "paper_lineup.json"

#: Labels use text ink, not the robot paint, for legibility at small size.


def design_row(name: str) -> dict:
    """Mass and the knee-class actuator, from the build's own report."""
    d = S.ROOT / "generated" / name
    rep = yaml.safe_load((d / "feasibility_report.yaml").read_text())
    par = yaml.safe_load((d / "parameters_resolved.yaml").read_text())
    ac = rep["actuator_checks"]["classes"][par["knee_mot"]]
    return {
        "mass": rep["mass_composition_check"]["total_mass_kg"],
        # dataset key -> display label ("Harmonic" -> "High GR")
        "family": S.label(par.get("L_motor_catalog_category", "—")),
        "tau": ac["effort_Nm"], "gear": ac["gear_ratio"],
    }


def main() -> int:
    import argparse

    from PIL import Image

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--labels", action="store_true",
                    help="letter each design above it (default: bare plate)")
    args = ap.parse_args()

    if not PLATE.exists():
        print("  no plate — run scripts/figures/render_lineup_plate.py "
              "--out generated/plots/paper_lineup.png")
        return 1
    S.use_style()
    side = S.load(str(PLATE.relative_to(S.ROOT)))
    plate = np.asarray(Image.open(SRC / side["plate"]).convert("RGB"))

    h_w = plate.shape[0] / plate.shape[1]
    FS_HEAD, FS_SUB = 7.4, 6.8                       # pt; the caption is 8
    PAD = 0.02                                       # in, plate top to label
    LABEL = PAD + (FS_SUB * 1.3 + FS_HEAD * 1.25) / 72   # in, two-line head
    if not args.labels:
        LABEL = 0.0
    ax_h = S.COL1 * h_w                              # in, the plate's height

    fig = plt.figure(figsize=(S.COL1, ax_h + LABEL))
    ax = fig.add_axes([0, 0, 1, ax_h / (ax_h + LABEL)])
    ax.imshow(plate)
    ax.set_axis_off()

    if not args.labels:
        S.save(fig, "fig6_quadruped_lineup")
        return 0

    rows = {d: design_row(d) for d in side["designs"]}
    y_sub = 1 + PAD / ax_h
    y_head = y_sub + (FS_SUB * 1.3 / 72) / ax_h
    for d, xf in zip(side["designs"], side["x_fraction"]):
        r = rows[d]
        # Line 1: name and mass. Line 2: knee actuator family, torque, gear.
        ax.text(xf, y_head, f"{d.capitalize()} --- {r['mass']:.0f} kg",
                transform=ax.transAxes, ha="center", va="bottom",
                fontsize=FS_HEAD, color=S.INK)
        ax.text(xf, y_sub,
                f"{r['family']}, {r['tau']:.0f}\\,N$\\cdot$m, "
                f"{r['gear']:.0f}:1",
                transform=ax.transAxes, ha="center", va="bottom",
                fontsize=FS_SUB, color=S.INK)

    S.save(fig, "fig6_quadruped_lineup")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
