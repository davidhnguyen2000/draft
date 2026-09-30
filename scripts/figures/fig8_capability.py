#!/usr/bin/env python3
"""Paper Fig. 8 — capability of the three designs over ten seeds (1x4 strip).

    python scripts/figures/fig8_capability.py

Solid line: nominal median over seeds; band: seed min-max; dashed: inertia-
perturbed median. Stairs pool ascent and descent. Efficiency is drawn only up
to each design's peak command. Speeds are per-rung replica medians. Reads
`draft.lineup.study.STUDY.record` (written by `scripts/train_all.py freeze`).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_style as S  # noqa: E402

from draft.lineup import study  # noqa: E402

RECORD = study.STUDY.record

ORDER = ["cheetah", "bear", "giraffe"]
PAINT = {"cheetah": "#0072B2", "bear": "#D55E00", "giraffe": "#009E73"}
MARK = {"cheetah": "o", "bear": "s", "giraffe": "^"}
SEEDS = range(10)
SURV = 0.8
#: Command axis limit (m/s); every design peaks well below it.
SPEED_XMAX = 10.0


def _stat(vals):
    a = np.array(vals, float)
    return np.median(a, 0), a.min(0), a.max(0)


def load(arm: str) -> dict:
    """Per design: every curve as a (seeds x rungs) array on the shared grid."""
    cells = json.loads(RECORD.read_text())["cells"][arm]
    out = {}
    for r in ORDER:
        cv = [c["curves"] for c in cells[r]]
        out[r] = dict(cmd=np.array(cv[0]["cmd"]), v=np.array([c["v"] for c in cv]),
                      sv=np.array([c["survive"] for c in cv]),
                      eta=np.array([c["eta"] for c in cv]), step=np.array(cv[0]["step"]),
                      trav=np.array([[0.5 * (u + d) for u, d in zip(c["trav_up"], c["trav_down"])]
                                     for c in cv]),
                      dv=np.array(cv[0]["dv"]), push=np.array([c["push_survive"] for c in cv]))
    return out


def _peak(d) -> int:
    """Index of the design's peak command: max median speed over held rungs."""
    vm, _, _ = _stat(d["v"])
    held = np.median(d["sv"], 0) >= SURV
    return int(np.argmax(np.where(held, vm, -np.inf)))


def _dashed(ax, x, y, c):
    ax.plot(x, y, color=c, lw=S.M("lw") * 0.75, ls=(0, (2.2, 1.4)), zorder=4)


def panel_speed(ax, N, P):
    top = 0.0
    for r in ORDER:
        c, d, p = PAINT[r], N[r], P[r]
        keep = d["cmd"] <= SPEED_XMAX + 1e-9
        x = d["cmd"][keep]
        med, lo, hi = (a[keep] for a in _stat(d["v"]))
        k = _peak(d)
        ax.fill_between(x, lo, hi, color=c, alpha=0.18, lw=0, zorder=1)
        # Explicit ls: the ieee style otherwise cycles line styles per series.
        ax.plot(x[:k + 1], med[:k + 1], color=c, lw=S.M("lw"), ls="-",
                marker=MARK[r], ms=S.M("ms"), markevery=[k], zorder=3)
        ax.plot(x[k:], med[k:], color=c, lw=S.M("lw") * 0.8, ls="-", alpha=0.55,
                zorder=2)
        ax.plot(x[k + 1:], med[k + 1:], color=c, ls="", marker=MARK[r],
                ms=S.M("ms"), mfc="white", mew=0.8, zorder=3)
        pm, _, _ = _stat(p["v"])
        _dashed(ax, x, pm[keep], c)
        top = max(top, hi.max())
        ax.annotate(f"{med[k]:.1f}", (x[k], med[k]), xytext=(2.5, 3.5),
                    textcoords="offset points", fontsize=S.M("note"), color=c,
                    ha="left", va="bottom", zorder=5)
    ax.plot([0, top], [0, top], color=S.FAINT, lw=0.6, ls=(0, (4, 3)), zorder=0)
    ax.set_xlim(0, SPEED_XMAX * 1.03)
    ax.set_ylim(0, top * 1.12)
    ax.set_xticks([0, 5, 10])
    ax.set_xlabel("commanded speed (m/s)")
    ax.set_ylabel("achieved speed (m/s)")


def panel_stairs(ax, N, P):
    ax.axhline(50, color=S.FAINT, lw=0.6, ls=(0, (4, 3)), zorder=0)
    for r in ORDER:
        c, d = PAINT[r], N[r]
        x = 100 * d["step"]
        med, lo, hi = _stat(100 * d["trav"])
        ax.fill_between(x, lo, hi, color=c, alpha=0.18, lw=0, zorder=1)
        ax.plot(x, med, color=c, lw=S.M("lw"), ls="-", zorder=3)
        pm, _, _ = _stat(100 * P[r]["trav"])
        _dashed(ax, x, pm, c)
    ax.set_xlim(3, 35)
    ax.set_ylim(-3, 103)
    ax.set_xticks([10, 20, 30])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel("step height (cm)")
    ax.set_ylabel("traversed (\\%)")


def panel_push(ax, N, P):
    ax.axhline(50, color=S.FAINT, lw=0.6, ls=(0, (4, 3)), zorder=0)
    for r in ORDER:
        c, d = PAINT[r], N[r]
        med, lo, hi = _stat(100 * d["push"])
        ax.fill_between(d["dv"], lo, hi, color=c, alpha=0.18, lw=0, zorder=1)
        ax.plot(d["dv"], med, color=c, lw=S.M("lw"), ls="-", zorder=3)
        pm, _, _ = _stat(100 * P[r]["push"])
        _dashed(ax, P[r]["dv"], pm, c)
    ax.set_xlim(0.5, 12.5)
    ax.set_ylim(-3, 103)
    ax.set_xticks([2, 6, 10])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel("velocity kick $\\Delta v$ (m/s)")
    ax.set_ylabel("recovered (\\%)")


def panel_eta(ax, N, P):
    for r in ORDER:
        c = PAINT[r]
        for arm, D in (("nom", N), ("dr", P)):
            d = D[r]
            k = _peak(d)
            vm, _, _ = _stat(d["v"])
            sl = slice(1, k + 1)          # rung 0 is a standstill command
            em, lo, hi = (100 * a[sl] for a in _stat(d["eta"]))
            x = vm[sl]
            if arm == "nom":
                ax.fill_between(x, lo, hi, color=c, alpha=0.18, lw=0, zorder=1)
                ax.plot(x, em, color=c, lw=S.M("lw"), ls="-", marker=MARK[r],
                        ms=S.M("ms"), markevery=[len(x) - 1], zorder=3)
            else:
                _dashed(ax, x, em, c)
    ax.set_ylim(0, 60)
    ax.set_yticks([0, 20, 40, 60])
    ax.set_xlabel("achieved speed (m/s)")
    ax.set_ylabel("drivetrain efficiency (\\%)")


def main() -> int:
    if not RECORD.exists():
        raise SystemExit(f"no {RECORD.relative_to(S.ROOT)} — run scripts/train_all.py freeze")
    N, P = load("nom"), load("dr")
    S.use_style("banner")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(1, 4, figsize=(S.COL2, 1.95))
    for ltr, fn, ax in zip("abcd", (panel_speed, panel_stairs, panel_push, panel_eta), axes):
        fn(ax, N, P)
        ax.grid(True, alpha=0.5)
        ax.set_axisbelow(True)
        S.panel(ax, f"({ltr})", dx=-0.36, dy=1.10)

    h = [Line2D([], [], color=PAINT[r], marker=MARK[r], ms=S.M("ms"), lw=S.M("lw"),
                label=r) for r in ORDER]
    h += [Line2D([], [], color=S.MUTED, lw=S.M("lw"), label="median of 10 seeds"),
          Patch(fc=S.MUTED, alpha=0.25, ec="none", label="seed min--max"),
          Line2D([], [], color=S.MUTED, lw=S.M("lw") * 0.75, ls=(0, (2.2, 1.4)),
                 label="inertia-perturbed median")]
    fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=6,
               frameon=False, fontsize=S.M("note") * 1.05, handletextpad=0.5,
               columnspacing=1.3, handlelength=1.8)
    fig.subplots_adjust(left=0.055, right=0.995, top=0.84, bottom=0.21, wspace=0.42)
    S.save(fig, "fig8_capability")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
