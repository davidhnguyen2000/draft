#!/usr/bin/env python3
"""Paper Fig. 5 — the four shipping robots rebuilt as twins.

    python scripts/figures/fig5_twin_dynamics.py

Left: twin/vendor render pairs with twin mass as % of vendor. Right: relative
error of the mass matrix M(q) (top) and gravity g(q) (bottom) over sampled
configurations; M and g are kept separate since they do not share units.

Prerequisites:
    python scripts/setup_data.py
    python scripts/make_twins.py       # incl. generated/twins/plots/*_bare.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_style as S                                      # noqa: E402
from draft.twins.dynamics_parity import misaligned, per_config  # noqa: E402

#: The shipped set, in the order they are drawn.
MAIN = ["g1", "h2", "go2", "b2"]
SHORT = {"g1": "G1", "h2": "H2", "go2": "Go2", "b2": "B2"}
N_CFG = 300

#: One spread's cloud shade and ink shade, in the order the spreads are drawn.
_SPREAD_SHADE = ((0.00, 0.00), (0.46, 0.30))

#: Residual drawn per panel (relative error, %).
_DYN_AXIS = {
    "err": r"$\|\Delta M\|_F/\|M\|_F$ (\%)",
    "g_err": r"$\|\Delta g\|/\|g\|$ (\%)",
}


def read(key):
    p = S.ROOT / "generated" / "twins" / key / "measurements.json"
    if not p.exists():
        return None
    return S.load(str(p.relative_to(S.ROOT)))


def _shade(c, f):
    """`f`<0 mixes a colour toward black, `f`>0 toward white."""
    import matplotlib.colors as mcolors
    base = np.array(mcolors.to_rgb(c))
    return tuple(base * (1 + f) if f < 0 else base + (1 - base) * f)


def _hue(key):
    """The twin's motor colour from the synthesizer palette (grey fallback)."""
    try:
        from draft.twins.synthesize import TwinSynthesizer
        return TwinSynthesizer.PALETTES[key]["motor"]
    except Exception:
        return "#8C8C8C"


def _twin_band(M, rows=2, bot=0.10):
    """The plate band plus `rows` stacked axes to its right, laid out in inches.

    Axes are pinned explicitly so a tight bbox does not shrink the figure.
    """
    from PIL import Image

    src = S.ROOT / "generated" / "twins" / "plots"
    # Bare plates; labels are drawn here.
    keys = [k for k in MAIN if (src / f"twin_render_{k}_bare.png").exists()]
    plates = {k: np.asarray(
        Image.open(src / f"twin_render_{k}_bare.png").convert("RGB"))
        for k in keys}

    hum = [k for k in keys if M[k]["target"]["category"] == "humanoid"]
    quad = [k for k in keys if M[k]["target"]["category"] == "quadruped"]
    r_h = np.mean([plates[k].shape[0] / plates[k].shape[1] for k in hum])
    r_q = np.mean([plates[k].shape[0] / plates[k].shape[1] for k in quad])

    # Label colour: the twin's hue, darkened for legibility.
    head = {k: _shade(_hue(k), -0.18) for k in keys}

    FIG_W = S.COL2
    PAD_L, PAD_R = 0.02, 0.02        # in, the figure's own edges
    GAP_HH, GAP_HQ = 0.05, 0.08      # humanoid to humanoid, humanoids to column
    GAP_QV = 0.06                    # between the stacked quadrupeds
    GAP_MID = 0.46                   # render block to right panel, for its ylabel
    PANEL_W = 2.75                   # right-hand axes
    TITLE = 0.14                     # the label line over every plate
    TOP = 0.01

    block_w = FIG_W - PAD_L - GAP_MID - PANEL_W - PAD_R
    # Two portrait plates of height `p_h`, then a column of two landscape
    # plates sharing that height with a title line and a gap between them.
    p_h = ((block_w - GAP_HH - GAP_HQ + (TITLE + GAP_QV) / (2 * r_q))
           / (2 / r_h + 1 / (2 * r_q)))
    w_h = p_h / r_h
    p_q = (p_h - TITLE - GAP_QV) / 2
    w_q = p_q / r_q

    FIG_H = p_h + TITLE + TOP
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    top = FIG_H - TOP

    def _box(x, y, w, h):
        return fig.add_axes([x / FIG_W, y / FIG_H, w / FIG_W, h / FIG_H])

    def _plate(k, x, y, w, h):
        a = _box(x, y, w, h)
        a.imshow(plates[k])
        a.set_axis_off()
        r = M[k]["twin"]["total_mass_kg"] / M[k]["target"]["total_mass_kg"]
        a.set_title(f"{SHORT[k]}  {100 * r:.0f}\\%", fontsize=6.4, pad=1.8,
                    color=head[k])

    for i, k in enumerate(hum):
        _plate(k, PAD_L + i * (w_h + GAP_HH), top - TITLE - p_h, w_h, p_h)
    x_q = PAD_L + 2 * w_h + GAP_HH + GAP_HQ
    for i, k in enumerate(quad):
        _plate(k, x_q, top - (i + 1) * (TITLE + p_q) - i * GAP_QV, w_q, p_q)

    # Only the bottom axes shows platform names.
    x_p, h_p = PAD_L + block_w + GAP_MID, top - bot
    GAP_V = 0.20
    h_each = (h_p - GAP_V * (rows - 1)) / rows
    axes = [_box(x_p, bot + i * (h_each + GAP_V), PANEL_W, h_each)
            for i in range(rows - 1, -1, -1)]
    for a in axes[:-1]:
        a.tick_params(labelbottom=False)
    return fig, axes, keys, head


def _dyn_runs(keys, n_cfg):
    """One sampler run per platform per spread, shared by both panels.

    Rotor armature is excluded (see `per_config`). A second spread holding
    misaligned joints is added only where `misaligned` finds any (the humanoids).
    """
    out = {}
    for k in keys:
        held = misaligned(k)
        out[k] = [per_config(k, n_cfg)]
        if held:
            out[k].append(per_config(k, n_cfg, hold=held))
    return out


def _dyn_panel(bx, keys, cfg, head, field, label_fs=6.2):
    """One platform per tick, one jittered cloud and median bar per spread."""
    rng = np.random.default_rng(1)
    for i, k in enumerate(keys):
        runs = cfg[k]
        # Centre the spreads on the platform's tick.
        offs = [(j - (len(runs) - 1) / 2) * 0.34 for j in range(len(runs))]
        for j, (c, dx) in enumerate(zip(runs, offs)):
            f_cloud, f_ink = _SPREAD_SHADE[j]
            e = c[field] * 100
            bx.scatter(i + dx + (rng.random(len(e)) - 0.5) * 0.22, e, s=1.4,
                       alpha=0.26, lw=0,
                       color=_shade(_hue(k), f_cloud), zorder=3 + j)
            med = float(np.median(e))
            # Explicit ls: the ieee style otherwise cycles line styles.
            bx.plot([i + dx - 0.15, i + dx + 0.15], [med] * 2,
                    color=_shade(head[k], f_ink), lw=1.6, ls="-",
                    solid_capstyle="butt", zorder=6)
            bx.text(i + dx + 0.17, med, f"{med:.0f}\\%", ha="left", va="center",
                    fontsize=5.0, color=_shade(head[k], f_ink), zorder=7,
                    bbox=dict(fc="white", ec="none", pad=0.7, alpha=0.82))
    bx.set_xticks(range(len(keys)))
    bx.set_xticklabels([SHORT[k] for k in keys], fontsize=6.0)
    for t, k in zip(bx.get_xticklabels(), keys):
        t.set_color(head[k])
    # y-limit covers the worst point, not a percentile.
    top = max(c[field].max() for k in keys for c in cfg[k]) * 100 * 1.04
    bx.set(xlim=(-0.50, len(keys) - 0.35), ylim=(0, top))
    bx.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
    bx.set_ylabel(_DYN_AXIS[field], fontsize=label_fs)
    bx.grid(axis="y", alpha=0.4)


def _dyn_legend(bx, fs=6.0, loc="upper right"):
    """A lightness key: the hue is the platform, the ramp is which spread."""
    hh = [Line2D([], [], color=_shade("#6E6E6E", f), lw=3.0, label=n)
          for n, f in (("every joint swept", -0.30),
                       ("misaligned axes held", 0.36))]
    bx.legend(handles=hh, loc=loc, fontsize=fs, borderpad=0.3,
              handlelength=1.0, handletextpad=0.4, labelspacing=0.2,
              borderaxespad=0.35)


def main() -> int:
    S.use_style()
    M = {k: read(k) for k in MAIN}
    if [k for k in MAIN if not M.get(k)]:
        print("  missing twins — run scripts/make_twins.py")
        return 1

    fig, (ax, bx), keys, head = _twin_band(M)
    cfg = _dyn_runs(keys, N_CFG)
    _dyn_panel(ax, keys, cfg, head, "err")
    _dyn_panel(bx, keys, cfg, head, "g_err")
    # One legend serves both panels.
    _dyn_legend(ax)
    S.save(fig, "fig5_twin_dynamics")

    # Print the medians quoted in the paper text.
    for k in keys:
        free = cfg[k][0]
        line = (f"  {SHORT[k]:4s} M {np.median(free['err']) * 100:5.1f}%   "
                f"g {np.median(free['g_err']) * 100:5.1f}%")
        if len(cfg[k]) > 1:
            held = cfg[k][1]
            line += (f"   held: M {np.median(held['err']) * 100:5.1f}%   "
                     f"g {np.median(held['g_err']) * 100:5.1f}%")
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
