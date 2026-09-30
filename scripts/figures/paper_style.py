"""Shared style for the paper's figures.

scienceplots `science`+`ieee` at IEEEtran column widths, Okabe-Ito colours with
marker shapes, Type-42 fonts, vector PDF plus 600-dpi PNG. Figures only read
data written by other stages; nothing is fitted here.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures"

COL1, COL2 = 3.5, 7.16                   # IEEEtran single / double column (in)

# One colour and marker per actuator category, shared across all figures.
CAT = {
    "QDD":        {"c": "#0072B2", "m": "o"},
    "MidGear":    {"c": "#D55E00", "m": "s"},
    "Harmonic":   {"c": "#009E73", "m": "^"},
}
#: Display labels. The dataset key "Harmonic" is shown as "High GR" (high
#: reduction; not every member is strain-wave). Legends use `label()`.
CAT_LABEL = {"QDD": "QDD", "MidGear": "MidGear", "Harmonic": "High GR"}


def label(cat):
    """Display name for a dataset category key."""
    return CAT_LABEL.get(cat, cat)


INK, MUTED, FAINT = "#111111", "#5A5A5A", "#9A9A9A"
LAW, BAND = "#111111", "#0072B2"


#: `paper`: IEEEtran body size. `banner`: four narrow panels per text width;
#: type stays near paper size and marks shrink instead.
MODE = "paper"
_SCALE = {
    "paper": dict(pt=8.0, note=6.2, sc=8, lw=1.2, ms=3.0, panel=8.0,
                  axlw=0.5, gridlw=0.3, dpi=600),
    "banner": dict(pt=7.6, note=6.4, sc=4.5, lw=1.0, ms=2.5, panel=8.0,
                   axlw=0.45, gridlw=0.3, dpi=600),
}


def M(key):
    """The current scale's value for `key` — see ``_SCALE``."""
    return _SCALE[MODE][key]


def _tex_works() -> bool:
    """Whether matplotlib can typeset through the system LaTeX.

    Setting the style succeeds either way; an incomplete TeX (no `cm-super`,
    so no `type1ec.sty`) fails only when the first label is drawn, mid-figure.
    """
    import io
    try:
        with plt.rc_context({"text.usetex": True}):
            fig = plt.figure(figsize=(1, 1))
            fig.text(0.5, 0.5, r"$\tau$ 1.0")
            fig.savefig(io.BytesIO(), format="png")
            plt.close(fig)
        return True
    except Exception:
        plt.close("all")
        return False


def use_style(mode: str = "paper") -> None:
    """Apply scienceplots `science`+`ieee`, falling back to its `no-latex`
    variant without a working TeX, and to a serif stack without scienceplots."""
    global MODE
    MODE = mode if mode in _SCALE else "paper"
    try:
        import scienceplots  # noqa: F401
        if _tex_works():
            plt.style.use(["science", "ieee"])
        else:
            print("  (no working LaTeX: drawing with mathtext; install texlive, "
                  "cm-super and dvipng for the paper's typesetting)")
            plt.style.use(["science", "ieee", "no-latex"])
    except Exception:
        plt.rcParams.update({"font.family": "serif",
                             "font.serif": ["Nimbus Roman", "DejaVu Serif"],
                             "mathtext.fontset": "dejavuserif"})
    pt = M("pt")
    plt.rcParams.update({
        "font.size": pt, "axes.labelsize": pt, "axes.titlesize": pt,
        "xtick.labelsize": pt * 0.88, "ytick.labelsize": pt * 0.88,
        "legend.fontsize": pt * 0.80,
        "axes.linewidth": M("axlw"), "lines.linewidth": M("lw"),
        "axes.edgecolor": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK, "text.color": INK,
        "xtick.major.width": M("axlw"), "ytick.major.width": M("axlw"),
        "grid.color": "#CCCCCC", "grid.linewidth": M("gridlw"), "grid.alpha": 0.6,
        "legend.frameon": True, "legend.framealpha": 0.93,
        "legend.edgecolor": "#CCCCCC", "legend.borderpad": 0.32,
        "legend.handletextpad": 0.45, "legend.labelspacing": 0.28,
        "figure.dpi": M("dpi"), "savefig.dpi": M("dpi"),
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


# ── annotation helpers ────────────────────────────────────────────────────────

def panel(ax, s, dx=-0.22, dy=1.06):
    """The (a)/(b) letter."""
    ax.text(dx, dy, s, transform=ax.transAxes, va="top", ha="left",
            fontsize=M("panel"))


def note(ax, s, loc="upper left", fs=None, alpha=0.92):
    """Boxed text stamp (e.g. fit quality) in an axes corner."""
    xy = {"upper left": (0.03, 0.97, "left", "top"),
          "upper right": (0.97, 0.97, "right", "top"),
          "lower left": (0.03, 0.03, "left", "bottom"),
          "lower right": (0.97, 0.03, "right", "bottom")}[loc]
    ax.text(xy[0], xy[1], s, transform=ax.transAxes, ha=xy[2], va=xy[3],
            fontsize=fs if fs is not None else M("note"), linespacing=1.35,
            bbox=dict(fc="white", ec="#CCCCCC", lw=0.4, alpha=alpha, pad=1.8))


def fold_band(ax, lo=0.75, hi=1.25, color=BAND, alpha=0.13):
    """Draw y=x with a ±25% band (the fold-error tolerance used throughout)."""
    lim = ax.get_xlim()
    xs = [lim[0], lim[1]]
    ax.fill_between(xs, [x * lo for x in xs], [x * hi for x in xs],
                    color=color, alpha=alpha, lw=0, zorder=0)
    ax.plot(xs, xs, color=INK, lw=0.8, ls="-", zorder=1)
    ax.set_xlim(*lim)


def save(fig, name: str) -> None:
    """Write `figures/<name>.pdf` and `.png`."""
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    print(f"  figures/{name}.pdf / .png")
    plt.close(fig)


def load(rel: str):
    with open(ROOT / rel) as f:
        return json.load(f)
