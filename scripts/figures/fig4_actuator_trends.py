#!/usr/bin/env python3
"""Paper Fig. 4 — the four actuator relations: reduction, mass, volume, rotor inertia.

    python scripts/figures/fig4_actuator_trends.py

Single-input fits (N, m) are drawn as trend over data; two-input or
physics-imposed relations (V, J) as predicted vs published. Reads `src/draft/trends/data/{actuator_catalog,actuator_trends}.json` and
`datasets/actuators/data/derived/loo_prediction.json`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))

import paper_style as S                                          # noqa: E402
from draft.trends.feasibility import motor_fits                 # noqa: E402

CATS = ["QDD", "MidGear", "Harmonic"]


# ── shared drawing helpers ────────────────────────────────────────────────────

def cat_scatter(ax, rows, xk, yk, alpha=0.82):
    for c in CATS:
        rs = [r for r in rows if r.get("category") == c
              and r.get(xk) is not None and r.get(yk) is not None]
        if rs:
            ax.scatter([r[xk] for r in rs], [r[yk] for r in rs], s=S.M("sc"),
                       marker=S.CAT[c]["m"], c=S.CAT[c]["c"], alpha=alpha,
                       lw=0, edgecolors="none", zorder=3)


def cat_legend(ax, **kw):
    h = [Line2D([], [], ls="none", marker=S.CAT[c]["m"], ms=S.M("ms"),
                color=S.CAT[c]["c"], label=S.label(c)) for c in CATS]
    ax.legend(handles=h, **kw)


def unity(ax, lo, hi):
    ax.set(xscale="log", yscale="log", xlim=(lo, hi), ylim=(lo, hi))
    S.fold_band(ax)
    ax.set_ylim(lo, hi)


# ── (a) reduction ─────────────────────────────────────────────────────────────

def gear_from_speed(laws):
    """(coef, exp) of the omega-on-N fit inverted to N from omega, as the
    generator uses it."""
    exp = 1.0 / laws.speed_exp
    return float(laws.speed_coef ** -exp), float(exp)


def gear_read_r2(rows, coef, exp):
    """R^2 of the inverted law on log N (differs from the forward fit's R^2)."""
    lw = np.log([r["omega_NL_rad_s"] for r in rows])
    ln = np.log([r["gear"] for r in rows])
    resid = ln - (np.log(coef) + exp * lw)
    return float(1 - np.sum(resid ** 2) / np.sum((ln - ln.mean()) ** 2))


def p_gear(ax, D):
    """(a) Gear ratio N against no-load speed."""
    laws, acts = D["trends"], D["acts"]
    rows = [r for r in acts if r.get("gear") and r.get("omega_NL_rad_s")]
    cat_scatter(ax, rows, "omega_NL_rad_s", "gear")
    coef, exp = gear_from_speed(laws)
    ws = np.geomspace(1.2, 82, 60)
    ax.plot(ws, coef * ws ** exp, color=S.LAW, lw=S.M("lw"), zorder=5)
    r2 = gear_read_r2(rows, coef, exp)
    ax.set(xscale="log", yscale="log", xlabel="$\\omega_{\\mathrm{NL}}$ (rad/s)",
           ylabel="gear ratio $N$")
    S.note(ax, f"$R^2 = {r2:.2f}$\n$n = {len(rows)}$", loc="lower left")


# ── (b) mass ──────────────────────────────────────────────────────────────────

def p_mass(ax, D):
    acts = D["acts"]
    # Geared entries only: the population the law is fitted on.
    rows = [r for r in acts if r.get("tau_peak_Nm") and r.get("mass_kg")
            and r.get("gear")]
    cat_scatter(ax, rows, "tau_peak_Nm", "mass_kg")
    ml = D["scaling"]["mass_trend"]
    xs = np.geomspace(2, 1600, 60)
    ax.plot(xs, ml["coef"] * xs ** ml["tau_exp"], color=S.LAW, lw=S.M("lw"), zorder=5)
    ax.set(xscale="log", yscale="log", xlabel="peak torque $\\tau$ (N$\\cdot$m)",
           ylabel="mass $m$ (kg)")
    # Leave-one-out error on the same geared population.
    held = D["loo"]["gear_term_value"]["unit_holdout"]["tau"]["gmfe"]
    S.note(ax, f"$R^2 = {ml['r2']:.2f}$\nheld out {held:.2f}$\\times$",
           loc="upper left")


# ── (c) package volume ────────────────────────────────────────────────────────

def _volume_pred(r, gl):
    return (gl["volume_coef_m3"] * r["tau_peak_Nm"] ** gl["volume_tau_exp"]
            * r["gear"] ** gl["volume_gear_exp"] * 1e6)


def volume_r2(rows, gl):
    """R^2 of the airgap law on log(pi r^2 L) (stored value if present)."""
    if gl.get("r2_volume"):
        return gl["r2_volume"]
    v = np.log([r["volume_cm3"] for r in rows])
    resid = v - np.log([_volume_pred(r, gl) for r in rows])
    return float(1 - resid @ resid / np.sum((v - v.mean()) ** 2))


def p_volume(ax, D):
    """(c) Package volume, predicted from (tau, N) against published."""
    acts, loo = D["acts"], D["loo"]
    gl = D["scaling"]["geometry_trend"]
    # Geared only: the airgap relation needs a reduction.
    rows = [r for r in acts if r.get("tau_peak_Nm") and r.get("volume_cm3")
            and r.get("gear")]
    for c in CATS:
        rs = [r for r in rows if r.get("category") == c]
        if rs:
            ax.scatter([_volume_pred(r, gl) for r in rs],
                       [r["volume_cm3"] for r in rs], s=S.M("sc"),
                       marker=S.CAT[c]["m"], c=S.CAT[c]["c"], alpha=0.82, lw=0,
                       zorder=3)
    v = [r["volume_cm3"] for r in rows] + [_volume_pred(r, gl) for r in rows]
    unity(ax, min(v) / 1.35, max(v) * 1.35)
    ax.set(xlabel="predicted $\\pi r^2\\ell$ (cm$^3$)",
           ylabel="published $\\pi r^2\\ell$ (cm$^3$)")
    g = loo["geometry"]["adopted_vs_mass_route"]["airgap_from_tau_gear"]
    S.note(ax, f"$R^2 = {volume_r2(rows, gl):.2f}$\nheld out "
               f"{g['volume_cm3']['gmfe']:.2f}$\\times$", loc="upper left")


# ── (d) rotor inertia ─────────────────────────────────────────────────────────

def predicted_J(r, il, laws):
    """k * r^4 with r derived from (tau, N), not the published radius."""
    return il["k_areal_kg_m2"] * laws.size(r["tau_peak_Nm"], r["gear"])["r"] ** 4


def inertia_r2(jr, il, laws):
    """R^2 of the tabulated law on log J, in-sample, matching (b) and (c)."""
    j = np.log([r["rotor_inertia_kgm2"] for r in jr])
    resid = j - np.log([predicted_J(r, il, laws) for r in jr])
    return float(1 - resid @ resid / np.sum((j - j.mean()) ** 2))


def inertia_loo_end_to_end(jr, laws):
    """Leave-one-out fold error on k, using the derived (not published) radius."""
    j = np.array([r["rotor_inertia_kgm2"] for r in jr], float)
    basis = np.array([laws.size(r["tau_peak_Nm"], r["gear"])["r"] ** 4 for r in jr])
    f = j / basis
    pred = np.array([np.exp(np.mean(np.log(np.delete(f, i)))) * basis[i]
                     for i in range(len(f))])
    return float(np.exp(np.mean(np.abs(np.log(pred / j)))))


def p_inertia(ax, D):
    """(d) Rotor inertia J = k r^4 (exponent fixed by physics), predicted vs published."""
    acts, il, laws = D["acts"], D["scaling"]["inertia_trend"], D["trends"]
    jr = [r for r in acts if r.get("rotor_inertia_kgm2") and r.get("radius_mm")
          and r.get("tau_peak_Nm") and r.get("gear")
          and not r.get("rotor_is_geared_inrunner")]
    for c in CATS:
        rs = [r for r in jr if r.get("category") == c]
        if rs:
            ax.scatter([predicted_J(r, il, laws) for r in rs],
                       [r["rotor_inertia_kgm2"] for r in rs], s=S.M("sc"),
                       marker=S.CAT[c]["m"], c=S.CAT[c]["c"], alpha=0.82, lw=0,
                       zorder=3)
    v = [r["rotor_inertia_kgm2"] for r in jr] + [predicted_J(r, il, laws) for r in jr]
    unity(ax, min(v) / 1.4, max(v) * 1.4)
    ax.set(xlabel="predicted $J$ (kg$\\cdot$m$^2$)",
           ylabel="published $J$ (kg$\\cdot$m$^2$)")
    S.note(ax, f"$R^2 = {inertia_r2(jr, il, laws):.2f}$\n"
               f"held out {inertia_loo_end_to_end(jr, laws):.2f}$\\times$",
           loc="upper left")


# ── the strip ─────────────────────────────────────────────────────────────────

BANNER = [("a", p_gear), ("b", p_mass), ("c", p_volume), ("d", p_inertia)]


def main() -> int:
    D = dict(acts=S.load("src/draft/trends/data/actuator_catalog.json"),
             scaling=S.load("src/draft/trends/data/actuator_trends.json"),
             loo=S.load("datasets/actuators/data/derived/loo_prediction.json"),
             trends=motor_fits())
    S.use_style("banner")
    fig, axes = plt.subplots(1, 4, figsize=(S.COL2, 1.82))
    for (ltr, fn), ax in zip(BANNER, axes):
        fn(ax, D)
        S.panel(ax, f"({ltr})", dx=-0.33, dy=1.07)
    # One legend, in (a)'s empty upper-right corner.
    cat_legend(axes[0], loc="upper right", handletextpad=0.3, borderpad=0.25)
    # Roughly square panels.
    fig.subplots_adjust(left=0.055, right=0.995, top=0.93, bottom=0.195,
                        wspace=0.36)
    S.save(fig, "fig4_actuator_trends")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
