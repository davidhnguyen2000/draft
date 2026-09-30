"""The four fitted actuator relations (paper Fig. 4), plus the open design's actuators.

  a  reduction from no-load speed        N   ∝ ω^-1.07
  b  mass from peak torque               m   ∝ τ^0.67
  c  volume from torque and reduction    V   ∝ τ^0.70 N^-0.17
  d  rotor inertia from housing radius   J   ∝ r^4      (imposed, not fitted)

`catalogue()` is static and cached; `design_points()` is recomputed per edit.
Both return plain JSON for the page to draw on log-log axes.
"""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from ...trends import ACTUATOR_TRENDS_JSON, CATALOG_JSON

#: panel -> (x field, y field) on a catalogue row, and the axis labels.
PANELS: dict[str, dict[str, Any]] = {
    "a": {"title": "Reduction from speed",
          "x": "no-load speed ω (rad/s)", "y": "reduction N"},
    "b": {"title": "Mass from torque",
          "x": "peak torque τ (N·m)", "y": "mass m (kg)"},
    "c": {"title": "Volume from torque and gear",
          "x": "peak torque τ (N·m)", "y": "volume πr²ℓ (cm³)"},
    "d": {"title": "Rotor inertia from radius",
          "x": "housing radius r (m)", "y": "rotor inertia J (kg·m²)"},
}


def _row_xy(r: dict) -> dict[str, tuple[float, float] | None]:
    """A catalogue row's (x, y) for each panel, or None where it does not publish one."""
    gear, tau = r.get("gear"), r.get("tau_peak_Nm")
    omega, mass = r.get("omega_NL_rad_s"), r.get("mass_kg")
    vol, rad = r.get("volume_cm3"), r.get("radius_mm")
    inertia = r.get("rotor_inertia_kgm2")
    return {
        "a": (omega, gear) if (omega and gear) else None,
        "b": (tau, mass) if (tau and mass) else None,
        "c": (tau, vol) if (tau and vol) else None,
        "d": (rad / 1000.0, inertia) if (rad and inertia) else None,
    }


def _law_curve(panel: str, trends: dict, lo: float, hi: float,
               n: int = 48) -> list[list[float]]:
    """The fitted trend (paper Table I, same JSON the generator uses), sampled
    across the panel's x range. Panel (c) is drawn at the median reduction.
    """
    def sample(f):
        out = []
        for i in range(n):
            x = lo * (hi / lo) ** (i / (n - 1)) if n > 1 else lo
            y = f(x)
            if y and y > 0:
                out.append([x, y])
        return out

    if panel == "a":
        # Fit is w = c*N^e; invert to N = (w/c)^(1/e).
        f = trends["raw_fits"]["speed_vs_gear"]
        c, e = f["coef"], f["exp"]
        return sample(lambda w: (w / c) ** (1.0 / e))
    if panel == "b":
        m = trends["mass_trend"]
        return sample(lambda t: m["coef"] * t ** m["tau_exp"])
    if panel == "c":
        g, N0 = trends["geometry_trend"], trends["mass_trend"]["N_median"]
        return sample(lambda t: 1e6 * g["volume_coef_m3"]
                      * t ** g["volume_tau_exp"] * N0 ** g["volume_gear_exp"])
    if panel == "d":
        return sample(lambda r: trends["inertia_trend"]["k_areal_kg_m2"] * r ** 4)
    return []


@lru_cache(maxsize=1)
def catalogue() -> dict:
    """The surveyed points and the fitted curve, per panel. Static; cached."""
    rows = json.loads(CATALOG_JSON.read_text())
    trends = json.loads(ACTUATOR_TRENDS_JSON.read_text())

    points: dict[str, list] = {k: [] for k in PANELS}
    for r in rows:
        xy = _row_xy(r)
        for panel, pair in xy.items():
            if pair:
                points[panel].append(
                    [pair[0], pair[1], r["category"], r["name"], r["vendor"]])

    laws = {}
    for panel, pts in points.items():
        if not pts:
            laws[panel] = []
            continue
        xs = [p[0] for p in pts]
        try:
            laws[panel] = _law_curve(panel, trends, min(xs), max(xs))
        except (KeyError, ZeroDivisionError):
            # Missing coefficient: drop the curve, keep the points.
            laws[panel] = []

    return {"panels": PANELS, "points": points, "laws": laws,
            "n": trends["provenance"]["n"]}


def design_points(motors: dict) -> list[dict]:
    """Where this design's actuator classes land, in each panel's units.

    `motors` is the solved actuator table; volume is converted m^3 -> cm^3 and
    armature (N^2 J) back to rotor inertia J.
    """
    out = []
    for c in motors.get("classes", []):
        tau, omega = c.get("tau"), c.get("omega")
        gear, mass = c.get("gear"), c.get("mass")
        r, vol_m3, arm = c.get("r"), c.get("volume"), c.get("armature")

        vol_cm3 = vol_m3 * 1e6 if vol_m3 else None
        j = (arm / (gear * gear)) if (arm and gear) else None

        out.append({
            "cls": c.get("cls"),
            "category": c.get("category"),
            "a": [omega, gear] if (omega and gear) else None,
            "b": [tau, mass] if (tau and mass) else None,
            "c": [tau, vol_cm3] if (tau and vol_cm3) else None,
            "d": [r, j] if (r and j) else None,
        })
    return out
