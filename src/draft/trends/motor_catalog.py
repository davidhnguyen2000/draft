"""Named motor lookup in the actuator catalogue.

Setting e.g. ``L_motor: AK80-9 V3`` in parameters.yaml fills the class's peak
torque, no-load speed and gear ratio from the catalogue; explicit fields win.
Size and mass are derived by the trends in ``feasibility.py``, not copied.
"""

from __future__ import annotations

import json
import math
import warnings
from typing import Optional

_CATALOG: dict[str, list[dict]] = {}


# ── Catalog I/O ────────────────────────────────────────────────────────────────

def load_catalog() -> list[dict]:
    """Load and cache ``actuator_catalog.json`` (the rows the trends are fitted on).

    Returns [] if it is missing.
    """
    from .feasibility import CATALOG_JSON
    if "rows" not in _CATALOG:
        if not CATALOG_JSON.exists():
            warnings.warn(
                f"Motor catalog not found at {CATALOG_JSON}. "
                "Named motor lookup and validation disabled.",
                UserWarning, stacklevel=3,
            )
            _CATALOG["rows"] = []
        else:
            with CATALOG_JSON.open() as f:
                _CATALOG["rows"] = json.load(f)
    return _CATALOG["rows"]


def _norm(name: str) -> str:
    """Normalize a motor name for case/dash/space-insensitive lookup."""
    return name.lower().replace(" ", "_").replace("-", "_")


def find_by_name(name: str) -> Optional[dict]:
    """Return the catalog entry whose name matches (case/dash/space-insensitive)."""
    norm = _norm(name)
    for entry in load_catalog():
        if _norm(entry["name"]) == norm:
            return entry
    return None


def catalog_names() -> list[str]:
    """Return all catalog entry names (for error messages / GUI pickers)."""
    return [e["name"] for e in load_catalog()]


# ── Catalog → parameter dict conversion ───────────────────────────────────────

def catalog_to_class_params(entry: dict) -> dict:
    """Catalogue entry -> r (m), L (m), effort (Nm), velocity (rad/s), rho (kg/m^3).

    rho is mass over the cylindrical package volume.
    """
    r_m = entry["OD_mm"] / 2.0 / 1000.0
    L_m = entry["L_mm"] / 1000.0
    vol = math.pi * r_m**2 * L_m
    rho = entry["mass_kg"] / vol if vol > 0 else 2700.0
    return {
        "r":        r_m,
        "L":        L_m,
        "effort":   entry["tau_peak_Nm"],
        "velocity": entry["omega_NL_rad_s"],
        "rho":      rho,
    }


# ── Named-motor resolution ────────────────────────────────────────────────────

def resolve_catalog_motors(params: dict) -> dict:
    """Fill ``{cls}_motor: "<name>"`` classes with the part's torque, speed and gear.

    Also records the anchor part name. Size and mass are left to the trends;
    keys already in *params* win.
    """
    catalog = load_catalog()
    if not catalog:
        return params

    out = dict(params)
    for cls in _declared_classes(params):
        name_val = params.get(f"{cls}_motor")
        if not isinstance(name_val, str):
            continue  # numeric definition — nothing to resolve

        entry = find_by_name(name_val)
        if entry is None:
            available = ", ".join(catalog_names()[:8]) + " …"
            warnings.warn(
                f"Motor catalog: '{name_val}' (class {cls}) not found in the "
                f"actuator catalogue. Available names include: {available}",
                UserWarning, stacklevel=4,
            )
            continue

        out.setdefault(f"{cls}_motor_effort", entry["tau_peak_Nm"])
        out.setdefault(f"{cls}_motor_velocity", entry["omega_NL_rad_s"])
        gear = gear_ratio_of(entry)
        if gear is not None:
            out.setdefault(f"{cls}_motor_gear", gear)

        out[f"_{cls}_motor_catalog"]         = entry["name"]
        out[f"{cls}_motor_catalog_name"]     = entry["name"]
        out[f"{cls}_motor_catalog_category"] = entry["category"]

    return out


def gear_ratio_of(entry: dict) -> Optional[float]:
    """Reduction ratio of a catalogue entry, from its notes or its name."""
    from .feasibility import parse_gear_ratio
    return parse_gear_ratio(entry.get("notes", "")) or parse_gear_ratio(entry["name"])


def _declared_classes(params: dict) -> list[str]:
    """Motor class letters that appear as ``{cls}_motor*`` keys."""
    seen = []
    for key in params:
        k = str(key)
        for suffix in ("_motor", "_motor_effort"):
            if k.endswith(suffix):
                cls = k[: -len(suffix)]
                if cls and "_" not in cls and cls not in seen:
                    seen.append(cls)
    return seen
