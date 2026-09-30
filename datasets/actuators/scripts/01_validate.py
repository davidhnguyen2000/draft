#!/usr/bin/env python3
"""Check the hand-maintained catalogue under ``data/actuators/`` before fitting.

Targets wrong units and values on the wrong side of a gearbox.

    ERROR   the pipeline would be wrong; exits non-zero
    WARN    probably wrong, or wrong somewhere else; exits zero
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset  # noqa: E402

errors, warns = [], []


def err(where, msg):
    errors.append(f"{where}: {msg}")


def warn(where, msg):
    warns.append(f"{where}: {msg}")


rows = dataset.actuators()          # raises on anything structurally broken

# ── uniqueness ───────────────────────────────────────────────────────────────
seen = {}
for r in rows:
    seen.setdefault(r["name"], []).append(r["source_file"])
for name, files in seen.items():
    if len(files) > 1:
        warn(name, f"appears {len(files)}x ({', '.join(files)}) — a duplicate row is "
                   "counted twice by every fit")

# ── the human-checked marker ─────────────────────────────────────────────────
# `checked` means a person verified the row against `source`, so it needs a source.
for r in rows:
    if r["checked"] and not r.get("source"):
        warn(f"{r['source_file']} {r['name']}",
             f"marked checked on {r['checked']} but has no source — a check needs "
             "a document it was checked against")

# ── plausibility, per field ──────────────────────────────────────────────────
for r in rows:
    w = f"{r['source_file']} {r['name']}"
    tau, m = r["tau_peak_Nm"], r["mass_kg"]
    if tau <= 0 or m <= 0:
        err(w, "tau_peak_Nm and mass_kg must be positive")
        continue
    td = tau / m
    if td > 400:
        err(w, f"torque density {td:.0f} N·m/kg — nothing built reaches that; "
               "check the mass unit (g vs kg)")
    elif td > 200:
        warn(w, f"torque density {td:.0f} N·m/kg is above anything else catalogued")
    # Every catalogued part is geared, so no-load speed is OUTPUT speed.
    if not 0 < r["omega_NL_rad_s"] < 1500.0:
        err(w, f"no-load speed {r['omega_NL_rad_s']:.1f} rad/s is above 1500 "
               "— for a geared unit, no_load_rpm is OUTPUT rpm, not motor rpm")
    if r["gear"] is not None and not 1 <= r["gear"] <= 3000:
        err(w, f"gear {r['gear']} outside 1:1..3000:1")

    od, L = r.get("OD_mm"), r.get("L_mm")
    if (od is None) != (L is None):
        err(w, "OD_mm and L_mm must both be set or both be null")
    if od is not None:
        if not 3 <= od <= 400 or not 3 <= L <= 400:
            err(w, f"envelope {od}x{L} mm out of range — millimetres, not metres")
        rho = m / (math.pi * (od / 2000) ** 2 * (L / 1000))
        lo, hi = 800, 5000
        if not 300 < rho < 9000:
            err(w, f"envelope density {rho:.0f} kg/m^3 is impossible — "
                   "OD/L or mass is wrong")
        elif not lo < rho < hi:
            warn(w, f"envelope density {rho:.0f} kg/m^3 is outside the {lo}-{hi} "
                    f"band for {r['category']}")

    # ── the rotor-inertia traps ──────────────────────────────────────────────
    e = r.get("rotor_inertia")
    if e:
        if not e.get("source"):
            warn(w, "rotor_inertia has no source — record the page it came from")
        j = r["rotor_inertia_kgm2"]
        if j is None:
            warn(w, "rotor_inertia is kind: output but the row has no gear, "
                    "so it cannot be brought motor-side and is ignored")
        elif od is not None:
            env = 0.5 * (m / (math.pi * (od / 2000) ** 2 * (L / 1000))) * \
                  math.pi * (od / 2000) ** 4 * (L / 1000)
            frac = j / env
            if frac > 1.0:
                err(w, f"rotor inertia is {frac:.1f}x the WHOLE package's solid "
                       "inertia — impossible; check `unit`, or whether the value is "
                       "output-side (`kind: output`)")
            elif r["rotor_is_geared_inrunner"]:
                pass                     # expected to be tiny; that is the point
            elif frac > 0.6:
                warn(w, f"rotor inertia is {frac:.0%} of the solid envelope — high; "
                        "the catalogue runs 3%-42%")
            elif frac < 0.005:
                warn(w, f"rotor inertia is {frac:.2%} of the solid envelope — very "
                        "low. Either the unit is off by a decade, or this really is "
                        "a small motor behind a big reduction, in which case set "
                        "`geared_inrunner: true`")

# ── coverage, so the open slots stay visible ─────────────────────────────────
INTEGRATED = {"QDD", "MidGear", "Harmonic"}
for label, sel in (("integrated", [r for r in rows if r["category"] in INTEGRATED]),):
    nj = sum(1 for r in sel if r["rotor_inertia_kgm2"])
    nd = sum(1 for r in sel if r.get("OD_mm") is not None)
    nc = sum(1 for r in sel if r["checked"])
    extra = ""
    print(f"  {label:11s} n={len(sel):3d}   envelope {nd:3d}   rotor inertia {nj:3d}"
          f"{extra}   checked {nc:3d}")

print()
for line in warns:
    print(f"WARN  {line}")
for line in errors:
    print(f"ERROR {line}")
nchecked = sum(1 for r in rows if r["checked"])
print(f"\n{len(rows)} actuators — "
      f"{len(errors)} error(s), {len(warns)} warning(s)")
print(f"human-checked: {nchecked}/{len(rows)} rows "
      f"({nchecked / len(rows):.0%})" + (
          "" if nchecked == len(rows) else
          "   — remaining: " + ", ".join(
              f"{cat} {sum(1 for r in rows if r['category'] == cat and not r['checked'])}"
              for cat in ("QDD", "MidGear", "Harmonic")
              if any(r["category"] == cat and not r["checked"] for r in rows))))
sys.exit(1 if errors else 0)
