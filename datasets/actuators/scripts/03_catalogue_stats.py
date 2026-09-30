"""Power-to-mass statistics and envelope geometry for the integrated catalogue.

Reads data/derived/actuators.json (from 02) plus data/actuators/*.yaml and writes
actuator_catalog.json, the population the fits consume. A non-round body is
recorded as OD = largest cross-section dimension, L = extent along the shaft.
"""

import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset  # noqa: E402
from draft.jsonio import write_json  # noqa: E402


ROOT = dataset.ROOT
import numpy as np

RPM = 2 * math.pi / 60

CATALOG = dataset.actuators()
BY_NAME = {r["name"]: r for r in CATALOG}      # last row wins on duplicate names

KEEP_CATS = {"QDD", "MidGear", "Harmonic"}     # the integrated actuator catalogue

DIMS = dataset.dims([r for r in CATALOG if r["category"] in KEEP_CATS])


def rotor_inertia_of(name):
    """(J_rotor kg.m^2, kind, source), already motor-side (see dataset.py)."""
    r = BY_NAME.get(name)
    if r is None:
        return None, None, None
    return (r["rotor_inertia_kgm2"], r["rotor_inertia_kind"], r["rotor_inertia_source"])


IN_PATH = dataset.derived("actuators.json")
OUT_PATH = dataset.trends("actuator_catalog.json")

with open(IN_PATH) as f:
    records = json.load(f)

# Parts flagged inactive stay catalogued but reach no fit from here on.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from active_flags import load_inactive, split as split_inactive


INACTIVE = load_inactive(os.path.join(ROOT, "data", "active_flags.json"),
                         expect_dataset="datasets/actuators/data/actuators/*.yaml")
records, _inactive_rows = split_inactive(records, "name", INACTIVE, "actuators")

# Filter: the integrated catalogue the trends are fitted on
filtered = [r for r in records if r["category"] in KEEP_CATS]

print(f"Filtered: {len(filtered)} of {len(records)} actuators")
print(f"Categories kept: QDD, MidGear, Harmonic\n")

# ============================================================
# Compute power-to-mass statistics
# ============================================================
W_per_kg = np.array([r["power_density_W_per_kg"] for r in filtered])
print("="*70)
print(f"PEAK POWER-TO-MASS RATIOS (n={len(filtered)})")
print("="*70)
print(f"  Mean:               {np.mean(W_per_kg):7.1f} W/kg")
print(f"  Median:             {np.median(W_per_kg):7.1f} W/kg")
print(f"  Standard deviation: {np.std(W_per_kg):7.1f} W/kg")
print(f"  90th percentile:    {np.percentile(W_per_kg, 90):7.1f} W/kg")
print(f"  95th percentile:    {np.percentile(W_per_kg, 95):7.1f} W/kg")
print(f"  Max (upper limit):  {np.max(W_per_kg):7.1f} W/kg")
print(f"  Min:                {np.min(W_per_kg):7.1f} W/kg")

# Top 5 frontier members
top5 = sorted(filtered, key=lambda r: -r["power_density_W_per_kg"])[:5]
print(f"\n  Top 5 by W/kg:")
for r in top5:
    print(f"    {r['name']:22s} ({r['vendor']:18s}) {r['category']:8s}  {r['power_density_W_per_kg']:6.0f} W/kg")

# Per-category stats
print(f"\n  Per category:")
print(f"  {'cat':10s} {'n':>3s} {'mean':>7s} {'median':>7s} {'P90':>7s} {'max':>7s}")
for cat in ["QDD", "MidGear", "Harmonic"]:
    pts = [r["power_density_W_per_kg"] for r in filtered if r["category"] == cat]
    if pts:
        print(f"  {cat:10s} {len(pts):3d} {np.mean(pts):7.1f} {np.median(pts):7.1f} "
              f"{np.percentile(pts, 90):7.1f} {np.max(pts):7.1f}")

# ============================================================
# Build dimensional records — only for entries with known dims
# ============================================================
# Rows without published dims are kept with null geometry (dropping them would
# bias the mass/power fits); geometry fits filter on `has_dims`.
dim_recs = []
missing = []
for r in filtered:
    r2 = dict(r)
    if r["name"] in DIMS:
        od_mm, L_mm = DIMS[r["name"]]
        r2["OD_mm"]     = od_mm
        r2["L_mm"]      = L_mm
        r2["radius_mm"] = od_mm / 2.0
        # Cylindrical envelope volume (overestimates rectangular bodies slightly)
        r2["volume_cm3"] = math.pi * (od_mm/2.0/10.0)**2 * (L_mm/10.0)
        r2["has_dims"]  = True
    else:
        missing.append(r["name"])
        r2["OD_mm"] = r2["L_mm"] = r2["radius_mm"] = r2["volume_cm3"] = None
        r2["has_dims"] = False
    j, kind, source = rotor_inertia_of(r["name"])
    r2["rotor_inertia_kgm2"] = j
    r2["rotor_inertia_kind"] = kind          # as published: "rotor" or "output"
    r2["rotor_inertia_source"] = source
    r2["rotor_is_geared_inrunner"] = bool(
        BY_NAME.get(r["name"], {}).get("rotor_is_geared_inrunner"))
    dim_recs.append(r2)

n_dims = sum(r["has_dims"] for r in dim_recs)
print(f"\nRecords: {len(dim_recs)} total, {n_dims} with published dimensions "
      f"({len(missing)} geometry-null)")
if missing:
    print(f"  No published envelope dims: {missing}")

n_rot = sum(1 for r in dim_recs if r["rotor_inertia_kgm2"])
n_out = sum(1 for r in dim_recs if r["rotor_inertia_kind"] == "output")
n_inr = sum(1 for r in dim_recs if r["rotor_is_geared_inrunner"])
print(f"Rotor inertia: {n_rot} of {len(dim_recs)} rows published "
      f"({n_out} of them output-side, reflected motor-side through N^2; "
      f"{n_inr} geared-inrunner, out of the envelope-fraction fit)")
no_j = sorted(r["vendor"] for r in dim_recs if not r["rotor_inertia_kgm2"])
if no_j:
    import collections as _c
    print("  No published rotor inertia: "
          + ", ".join(f"{v} x{n}" for v, n in sorted(_c.Counter(no_j).items())))

# Save filtered data
write_json(OUT_PATH, dim_recs, indent=2)
print(f"\nWrote {OUT_PATH}")
