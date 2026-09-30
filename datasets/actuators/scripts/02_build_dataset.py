"""Flatten ``data/actuators/*.yaml`` into ``data/derived/actuators.json``.

Peak power: the vendor's published peak MECHANICAL power if given, otherwise
τ_peak·ω_NL/4 (linear torque-speed curve, peak at half torque and half speed).
"""

import os
import sys
import math, json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset  # noqa: E402
from draft.jsonio import write_json  # noqa: E402

RPM = 2*math.pi/60

def envelope(tau, omega):
    """Idealized peak mech power for a linear torque-speed actuator."""
    return tau * omega / 4.0

import os
import sys



DATA = dataset.actuators()

records = []
for row in DATA:
    name, vendor, cat = row["name"], row["vendor"], row["category"]
    tau, mass = row["tau_peak_Nm"], row["mass_kg"]
    omega_nl, P_published, note = row["omega_NL_rad_s"], row["P_peak_published_W"], row["notes"]
    P_env = envelope(tau, omega_nl)
    P_peak = P_published if P_published is not None else P_env
    src = "published peak power" if P_published is not None else "envelope τ·ω/4"
    records.append({
        "name": name, "vendor": vendor, "category": cat,
        "tau_peak_Nm": tau,
        "mass_kg": mass,
        "omega_NL_rad_s": omega_nl,
        "no_load_rpm_output": omega_nl/RPM,
        "P_peak_W": P_peak,
        "P_peak_source": src,
        "envelope_P_W": P_env,
        "torque_density_Nm_per_kg": tau / mass,
        "power_density_W_per_kg": P_peak / mass,
        "notes": note,
        "gear": row.get("gear"),      # curated ratio; never parse it from `notes`
        "source": row.get("source"),  # provenance, see docs/provenance.md
    })

OUT_PATH = dataset.derived("actuators.json")
write_json(OUT_PATH, records, indent=2)
print(f"Wrote {OUT_PATH}")

print(f"Total actuators: {len(records)} from {len(dataset.CLASS_FILES)} class files\n")
print(f"{'name':24s} {'vendor':18s} {'cat':10s} {'τpk':>6s} {'m':>6s} {'P_pk':>6s} {'W/kg':>6s}")
print("-"*88)
for r in sorted(records, key=lambda x: -x["power_density_W_per_kg"]):
    print(f"{r['name']:24s} {r['vendor']:18s} {r['category']:10s} "
          f"{r['tau_peak_Nm']:6.1f} {r['mass_kg']:6.3f} {r['P_peak_W']:6.0f} "
          f"{r['power_density_W_per_kg']:6.0f}")
