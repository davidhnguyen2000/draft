# The actuator catalogue

Each file holds one actuator class: `qdd.yaml` (33), `midgear.yaml` (33),
`harmonic.yaml` (48). Each row is one complete integrated module. The file sets
the category, so rows have no `category:` field. After editing, run
stages `01` → `05` in `datasets/actuators/scripts/`. To keep a row out
of the fits, list it in `../active_flags.json`; do not delete it.

**Required fields:** `name`, `vendor`, `tau_peak_Nm`, `mass_kg`, and exactly one
of `no_load_rpm` / `omega_NL_rad_s`.

| field | unit | notes |
|---|---|---|
| `name` | | unique across the whole catalogue |
| `vendor` | | groups the vendor-holdout test |
| `source` | | URL of the datasheet the values were read from |
| `tau_peak_Nm` | N·m | peak torque at the output |
| `mass_kg` | kg | whole shipped module: motor, gearbox, driver, encoder, housing |
| `no_load_rpm` / `omega_NL_rad_s` | rpm / rad/s | output side; set only one |
| `P_peak_W` | W | vendor-published mechanical peak only; `null` uses `τ·ω/4` |
| `gear` | N:1 | the only source of the reduction; `null` leaves the row out of gear-dependent fits |
| `OD_mm`, `L_mm` | mm | envelope; for a non-round body, OD is the largest cross-section |
| `rotor_inertia` | block | see below |
| `geared_inrunner` | bool | published rotor sits behind a very large reduction; row is left out of the inertia fit |
| `notes`, `checked` | | free text; `checked` is the date a person verified the row against `source` |

```yaml
rotor_inertia:
  value: 607
  unit: g.cm2      # g.cm2 | kg.cm2 | g.mm2 | kg.m2 — copy the datasheet; do not convert
  kind: rotor      # rotor (motor side) | output (already × N²; needs `gear`)
  source: cubemars.com AK80-6 KV100
```

Getting `kind` wrong puts the armature off by up to 10⁴. Methodology and fitted
trends are in [`docs/actuator-dataset.md`](../../../../docs/actuator-dataset.md).
