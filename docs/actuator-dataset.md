# The actuator dataset

`datasets/actuators/` is a catalogue of 114 integrated rotary actuators and the
stages that fit trends to it. Each entry is a complete module you can buy as one
part: motor, gearing, driver, encoder, bearings and housing. 111 come from the
datasheets of 11 vendors (ZeroErr, MyActuator, CubeMars/T-Motor, HEBI, RobStride,
SteadyWin, Unitree, maxon, Techsoft, Xiaomi, FAULHABER). The other three are MIT
research actuators taken from publications. Because almost every part can be
bought, the catalogue is a performance floor, not a ceiling.

How the generator uses the fits: [`fitted-trends.md`](fitted-trends.md). Where a
number comes from: [`provenance.md`](provenance.md).

## Population

| category | reduction | n | typical parts |
|---|---|---|---|
| QDD | 6–10:1 | 33 | CubeMars AK, Unitree GO-M8010-6, MIT U8/U10/U12, SteadyWin GIM, RobStride |
| MidGear | 7–64:1, planetary | 33 | MyActuator RMD-X, CubeMars high-ratio AK, HEBI X/R, maxon HEJ, FAULHABER BXI |
| Harmonic (High GR) | 50–161:1, mostly strain-wave | 48 | ZeroErr eRob, MyActuator EPS-RH/HM, Techsoft MJ-H, HEBI H |

The catalogue covers 2.5–1180 N·m and 0.19–9.3 kg. 103 entries publish a
reduction and 88 publish a rotor inertia. A row's category is set by the file
it lives in.

## Layout

| path | written by |
|---|---|
| `data/actuators/{qdd,midgear,harmonic}.yaml` | hand; one row per part. Never written by a script |
| `data/active_flags.json` | hand; rows kept in the catalogue but left out of every fit, each with a reason |
| `data/derived/` | the stages; never hand-edited. Delete it and re-run to rebuild |
| `src/draft/trends/data/actuator_{catalog,trends}.json` | the stages; the files the generator reads |

## Pipeline

Run from `datasets/actuators/`, or all at once with
`python scripts/fit_trends.py --only actuators`. The stages take a few seconds.

| stage | reads | writes |
|---|---|---|
| `01_validate.py` | the YAML | nothing. Exits non-zero on unit errors, masses in grams, or a motor-side rpm where an output-side one belongs. Run after every edit |
| `02_build_dataset.py` | the YAML | `data/derived/actuators.json`: flat rows with the peak-power rule applied |
| `03_catalogue_stats.py` | `actuators.json`, `active_flags.json` | `src/draft/trends/data/actuator_catalog.json`: active rows with envelope, gear and rotor inertia (motor-side, kg·m²). Also prints per-category W/kg statistics |
| `04_fit_trends.py` | `actuator_catalog.json` | `src/draft/trends/data/actuator_trends.json`: adopted trends, raw fits, per-category frontier |
| `05_leave_one_out.py` | `actuator_catalog.json` | `data/derived/loo_prediction.json`: every holdout score quoted below |

## Editing the catalogue

The fields are described in
[`datasets/actuators/data/actuators/README.md`](../datasets/actuators/data/actuators/README.md).
The rules that affect a fit:

- **Required fields:** `name` (unique across the whole catalogue), `vendor`,
  `tau_peak_Nm` (peak torque at the output), `mass_kg` (the whole shipped
  module) and exactly one of `no_load_rpm` / `omega_NL_rad_s`, both measured at
  the output. Every other field may be `null`.
- **`gear`** is the only source of the reduction. Nothing parses it out of
  `notes`. A `null` value keeps the row out of every gear-dependent fit.
- **`OD_mm`, `L_mm`** give the envelope. For a non-round body, OD is the largest
  cross-section and L is the length along the shaft.
- **`rotor_inertia`** is a block with `value`, `unit` (`g.cm2 | kg.cm2 | g.mm2 |
  kg.m2`), `kind` and `source`. Enter the number exactly as the datasheet prints
  it. `kind: output` means the vendor quotes it already reflected through the
  gearbox (RobStride does, and so do MyActuator's EPS-RH/HM). The loader divides
  it by N², so it needs a `gear`. Reading `output` as `rotor` puts the armature
  off by up to 10⁴.
- **`geared_inrunner: true`** (HEBI X5/X8/R8) means the published rotor is a
  small motor behind a very large reduction. These rows are kept out of the
  inertia fit.
- **Peak power.** Set `P_peak_W` only when the vendor publishes a peak
  *mechanical* output power. Otherwise `02` computes `P_peak = τ_peak·ω_NL/4`,
  the peak of a linear torque–speed curve. Both values are kept, in `P_peak_W`
  and `envelope_P_W`, and `P_peak_source` records which rule applied. Only one
  row takes the published value: the MIT Mini Cheetah (U8), at 250 W against a
  170 W envelope.
- **Excluding a row.** Add its name to `inactive` in `data/active_flags.json`
  with a reason. `03` drops it before anything is fitted, and the row stays in
  its file.

After an edit, run stages `01` → `05`, then `pytest` (`tests/test_trends.py`
pins Table I).

## Fitted trends

![Paper Fig. 4 — the fitted actuator relations](../figures/fig4_actuator_trends.png)

These are the paper's Table I, in pipeline order. LOO refits the trend without
the module being scored; GMFE is the geometric-mean fold error.

| quantity | trend | n | R² | LOO GMFE |
|---|---|---|---|---|
| reduction | `N = 327·ω_NL^-1.066` (the inverse of `ω_NL = 228.5·N^-0.938`) | 103 | 0.89 | 1.33× (94% within 2×) |
| mass | `m = 0.0653·τ^0.666` kg | 103 | 0.90 | 1.28× (vendor-out 1.31×) |
| volume | `πr²L = τ^0.700·N^-0.167 / 23948` m³ | 103 | 0.87 | 1.30× |
| rotor inertia | `J = 27.2·r⁴` kg·m², r in m | 88 | 0.87 | 1.56× through the generator |

**Mass.** The exponent is 0.666 ± 0.022, so torque goes as `m^1.50` and
isometric scaling is rejected. A gear term was fitted at −0.021 ± 0.034 and then
dropped. It changes mass by 7% across 6:1–161:1, and the paired LOO test cannot
tell the two forms apart (ΔGMFE 0.0008, 95% CI [−0.011, 0.012]). The superseded
fit `0.0670·τ^0.678·N^-0.021` (R² 0.903) is kept under
`mass_trend.gear_term_dropped`. `datasets/robot_descriptions` stage 4 prices
joints with this same adopted `mass_trend`; `raw_fits.mass_vs_tau` over all 114
rows, `0.0749·τ^0.637` (R² 0.88), is kept for reference only.

**Envelope.** A motor makes torque by shearing its airgap:
`τ/N = 2π·σ_eff·r²L`. The `r²L` part is imposed by physics, and only the
effective shear stress is fitted:
`σ_eff = 11974·τ^0.300±0.031·N^-0.833±0.037` Pa (R² 0.84). The positive torque
exponent means shear stress rises with machine size. The negative gear exponent
means the rotor takes a smaller share of the package as the gearbox grows.
Held-out radius predicts to 1.12× and length to 1.14×. Routing geometry through
the mass trend gives 1.19× for both.

**Slenderness.** The airgap relation does not fix how `r²L` splits between r
and L. The catalogue default is
`L/D = 0.401·τ^-0.098±0.021·N^+0.316±0.025` (R² 0.64, LOO 1.19×, against 1.35×
when predicted from mass). Category medians are QDD 0.54, MidGear 0.79 and
Harmonic 1.05. The catalogue spans 0.28–1.69, and a design outside that range
gets a warning, not a refusal.

**Rotor inertia.** The fitted exponent is `J ∝ r^4.02`. The adopted form fixes
it at 4 and fits one areal constant. Reflected inertia is `armature = N²·J`.
Scored against each part's own measured radius, LOO is 1.44×. Scored against
the radius the generator predicts from (τ, N), it is **1.56×**, and that is the
figure to quote. The superseded form `φ·½ρπr⁴L` scored 1.66×: package length is
mostly gearbox, so it overcharged high-reduction modules, and its residual
correlates with log N (−0.67, against 0.16 for the adopted form). The pooled
active fraction φ = 0.0961 is still reported (QDD 0.146, MidGear 0.139,
Harmonic 0.066).

**Speed and frontier.** `ω_NL = 228.5·N^-0.938` (R² 0.90). Mass grows more
slowly than torque, so torque density rises with torque. The per-category
frontier is therefore a ceiling on torque at a given reduction, not a band:

| category | τ/m max (p90), N·m/kg | P/m max (p90), W/kg |
|---|---|---|
| QDD | 84.5 (64.0) | 729 (532) |
| MidGear | 159.0 (117.4) | 871 (362) |
| Harmonic | 141.7 (128.6) | 132 (96) |
| all | 159.0 (121.1) | 871 (419) |

Effective density `m/V` has a median of 2482 kg/m³ (p10–p90 1668–3060).

## Validation (`05_leave_one_out.py`)

Each trend is refit with the held-out unit, vendor or category removed, and then
predicts that unit's published mass from its inputs alone. The table below uses
all 114 rows. `tau_gear` scores only the 103 geared rows.

| predictor | inputs | unit-out | vendor-out | category-out |
|---|---|---|---|---|
| `null` | training median | 2.12× | 2.21× | 2.53× |
| `tau` (adopted) | τ | 1.29× | 1.32× | 1.38× |
| `tau_gear` | τ, N | 1.28× | 1.31× | 1.42× |
| `envelope` | package volume × median ρ | 1.27× | 1.35× | 1.39× |
| `td_budget` | τ ÷ median τ/m | 1.61× | 1.67× | 1.78× |
| `power` | P_peak | 1.94× | 2.25× | 2.69× |

- On the 103 geared rows the adopted trend scores 1.28× unit-out, with 99% of
  modules within 2×, and 1.31× vendor-out. Product siblings share a stator, so
  the vendor holdout is the honest figure.
- Holding out a whole category costs more: QDD 1.18×, MidGear 1.45×,
  Harmonic 1.48×. The gearing regimes are not interchangeable.
- The trend is at its noise floor. The LOO σ_log of 0.326 implies 1.30×, and
  1.29× is observed.
- A robot buys many joints. Resampled as independent parts, the per-part errors
  shrink to 1.18× over 4 joints, 1.14× over 12 and 1.12× over 29. Joints from
  one vendor share their errors, so the true figure lies between the per-part
  and the aggregated numbers.
- Among the dimensionless constants, effective density is the tightest (1.27×;
  QDD 1.16×, MidGear 1.41×, Harmonic 1.10×). Torque density (1.61×) and power
  density (2.36×) are much looser.
- Fold error is ranked rather than R² because R² depends on the spread of the
  test set. For `tau`, R²(log) drops from 0.88 over 4–1180 N·m to 0.43 over
  20–120 N·m, while GMFE stays at 1.28–1.29×.

## Limitations

- The values are datasheet numbers, not bench measurements. Some vendors quote
  torque and speed at optimistic bus voltages with no thermal derating. No
  resampling scheme can detect that; only a prospective holdout against parts
  released later could.
- `P_peak` is derived for 113 of 114 rows, so any fit on power inherits the
  error of two multiplied datasheet numbers. The envelope can miss the real peak
  by about ±30%, depending on whether gear strength, gear speed or heat sets the
  limit.
- There is no thermal data. Peak figures hold for hundreds of milliseconds, and
  continuous capability is typically 30–50% of peak.
- Rotor inertia is sized from the outer housing, and no vendor says whether a
  module is an inrunner or an outrunner. The radius that enters at the fourth
  power can therefore be the wrong one. It is the loosest number the generator
  emits.
- 30 of the 48 Harmonic rows are seven eRob modules repeated once per
  reduction. A torque-only fit weights eRob accordingly.
- The FAULHABER 9317 BXI has no onboard driver, and its no-load speed is quoted
  at 24 V while its torque is quoted at up to 50 V. maxon HEJ and HEBI rows carry
  `gear: null` because their published pages give no reduction a fit can use.
- HEBI's rectangular bodies are approximated as cylinders.
