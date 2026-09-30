#!/usr/bin/env python3
"""Stage 5 — fit the structural (non-actuator) mass trends.

Fit on `m_struct = segment mass - attributed actuator mass` (stage 4), separately
for humanoids (top level of `link_trends.json`) and quadrupeds (`["quadruped"]`):

1. Segment mass fractions (per instance, vs total robot mass).
2. Whole-robot allometry: total mass vs size.
3. Structural linear density `m_struct / L` per segment class.
4. Effective structural density `m_struct / V` (V from the inertia box; biased,
   see docs/robot-dataset.md).
5. Within-robot distal taper along each limb.

Output: `src/draft/trends/data/link_trends.json`, `data/segment_rows.json`.
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict

import numpy as np

from paths import DATA, ROOT, TRENDS_DATA  # noqa: E402
from draft.jsonio import write_json  # noqa: E402

LIMB_CLASSES = ["thigh", "shank", "foot", "upper_arm", "forearm", "hand"]
BODY_CLASSES = ["pelvis", "torso", "head"]
# Short bodies between the axes of a multi-DOF joint, fitted separately from limbs.
JOINT_LINK_CLASSES = ["hip_link", "knee_link", "ankle_link", "shoulder_link",
                   "elbow_link", "wrist_link", "waist_link", "neck_link"]
TRUNK_CLASSES = BODY_CLASSES + ["waist_link", "neck_link"]
FIT_CLASSES = LIMB_CLASSES + BODY_CLASSES + JOINT_LINK_CLASSES

# Links shorter than this have no usable length (some are ~1e-17 m). Also used by stage 6.
MIN_LINK_LEN_M = 1e-3

# Limb links in chain order, proximal first (joint links excluded: dense gearbox lumps).
LIMB_CHAINS = {"leg": ["thigh", "shank", "foot"],
               "arm": ["upper_arm", "forearm", "hand"]}

# Winter (2009), Biomechanics and Motor Control of Human Movement, Table 4.1 --
# segment mass as a fraction of body mass, per side where the segment is paired.
HUMAN_FRACTIONS = {
    "thigh": 0.100, "shank": 0.0465, "foot": 0.0145,
    "upper_arm": 0.028, "forearm": 0.016, "hand": 0.006,
    "torso+pelvis+head": 0.578,
}


def linear_fit(L, m) -> dict | None:
    """m = c L through the origin, fitted in log space: c = exp(mean(log(m/L)))."""
    L, m = np.asarray(L, float), np.asarray(m, float)
    ok = np.isfinite(L) & np.isfinite(m) & (L > 0) & (m > 0)
    L, m = L[ok], m[ok]
    if len(L) < 2:
        return None
    lr = np.log(m / L)
    c = float(np.exp(lr.mean()))
    return {"coef": round(c, 6), "n": int(len(L)),
            "sigma_log": round(float(np.std(lr, ddof=1)), 4),
            "gmfe_insample": round(float(np.exp(np.mean(np.abs(lr - lr.mean())))), 4)}


#: Floor on `m_struct_vs_L`'s length exponent, so a longer member never weighs
#: less; at the floor a class is a constant mass. Stage 6 uses the same floor.
STRUCT_MIN_EXP = 0.0


def power_fit(x, y, min_exp: float | None = None) -> dict | None:
    """Least-squares y = coef * x^exp in log space.

    With `min_exp`, a lower exponent is clamped and the coefficient refitted;
    the free exponent is kept as `exp_free`.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x, y = x[ok], y[ok]
    if len(x) < 4:
        return None
    lx, ly = np.log(x), np.log(y)
    A = np.vstack([lx, np.ones_like(lx)]).T
    (exp, log_coef), *_ = np.linalg.lstsq(A, ly, rcond=None)
    resid = ly - A @ [exp, log_coef]
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    n = len(x)
    # standard error on the slope, of the free fit
    s2 = float(resid @ resid) / (n - 2)
    sxx = float(np.sum((lx - lx.mean()) ** 2))
    se = math.sqrt(s2 / sxx) if sxx > 0 else float("nan")
    free_exp = float(exp)
    bounded = min_exp is not None and free_exp < min_exp
    if bounded:
        exp = float(min_exp)
        log_coef = float(np.mean(ly - exp * lx))
        resid = ly - (exp * lx + log_coef)
    out = {
        "exp": round(float(exp), 4), "coef": round(float(math.exp(log_coef)), 6),
        "exp_se": round(se, 4), "exp_ci95": [round(free_exp - 1.96 * se, 4),
                                             round(free_exp + 1.96 * se, 4)],
        "r2": round(1.0 - float(resid @ resid) / ss_tot, 4) if ss_tot > 0 else None,
        "sigma_log": round(float(np.std(resid, ddof=2)), 4),
        "gmfe_insample": round(float(np.exp(np.mean(np.abs(resid)))), 4),
        "n": n,
    }
    if bounded:
        out["exp_free"] = round(free_exp, 4)
        out["exp_floor"] = float(min_exp)
    return out


def stats(vals) -> dict | None:
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], float)
    if len(v) == 0:
        return None
    return {"n": int(len(v)), "median": round(float(np.median(v)), 5),
            "mean": round(float(np.mean(v)), 5),
            # Reported only; the generator uses the median.
            "sd": (round(float(np.std(v, ddof=1)), 5) if len(v) > 1 else None),
            "p10": round(float(np.percentile(v, 10)), 5),
            "p90": round(float(np.percentile(v, 90)), 5),
            "min": round(float(v.min()), 5), "max": round(float(v.max()), 5)}


def within_robot_taper(rows, chain: list[str], value) -> dict | None:
    """Per-step multiplicative falloff along a limb chain, fit within robots.

    Each robot is demeaned in log space, so this measures shape, not level
    (between-robot scatter is removed). `value` maps a row to the tapered quantity.
    """
    depth = {cls: i for i, cls in enumerate(chain)}
    pts = defaultdict(list)
    for r in rows:
        if r["seg_class"] not in depth:
            continue
        v = value(r)
        if v is None or not np.isfinite(v) or v <= 0:
            continue
        pts[r["robot"]].append((depth[r["seg_class"]], math.log(v), r["seg_class"]))
    # Need at least two chain positions per robot.
    pts = {k: v for k, v in pts.items() if len({d for d, _, _ in v}) >= 2}
    if len(pts) < 4:
        return None
    d, y, cls = [], [], []
    for segs in pts.values():
        mu = float(np.mean([ly for _, ly, _ in segs]))
        for dd, ly, cc in segs:
            d.append(dd); y.append(ly - mu); cls.append(cc)
    d, y = np.asarray(d, float), np.asarray(y, float)
    A = np.vstack([np.ones_like(d), d]).T
    (b0, slope), *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ [b0, slope]
    n = len(d)
    s2 = float(resid @ resid) / (n - 2)
    sdd = float(np.sum((d - d.mean()) ** 2))
    se = math.sqrt(s2 / sdd) if sdd > 0 else float("nan")
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    # the per-class shape, for a consumer that wants the steps rather than a slope
    rel = {}
    for c in chain:
        v = [yy for yy, cc in zip(y, cls) if cc == c]
        if v:
            rel[c] = round(float(math.exp(np.mean(v))), 4)
    return {
        "per_step": round(float(math.exp(slope)), 4),
        "per_step_ci95": [round(float(math.exp(slope - 1.96 * se)), 4),
                          round(float(math.exp(slope + 1.96 * se)), 4)],
        "log_slope": round(float(slope), 4), "log_slope_se": round(float(se), 4),
        "r2": round(1.0 - float(resid @ resid) / ss_tot, 4) if ss_tot > 0 else None,
        "sigma_log": round(float(np.std(resid, ddof=2)), 4),
        "n": n, "n_robots": len(pts),
        "class_relative": rel,
    }


#: The generated quadruped's knee is remotized, so quadruped thigh/shank are fit
#: only over robots with the same knee drive (see stage 4).
GENERATOR_KNEE_DRIVE = "remotized"
KNEE_DRIVE_CLASSES = ("thigh", "shank")


def knee_drive_matches(robot: dict, seg_class: str) -> bool:
    """Whether this robot's `seg_class` segments belong in the generator's fit."""
    return (not robot.get("is_quadruped") or seg_class not in KNEE_DRIVE_CLASSES
            or robot.get("knee_drive") == GENERATOR_KNEE_DRIVE)


def seg_length(seg: dict) -> float | None:
    """Segment length; terminal segments fall back to the furthest fixed frame,
    then the inertia box's long axis."""
    if seg["length"] and seg["length"] > 1e-3:
        return seg["length"]
    return seg.get("tip_len") or seg.get("box_a") or None


def seg_volume(seg: dict) -> float | None:
    """Solid-body volume implied by the inertia-equivalent box."""
    a, b, c = seg.get("box_a"), seg.get("box_b"), seg.get("box_c")
    if not all(v and v > 1e-4 for v in (a, b, c)):
        return None
    return a * b * c


def collect(pop):
    """One row per fittable segment, plus per-robot aggregates."""
    rows, per_robot = [], []
    for r in pop:
        tot = r["total_mass_kg"]
        agg = defaultdict(float)
        for s in r["segments"]:
            if s["seg_class"] not in FIT_CLASSES or not s["motor_known"]:
                continue
            if s["mass"] <= 1e-6:
                continue
            agg[s["seg_class"]] += s["mass"]
            agg["struct_" + s["seg_class"]] += max(s["m_struct"], 0.0)
            if not knee_drive_matches(r, s["seg_class"]):
                continue
            rows.append({
                "robot": r["key"], "total_mass_kg": tot, "z_span_m": r["z_span_m"],
                "leg_length_m": r.get("leg_length_m"),
                "body_length_m": r.get("body_length_m"),
                "seg_class": s["seg_class"], "mass": s["mass"],
                "m_struct": s["m_struct"], "m_motor": s["m_motor"],
                "n_dof": s["n_dof"], "L": seg_length(s), "V": seg_volume(s),
                "cyl_r": s.get("cyl_r"),
                "struct_frac_of_seg": s["struct_frac"],
            })
        per_robot.append({
            "robot": r["key"], "total_mass_kg": tot, "z_span_m": r["z_span_m"],
            "leg_length_m": r.get("leg_length_m"),
            "body_length_m": r.get("body_length_m"),
            "motor_frac": r["motor_frac"], "n_dof": r["n_dof"],
            "class_mass": dict(agg),
        })
    return rows, per_robot


def size_rows(pop) -> list[dict]:
    """Mass + size only, for the allometry — no motor/structure split needed."""
    return [{"robot": r["key"], "total_mass_kg": r["total_mass_kg"],
             "z_span_m": r["z_span_m"], "leg_length_m": r.get("leg_length_m"),
             "body_length_m": r.get("body_length_m"),
             "decomposed": not r["fit_excluded"]}
            for r in pop]


def fit_population(rows: list[dict], per_robot: list[dict], sized: list[dict],
                   size_key: str, size_note: str) -> dict:
    """The four law families for one population.

    ``size_key`` is the allometry's size metric. ``sized`` may include robots the
    decomposition dropped, since allometry needs only mass and size.
    """
    out: dict = {}

    # ---- 1. segment mass fractions -----------------------------------------
    counts = defaultdict(lambda: defaultdict(int))
    for x in rows:
        counts[x["robot"]][x["seg_class"]] += 1
    frac, frac_struct = {}, {}
    for cls in FIT_CLASSES:
        vals, svals = [], []
        for pr in per_robot:
            m = pr["class_mass"].get(cls)
            n_inst = counts[pr["robot"]].get(cls, 0)
            if not m or not n_inst:
                continue
            # per single instance, comparable to Winter's human fractions
            vals.append(m / n_inst / pr["total_mass_kg"])
            svals.append(pr["class_mass"].get("struct_" + cls, 0.0)
                         / n_inst / pr["total_mass_kg"])
        frac[cls] = stats(vals)
        frac_struct[cls] = stats(svals)
    out["segment_mass_fraction"] = {
        "definition": "mass of ONE instance of the segment / total robot mass "
                      "(one thigh, one hand, ...), directly comparable to Winter",
        "total": frac, "structural_only": frac_struct,
        "human_reference_winter2009": HUMAN_FRACTIONS,
    }

    # trunk = torso + pelvis + head (+ waist/neck links)
    trunk = []
    for pr in per_robot:
        t = sum(pr["class_mass"].get(c, 0.0) for c in TRUNK_CLASSES)
        if t > 0:
            trunk.append(t / pr["total_mass_kg"])
    out["segment_mass_fraction"]["trunk_pelvis_torso_head"] = stats(trunk)

    # ---- 2. whole-robot allometry ------------------------------------------
    # All candidate size metrics are fit; `adopted_size_metric` names the one used.
    fits = {k: power_fit([s[k] for s in sized], [s["total_mass_kg"] for s in sized])
            for k in ("z_span_m", "leg_length_m", "body_length_m")}
    sizes = [s[size_key] for s in sized if s.get(size_key)]
    out["whole_robot"] = {
        "mass_vs_height": fits[size_key],
        "adopted_size_metric": size_key,
        # Fitted range; extrapolating a steep power law is unreliable.
        "size_range_m": [round(min(sizes), 4), round(max(sizes), 4)],
        "note": size_note,
        "mass_vs_size_candidates": fits,
        "n_size_robots": len(sized),
        "size_robots": sorted(s["robot"] for s in sized),
        "actuator_mass_fraction": stats([pr["motor_frac"] for pr in per_robot]),
        "dof_vs_mass": power_fit([pr["total_mass_kg"] for pr in per_robot],
                                 [pr["n_dof"] for pr in per_robot]),
    }

    # ---- 3. structural linear density --------------------------------------
    lin, by_class = {}, defaultdict(list)
    for row in rows:
        by_class[row["seg_class"]].append(row)
    # Without a neck joint the head is merged into the torso; fit the torso only
    # on robots with a separate head segment so it is not charged twice.
    head_robots = {r["robot"] for r in rows if r["seg_class"] == "head"}
    if head_robots and by_class.get("torso"):
        by_class["torso"] = [r for r in by_class["torso"] if r["robot"] in head_robots]
    for cls, rr in sorted(by_class.items()):
        good = [x for x in rr if (x["L"] or 0) > MIN_LINK_LEN_M
                and x["m_struct"] and x["m_struct"] > 0]
        lam = [x["m_struct"] / x["L"] for x in good]
        lin[cls] = {
            "lambda_kg_per_m": stats(lam),
            # Support of `m_struct_vs_L`; outside it the fit extrapolates.
            "length_range_m": ([round(min(x["L"] for x in good), 4),
                                round(max(x["L"] for x in good), 4)] if good else None),
            "m_struct_vs_L": power_fit([x["L"] for x in good],
                                       [x["m_struct"] for x in good],
                                       min_exp=STRUCT_MIN_EXP),
            # `m_struct_vs_L` is what the generator uses (`LinkTrends.structural_mass`);
            # this is the exponent-1 comparison, m = c L.
            "lambda_linear": linear_fit([x["L"] for x in good],
                                        [x["m_struct"] for x in good]),
            "m_struct_vs_total_mass": power_fit([x["total_mass_kg"] for x in good],
                                                [x["m_struct"] for x in good]),
            "lambda_vs_total_mass": power_fit([x["total_mass_kg"] for x in good], lam),
            "L_vs_height": power_fit([x[size_key] for x in good],
                                     [x["L"] for x in good]),
        }
    out["structural_linear_density"] = lin

    # ---- 4. effective structural density -----------------------------------
    dens = {}
    for cls, rr in sorted(by_class.items()):
        good = [x for x in rr if x["V"] and x["m_struct"] and x["m_struct"] > 0]
        dens[cls] = {
            "rho_eff_kg_m3": stats([x["m_struct"] / x["V"] for x in good]),
            "rho_total_kg_m3": stats([x["mass"] / x["V"] for x in good if x["V"]]),
            "m_struct_vs_V": power_fit([x["V"] for x in good],
                                       [x["m_struct"] for x in good]),
        }
    out["effective_density"] = dens

    # ---- 5. distal taper ----------------------------------------------------
    # `lambda_struct` is adopted (generated links are structure only);
    # `lambda_total` and the rho tapers are reported for comparison.
    QUANTS = {
        "lambda_struct": lambda r: (r["m_struct"] / r["L"]
                                    if (r["L"] or 0) > MIN_LINK_LEN_M
                                    and r["m_struct"] > 0 else None),
        "lambda_total": lambda r: (r["mass"] / r["L"]
                                   if (r["L"] or 0) > MIN_LINK_LEN_M else None),
        "rho_struct": lambda r: (r["m_struct"] / r["V"]
                                 if r["V"] and r["m_struct"] > 0 else None),
        "rho_total": lambda r: r["mass"] / r["V"] if r["V"] else None,
    }
    taper = {}
    for chain_name, classes in LIMB_CHAINS.items():
        fits = {q: within_robot_taper(rows, classes, f) for q, f in QUANTS.items()}
        # Proximal pair only (the falloff flattens at the foot/hand).
        fits_2 = {q: within_robot_taper(rows, classes[:2], f)
                  for q, f in QUANTS.items()}
        if any(fits.values()) or any(fits_2.values()):
            taper[chain_name] = {"full_chain": fits, "proximal_pair": fits_2,
                                 "classes": classes}
    out["distal_taper"] = {
        "definition": "per-chain-step multiplicative factor of a within-robot "
                      "log-linear fit; every robot demeaned in log space first, "
                      "so this is limb SHAPE and carries no information about "
                      "the absolute level",
        "adopted": {"quantity": "lambda_struct", "chain": "leg",
                    "span": "proximal_pair",
                    "why": "lambda_struct is what a generated link emits; the "
                           "leg is the only chain with a gradient the data "
                           "resolves (the arm's CI touches 1.0)"},
        "chains": taper,
    }
    return out


def main() -> None:
    payload = json.loads((DATA / "decomposition.json").read_text())
    robots = payload["robots"]
    usable = [r for r in robots if not r["fit_excluded"]]
    humanoids = [r for r in usable if not r["is_quadruped"]]
    quads = [r for r in usable if r["is_quadruped"]]
    # Allometry-only sets: decomposition not required (see stage 4).
    h_sized = size_rows([r for r in robots
                         if not r["size_excluded"] and not r["is_quadruped"]])
    q_sized = size_rows([r for r in robots
                         if not r["size_excluded"] and r["is_quadruped"]])
    print(f"{len(humanoids)} humanoids + {len(quads)} quadrupeds decomposed; "
          f"{len(h_sized)} + {len(q_sized)} usable for allometry\n")

    rows, per_robot = collect(humanoids)
    q_rows, q_per_robot = collect(quads)

    laws = {
        "provenance": {
            "n_robots_total": len(robots),
            "n_usable_humanoids": len(humanoids),
            "n_usable_quadrupeds": len(quads),
            "n_segments": len(rows),
            "motor_rules": payload["rules"],
            "robots": sorted(r["key"] for r in humanoids),
        }
    }

    laws.update(fit_population(
        rows, per_robot, h_sized, "z_span_m",
        "z_span is the neutral-pose vertical span of the joint frames, "
        "a lower bound on standing height (excludes head crown and sole)"))

    # Quadrupeds: same laws, sized by leg length (z_span depends on the neutral pose).
    laws["quadruped"] = fit_population(
        q_rows, q_per_robot, q_sized, "leg_length_m",
        "leg_length is thigh + shank, medianed over the legs — a quadruped's "
        "z_span is a neutral-pose artifact, not a size")
    laws["quadruped"]["robots"] = sorted(r["key"] for r in quads)
    laws["quadruped"]["n_robots"] = len(quads)

    write_json(TRENDS_DATA / "link_trends.json", laws, indent=1)
    write_json(DATA / "segment_rows.json",
               {"humanoid": rows, "quadruped": q_rows, "per_robot": per_robot,
                "quadruped_per_robot": q_per_robot}, indent=1)

    report("HUMANOID", laws, "H")
    report("QUADRUPED", laws["quadruped"], "L_leg")
    print(f"\nwrote {TRENDS_DATA / 'link_trends.json'}")


def report(title: str, laws: dict, size_symbol: str) -> None:
    frac = laws["segment_mass_fraction"]["total"]
    frac_struct = laws["segment_mass_fraction"]["structural_only"]
    lin = laws["structural_linear_density"]
    dens = laws["effective_density"]

    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")
    print("SEGMENT MASS FRACTION (per instance, fraction of total robot mass)")
    print(f"{'class':12s} {'n':>3s} {'median':>8s} {'p10':>8s} {'p90':>8s} "
          f"{'struct':>8s} {'human':>8s}")
    for cls in FIT_CLASSES:
        f, sf = frac.get(cls), frac_struct.get(cls)
        if not f:
            continue
        h = HUMAN_FRACTIONS.get(cls)
        print(f"{cls:12s} {f['n']:3d} {f['median']:8.4f} {f['p10']:8.4f} "
              f"{f['p90']:8.4f} {sf['median']:8.4f} "
              f"{(f'{h:.4f}' if h else '-'):>8s}")
    t = laws["segment_mass_fraction"]["trunk_pelvis_torso_head"]
    if t:
        print(f"{'trunk':12s} {t['n']:3d} {t['median']:8.4f} {t['p10']:8.4f} "
              f"{t['p90']:8.4f} {'':8s} {HUMAN_FRACTIONS['torso+pelvis+head']:8.4f}")

    print("\nWHOLE ROBOT")
    wr = laws["whole_robot"]
    mh = wr["mass_vs_height"]
    print(f"  mass = {mh['coef']:.2f} * {size_symbol}^{mh['exp']:.2f}  "
          f"(95% CI [{mh['exp_ci95'][0]:.2f}, {mh['exp_ci95'][1]:.2f}], "
          f"R2={mh['r2']:.3f}, n={mh['n']}, sigma_log={mh['sigma_log']:.3f}, "
          f"on {wr['adopted_size_metric']})")
    for k, f in wr["mass_vs_size_candidates"].items():
        if f and k != wr["adopted_size_metric"]:
            print(f"    alternative {k:14s} R2={f['r2']:.3f} exp={f['exp']:.2f}")
    af = wr["actuator_mass_fraction"]
    print(f"  actuator mass fraction: median {af['median']:.1%}, "
          f"p10-p90 {af['p10']:.1%}-{af['p90']:.1%}, n={af['n']}")

    print("\nSTRUCTURAL LINEAR DENSITY  lambda = m_struct / L   [kg/m]")
    print(f"{'class':12s} {'n':>3s} {'median':>8s} {'p10':>8s} {'p90':>8s}   "
          f"lambda ~ M_total^b")
    for cls in FIT_CLASSES:
        d = lin.get(cls)
        if not d or not d["lambda_kg_per_m"]:
            continue
        s, f = d["lambda_kg_per_m"], d["lambda_vs_total_mass"]
        tail = (f"b={f['exp']:+.3f}+-{f['exp_se']:.3f} R2={f['r2']:.2f}" if f else "-")
        print(f"{cls:12s} {s['n']:3d} {s['median']:8.3f} {s['p10']:8.3f} "
              f"{s['p90']:8.3f}   {tail}")

    print("\nEFFECTIVE STRUCTURAL DENSITY  m_struct / V_inertia   [kg/m3]")
    print(f"{'class':12s} {'n':>3s} {'median':>8s} {'p10':>8s} {'p90':>8s} {'total':>8s}")
    for cls in FIT_CLASSES:
        d = dens.get(cls)
        if not d or not d["rho_eff_kg_m3"]:
            continue
        s, st = d["rho_eff_kg_m3"], d["rho_total_kg_m3"]
        print(f"{cls:12s} {s['n']:3d} {s['median']:8.0f} {s['p10']:8.0f} "
              f"{s['p90']:8.0f} {st['median']:8.0f}")


if __name__ == "__main__":
    sys.exit(main())
