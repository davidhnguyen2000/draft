#!/usr/bin/env python3
"""Stage 6: leave-one-out prediction of real robots' total mass.

Refit every trend with robot k held out, predict k's mass from its inputs only
(height, link lengths, inertia-implied volumes, joint effort limits), and score
against the description's inertials. Predictors:

  null        training-median mass (no inputs)
  allometry   M = c * H^b
  struct_L    sum of per-class m_struct = c L^e, plus actuator mass from efforts
  struct_V    same, with structural mass = effective density x inertia volume
  struct_LV   trunk by volume, other segments by length (the generator's split)
  act_budget  M = M_actuator / median actuator fraction

Total mass is ground truth; the motor/structure split is a stage-4 estimate, so
per-class structural numbers are indicative only. Quadrupeds are scored twice:
with the humanoid trends (off-population) and with their own LOO.

Output: `data/loo_prediction.json` and a report on stdout.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from collections import defaultdict

import numpy as np

from paths import DATA, ROOT, TRENDS_DATA  # noqa: E402
from draft.jsonio import write_json  # noqa: E402
SCRIPTS = ROOT / "scripts"


def load_stage(name: str):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), SCRIPTS / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S5 = load_stage("05_fit_trends.py")

FIT_CLASSES = S5.FIT_CLASSES
LIMB_CLASSES = S5.LIMB_CLASSES
TRUNK_CLASSES = S5.TRUNK_CLASSES

PREDICTORS = ["null", "allometry", "struct_L", "struct_V", "struct_LV", "act_budget"]

#: The trunk the generator prices by volume (its root mesh); every other
#: segment is priced by length. `struct_LV` is that split.
TRUNK_BY_VOLUME = ("pelvis", "torso")

# Below this implied density the inertia tensor is a placeholder, not a solid
# body; such segments fall back to the length-based law.
MIN_IMPLIED_DENSITY = 50.0   # kg/m3

_dropped_volumes: list[dict] = []


def checked_volume(seg: dict, cls: str, robot: str) -> float | None:
    """`S5.seg_volume`, with the placeholder-tensor guard applied."""
    V = S5.seg_volume(seg)
    if V is None:
        return None
    if seg.get("inertia_valid") is False or seg["mass"] / V < MIN_IMPLIED_DENSITY:
        _dropped_volumes.append({"robot": robot, "seg_class": cls,
                                 "mass_kg": round(seg["mass"], 4),
                                 "volume_m3": round(V, 5),
                                 "implied_density": round(seg["mass"] / V, 2)})
        return None
    return V


# ---------------------------------------------------------------------------
# per-robot input extraction
# ---------------------------------------------------------------------------

def prep(rec: dict) -> dict:
    """Everything a predictor is allowed to see, plus the ground truth.

    `fit_segments` are those stage 5 fits on; `all_segments` is every classified
    segment, which a bottom-up prediction must sum over.
    """
    fit_segs, all_segs = [], []
    for s in rec["segments"]:
        cls = s.get("seg_class")
        if cls not in FIT_CLASSES or s["mass"] <= 1e-6:
            continue
        row = {
            "seg_class": cls,
            "mass": s["mass"],
            "m_struct": s.get("m_struct"),
            "motor_known": bool(s.get("motor_known")),
            "L": S5.seg_length(s),
            "V": checked_volume(s, cls, rec["key"]),
        }
        all_segs.append(row)
        if (row["motor_known"] and row["m_struct"] is not None and row["m_struct"] > 0
                and S5.knee_drive_matches(rec, cls)):
            fit_segs.append(row)

    covered = sum(s["mass"] for s in all_segs)
    counts = defaultdict(int)
    for s in all_segs:
        counts[s["seg_class"]] += 1
    return {
        "key": rec["key"],
        "robot": rec.get("robot", rec["key"]),
        "H": rec["z_span_m"],
        "M_true": rec["total_mass_kg"],           # ground truth, never fitted on
        "M_act": rec["motor_mass_kg"],            # from the datasets/actuators law (external)
        "motor_frac": rec["motor_frac"],
        "trunk_frac": sum(s["mass"] for s in all_segs
                          if s["seg_class"] in TRUNK_CLASSES) / rec["total_mass_kg"],
        "class_frac": {c: sum(s["mass"] for s in all_segs if s["seg_class"] == c)
                          / counts[c] / rec["total_mass_kg"] for c in counts},
        "coverage": covered / rec["total_mass_kg"],
        "fit_segments": fit_segs,
        "all_segments": all_segs,
    }


# ---------------------------------------------------------------------------
# fitting on a training set
# ---------------------------------------------------------------------------

def fit_laws(train: list[dict]) -> dict:
    """Refit every trend this stage predicts with, on `train` only."""
    allom = S5.power_fit([r["H"] for r in train], [r["M_true"] for r in train])

    by_class = defaultdict(list)
    for r in train:
        for s in r["fit_segments"]:
            by_class[s["seg_class"]].append(s)

    struct_vs_L, lam_med, rho_med = {}, {}, {}
    for cls, rows in by_class.items():
        with_L = [s for s in rows if (s["L"] or 0) > S5.MIN_LINK_LEN_M]
        with_V = [s for s in rows if s["V"]]
        if with_L:
            # m = c L^e, the power fit the generator sizes members from.
            struct_vs_L[cls] = S5.power_fit([s["L"] for s in with_L],
                                            [s["m_struct"] for s in with_L],
                                            min_exp=S5.STRUCT_MIN_EXP)
            lam_med[cls] = float(np.median([s["m_struct"] / s["L"] for s in with_L]))
        if with_V:
            rho_med[cls] = float(np.median([s["m_struct"] / s["V"] for s in with_V]))

    class_frac = {}
    for cls in FIT_CLASSES:
        vals = [r["class_frac"][cls] for r in train if cls in r["class_frac"]]
        if vals:
            class_frac[cls] = float(np.median(vals))

    return {
        "allometry": allom,
        "struct_vs_L": struct_vs_L,
        "lambda_med": lam_med,
        "rho_med": rho_med,
        "M_median": float(np.median([r["M_true"] for r in train])),
        "f_act_median": float(np.median([r["motor_frac"] for r in train])),
        "trunk_frac_median": float(np.median([r["trunk_frac"] for r in train])),
        "class_frac_median": class_frac,
        # fallbacks for a class the training set happens not to contain
        "lambda_global": float(np.median([s["m_struct"] / s["L"]
                                          for rows in by_class.values() for s in rows
                                          if (s["L"] or 0) > S5.MIN_LINK_LEN_M])),
        "rho_global": float(np.median([s["m_struct"] / s["V"]
                                       for rows in by_class.values() for s in rows
                                       if s["V"]])),
    }


# ---------------------------------------------------------------------------
# predictors
# ---------------------------------------------------------------------------

def predict_struct_L(target: dict, laws: dict) -> tuple[float, list[float]]:
    """Bottom-up structural mass from per-class m_struct = c L^e."""
    per_seg = []
    for s in target["all_segments"]:
        cls, L, V = s["seg_class"], s["L"], s["V"]
        fit = laws["struct_vs_L"].get(cls)
        has_L = (L or 0) > S5.MIN_LINK_LEN_M
        if has_L and fit:
            m = fit["coef"] * L ** fit["exp"]
        elif has_L:
            m = laws["lambda_med"].get(cls, laws["lambda_global"]) * L
        elif V:
            m = laws["rho_med"].get(cls, laws["rho_global"]) * V
        else:
            m = 0.0
        per_seg.append(m)
    return float(sum(per_seg)), per_seg


def predict_struct_LV(target: dict, laws: dict) -> tuple[float, list[float]]:
    """The generator's split: the trunk by effective density x volume, every
    other segment by its class's m = c L^e."""
    per_seg = []
    for s in target["all_segments"]:
        cls, L, V = s["seg_class"], s["L"], s["V"]
        if cls in TRUNK_BY_VOLUME and V:
            per_seg.append(laws["rho_med"].get(cls, laws["rho_global"]) * V)
        else:
            per_seg.append(predict_struct_L({"all_segments": [s]}, laws)[0])
    return float(sum(per_seg)), per_seg


def predict_struct_V(target: dict, laws: dict) -> tuple[float, list[float]]:
    """Bottom-up structural mass from per-class effective density x volume."""
    per_seg = []
    for s in target["all_segments"]:
        cls, L, V = s["seg_class"], s["L"], s["V"]
        if V:
            m = laws["rho_med"].get(cls, laws["rho_global"]) * V
        elif (L or 0) > S5.MIN_LINK_LEN_M:
            m = laws["lambda_med"].get(cls, laws["lambda_global"]) * L
        else:
            m = 0.0
        per_seg.append(m)
    return float(sum(per_seg)), per_seg


def predict_all(target: dict, laws: dict) -> dict:
    a = laws["allometry"]
    sL, _ = predict_struct_L(target, laws)
    sV, _ = predict_struct_V(target, laws)
    sLV, _ = predict_struct_LV(target, laws)
    return {
        "null": laws["M_median"],
        "allometry": a["coef"] * target["H"] ** a["exp"] if a else float("nan"),
        "struct_L": sL + target["M_act"],
        "struct_V": sV + target["M_act"],
        "struct_LV": sLV + target["M_act"],
        "act_budget": target["M_act"] / laws["f_act_median"],
    }


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def score(pred: list[float], actual: list[float]) -> dict:
    """Log-space (fold) error metrics; robot masses span decades."""
    p = np.asarray(pred, float)
    a = np.asarray(actual, float)
    ok = np.isfinite(p) & np.isfinite(a) & (p > 0) & (a > 0)
    p, a = p[ok], a[ok]
    if len(p) < 2:
        return {"n": int(len(p))}
    lr = np.log(p / a)
    la = np.log(a)
    ss_res = float(np.sum(lr ** 2))
    ss_tot = float(np.sum((la - la.mean()) ** 2))
    return {
        "n": int(len(p)),
        "bias_fold": round(float(np.exp(np.median(lr))), 4),      # >1 = over-predicts
        "gmfe": round(float(np.exp(np.mean(np.abs(lr)))), 4),     # geometric mean fold error
        "median_abs_fold": round(float(np.exp(np.median(np.abs(lr)))), 4),
        "sigma_log": round(float(np.std(lr)), 4),
        "within_1p25": round(float(np.mean(np.abs(lr) < math.log(1.25))), 4),
        "within_1p5": round(float(np.mean(np.abs(lr) < math.log(1.5))), 4),
        "within_2x": round(float(np.mean(np.abs(lr) < math.log(2.0))), 4),
        "max_fold": round(float(np.exp(np.max(np.abs(lr)))), 4),
        "r2_log": round(1.0 - ss_res / ss_tot, 4) if ss_tot > 0 else None,
    }


def metric_stability(rows: list[dict], preds: dict, actual: list[float]) -> dict:
    """Scores per mass band, showing why GMFE is used rather than R^2: R^2
    collapses on a narrow band while the predictions get no worse."""
    bands = [("all (3.1-136 kg)", lambda m: True),
             (">= 20 kg", lambda m: m >= 20.0),
             ("30-90 kg", lambda m: 30.0 <= m <= 90.0)]
    out = {}
    for label, keep in bands:
        idx = [i for i, m in enumerate(actual) if keep(m)]
        out[label] = {k: score([preds[k][i] for i in idx], [actual[i] for i in idx])
                      for k in PREDICTORS if k != "null"}
        out[label]["n"] = len(idx)
    return out


def loo_scalar(rows: list[dict], field: str) -> dict:
    """LOO of a per-robot ratio by the training median (population homogeneity)."""
    pred, act = [], []
    for i, r in enumerate(rows):
        train = rows[:i] + rows[i + 1:]
        pred.append(float(np.median([t[field] for t in train])))
        act.append(r[field])
    s = score(pred, act)
    s["per_robot"] = [{"robot": r["key"], "predicted": round(p, 5),
                       "actual": round(a, 5)} for r, p, a in zip(rows, pred, act)]
    return s


def main() -> None:
    payload = json.loads((DATA / "decomposition.json").read_text())
    usable = [r for r in payload["robots"] if not r["fit_excluded"]]
    humanoids = [prep(r) for r in usable if not r["is_quadruped"]]
    quads = [prep(r) for r in usable if r["is_quadruped"]]
    humanoids.sort(key=lambda r: r["M_true"])

    out: dict = {"provenance": {
        "n_humanoids": len(humanoids), "n_quadrupeds": len(quads),
        "robots": [r["key"] for r in humanoids],
        "ground_truth": "total_mass_kg from the official description inertials",
        "external_inputs": "actuator mass from datasets/actuators's law (never refit here)",
    }}

    print("=" * 84)
    print("LEAVE-ONE-OUT TOTAL MASS PREDICTION — 21 humanoids, laws refit without each")
    print("=" * 84)
    cov = [r["coverage"] for r in humanoids]
    print(f"  classified-segment mass coverage: min {min(cov):.3f}  "
          f"median {np.median(cov):.3f}  (1.000 = every kg is in a fitted class)")
    out["dropped_volumes"] = _dropped_volumes
    if _dropped_volumes:
        print(f"  placeholder inertia tensors dropped from the volume predictor: "
              f"{len(_dropped_volumes)}")
        for d in _dropped_volumes:
            print(f"    {d['robot']:26s} {d['seg_class']:12s} "
                  f"{d['implied_density']:8.1f} kg/m3 implied")
    print()

    # ---- humanoid total-mass LOO --------------------------------------------
    preds = {k: [] for k in PREDICTORS}
    per_robot = []
    for i, tgt in enumerate(humanoids):
        train = humanoids[:i] + humanoids[i + 1:]
        laws = fit_laws(train)
        p = predict_all(tgt, laws)
        for k in PREDICTORS:
            preds[k].append(p[k])
        per_robot.append({"robot": tgt["key"], "H_m": round(tgt["H"], 4),
                          "M_true_kg": round(tgt["M_true"], 3),
                          "M_act_kg": round(tgt["M_act"], 3),
                          "predicted_kg": {k: round(v, 3) for k, v in p.items()}})

    actual = [r["M_true"] for r in humanoids]
    scores = {k: score(preds[k], actual) for k in PREDICTORS}
    out["total_mass"] = {"scores": scores, "per_robot": per_robot}

    print(f"  {'predictor':11s} {'inputs used':26s} {'GMFE':>6s} {'bias':>7s} "
          f"{'<1.25x':>7s} {'<1.5x':>7s} {'<2x':>6s} {'worst':>7s} {'R2_log':>7s}")
    labels = {
        "null": "(none)",
        "allometry": "height",
        "struct_L": "link lengths + torques",
        "struct_V": "segment volumes + torques",
        "struct_LV": "lengths, trunk volume + torques",
        "act_budget": "torques",
    }
    for k in PREDICTORS:
        s = scores[k]
        r2 = "-" if s["r2_log"] is None else f"{s['r2_log']:.3f}"
        print(f"  {k:11s} {labels[k]:26s} {s['gmfe']:6.3f} {s['bias_fold']:7.3f} "
              f"{s['within_1p25']:7.0%} {s['within_1p5']:7.0%} {s['within_2x']:6.0%} "
              f"{s['max_fold']:7.2f} {r2:>7s}")

    print(f"\n  {'robot':22s} {'H':>5s} {'true':>7s} " +
          " ".join(f"{k:>9s}" for k in PREDICTORS))
    for row in per_robot:
        print(f"  {row['robot'][:22]:22s} {row['H_m']:5.2f} {row['M_true_kg']:7.2f} " +
              " ".join(f"{row['predicted_kg'][k]:9.2f}" for k in PREDICTORS))

    # ---- why GMFE and not R^2 -----------------------------------------------
    stab = metric_stability(humanoids, preds, actual)
    out["metric_stability"] = stab
    print()
    print("=" * 84)
    print("METRIC CHOICE — GMFE vs R^2 as the test population narrows")
    print("=" * 84)
    keys = [k for k in PREDICTORS if k != "null"]
    print(f"  {'mass band':18s} {'n':>3s} " +
          " ".join(f"{k:>21s}" for k in keys))
    print(f"  {'':18s} {'':>3s} " + " ".join(f"{'GMFE':>10s}{'R2':>11s}" for k in keys))
    for label, row in stab.items():
        cells = " ".join(f"{row[k]['gmfe']:10.3f}{row[k]['r2_log']:11.3f}" for k in keys)
        print(f"  {label:18s} {row['n']:3d} {cells}")
    print("  R^2 halves as the population narrows while GMFE holds or improves:")
    print("  R^2 is scoring the spread of the test set, GMFE is scoring the model.")

    # ---- per-class structural mass ------------------------------------------
    print()
    print("=" * 84)
    print("LEAVE-ONE-OUT SEGMENT STRUCTURAL MASS — per class")
    print("=" * 84)
    print("  (structural mass is itself a stage-4 ESTIMATE, biased ~18% low; read")
    print("   these as indicative of which classes the length law can carry, not")
    print("   as ground truth the way the total mass above is.)\n")

    seg_pred = defaultdict(lambda: {"L": [], "V": [], "act": []})
    for i, tgt in enumerate(humanoids):
        train = humanoids[:i] + humanoids[i + 1:]
        laws = fit_laws(train)
        _, pL = predict_struct_L(tgt, laws)
        _, pV = predict_struct_V(tgt, laws)
        for s, mL, mV in zip(tgt["all_segments"], pL, pV):
            if not (s["motor_known"] and s["m_struct"] and s["m_struct"] > 0):
                continue
            d = seg_pred[s["seg_class"]]
            d["L"].append(mL)
            d["V"].append(mV)
            d["act"].append(s["m_struct"])

    seg_scores = {}
    print(f"  {'class':16s} {'n':>3s} {'GMFE_L':>7s} {'bias_L':>7s} "
          f"{'GMFE_V':>7s} {'bias_V':>7s}")
    for cls in FIT_CLASSES:
        d = seg_pred.get(cls)
        if not d or len(d["act"]) < 4:
            continue
        sL, sV = score(d["L"], d["act"]), score(d["V"], d["act"])
        seg_scores[cls] = {"struct_L": sL, "struct_V": sV}
        print(f"  {cls:16s} {sL['n']:3d} {sL['gmfe']:7.3f} {sL['bias_fold']:7.3f} "
              f"{sV['gmfe']:7.3f} {sV['bias_fold']:7.3f}")
    out["segment_structural_mass"] = seg_scores

    # ---- dimensionless per-robot quantities ---------------------------------
    print()
    print("=" * 84)
    print("LEAVE-ONE-OUT DIMENSIONLESS QUANTITIES — predicted by the training median")
    print("=" * 84)
    dimless = {}
    for field, label in (("motor_frac", "actuator mass fraction"),
                         ("trunk_frac", "trunk mass fraction")):
        s = loo_scalar(humanoids, field)
        dimless[field] = s
        print(f"  {label:24s} GMFE {s['gmfe']:.3f}  within 1.25x {s['within_1p25']:.0%}"
              f"  worst {s['max_fold']:.2f}x")
    # per-class mass fraction
    print()
    print(f"  {'class mass fraction':24s} {'n':>3s} {'GMFE':>7s} {'<1.25x':>7s} {'worst':>7s}")
    for cls in FIT_CLASSES:
        rows = [r for r in humanoids if cls in r["class_frac"]]
        if len(rows) < 5:
            continue
        pred, act = [], []
        for i, r in enumerate(rows):
            train = rows[:i] + rows[i + 1:]
            pred.append(float(np.median([t["class_frac"][cls] for t in train])))
            act.append(r["class_frac"][cls])
        s = score(pred, act)
        dimless[f"class_frac_{cls}"] = s
        print(f"  {cls:24s} {s['n']:3d} {s['gmfe']:7.3f} {s['within_1p25']:7.0%} "
              f"{s['max_fold']:7.2f}")
    out["dimensionless"] = dimless

    # ---- off-population holdout: quadrupeds ---------------------------------
    print()
    print("=" * 84)
    print("OFF-POPULATION HOLDOUT — humanoid laws applied to quadrupeds")
    print("=" * 84)
    full = fit_laws(humanoids)
    q_pred = {k: [] for k in PREDICTORS}
    q_rows = []
    for tgt in sorted(quads, key=lambda r: r["M_true"]):
        p = predict_all(tgt, full)
        for k in PREDICTORS:
            q_pred[k].append(p[k])
        q_rows.append({"robot": tgt["key"], "M_true_kg": round(tgt["M_true"], 3),
                       "predicted_kg": {k: round(v, 3) for k, v in p.items()}})
    q_actual = [r["M_true"] for r in sorted(quads, key=lambda r: r["M_true"])]
    q_scores = {k: score(q_pred[k], q_actual) for k in PREDICTORS}
    out["quadruped_holdout"] = {"scores": q_scores, "per_robot": q_rows}
    print(f"  {'predictor':11s} {'GMFE':>6s} {'bias':>7s} {'<1.5x':>7s} {'<2x':>6s}  (n={len(q_actual)})")
    for k in PREDICTORS:
        s = q_scores[k]
        if not s.get("gmfe"):
            continue
        print(f"  {k:11s} {s['gmfe']:6.3f} {s['bias_fold']:7.3f} "
              f"{s['within_1p5']:7.0%} {s['within_2x']:6.0%}")

    # ---- quadruped leave-one-out --------------------------------------------
    # Score the quadruped trends themselves: refit without each machine.
    print()
    print("=" * 84)
    print(f"LEAVE-ONE-OUT TOTAL MASS PREDICTION — {len(quads)} quadrupeds, "
          "laws refit without each")
    print("=" * 84)
    quads.sort(key=lambda r: r["M_true"])
    q_loo_pred = {k: [] for k in PREDICTORS}
    q_loo_rows = []
    for i, tgt in enumerate(quads):
        laws = fit_laws(quads[:i] + quads[i + 1:])
        p = predict_all(tgt, laws)
        for k in PREDICTORS:
            q_loo_pred[k].append(p[k])
        q_loo_rows.append({"robot": tgt["key"], "H_m": round(tgt["H"], 4),
                           "M_true_kg": round(tgt["M_true"], 3),
                           "M_act_kg": round(tgt["M_act"], 3),
                           "predicted_kg": {k: round(v, 3) for k, v in p.items()}})
    q_loo_actual = [r["M_true"] for r in quads]
    q_loo_scores = {k: score(q_loo_pred[k], q_loo_actual) for k in PREDICTORS}
    out["quadruped_total_mass"] = {"scores": q_loo_scores, "per_robot": q_loo_rows}
    print(f"  {'predictor':11s} {'inputs used':26s} {'GMFE':>6s} {'bias':>7s} "
          f"{'<1.25x':>7s} {'<1.5x':>7s} {'<2x':>6s} {'worst':>7s} {'R2_log':>7s}")
    for k in PREDICTORS:
        s_ = q_loo_scores[k]
        if not s_.get("gmfe"):
            continue
        r2 = "-" if s_["r2_log"] is None else f"{s_['r2_log']:.3f}"
        print(f"  {k:11s} {labels[k]:26s} {s_['gmfe']:6.3f} {s_['bias_fold']:7.3f} "
              f"{s_['within_1p25']:7.0%} {s_['within_1p5']:7.0%} "
              f"{s_['within_2x']:6.0%} {s_['max_fold']:7.2f} {r2:>7s}")

    write_json(DATA / "loo_prediction.json", out, indent=1)
    print(f"\nwrote {DATA / 'loo_prediction.json'}")


if __name__ == "__main__":
    sys.exit(main())
