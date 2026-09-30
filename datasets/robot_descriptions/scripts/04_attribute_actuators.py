#!/usr/bin/env python3
"""Stage 4 — split each segment's mass into actuator mass and structural mass.

Actuator mass per joint:
1. **Disclosed lineup** (`DISCLOSED`, `DISCLOSED_BY_ROLE`): the published actuator.
2. **Class law**: `m = coef * tau^exp` from the effort limit, using the adopted
   `mass_trend` in `src/draft/trends/data/actuator_trends.json` (Table I), or a
   servo law refit over `data/servo_actuators.json`.
3. **Attribution**: humanoids split each actuator 50/50 across its joint;
   quadrupeds charge it to the segment named in `QUAD_ACTUATOR_HOSTS`.

A URDF effort limit is an author-chosen operating limit, not a datasheet peak,
so `TAU_FACTORS` gives a sensitivity sweep. Output: `data/decomposition.json`.
"""

from __future__ import annotations

import json
import math
import sys

import numpy as np

from paths import DATA, ROOT, TRENDS_DATA  # noqa: E402
from draft.jsonio import write_json  # noqa: E402

# Peak-torque multipliers on URDF effort limits (1.0 = face value).
TAU_FACTORS = [1.0, 1.25, 1.5]
ADOPTED_TAU_FACTOR = 1.0

# Which end of the joint is charged for the actuator, per population. On
# quadrupeds a 50/50 split leaves most shanks with negative structural mass
# (the knee drive sits up the leg); `split_negatives` records the counts.
ADOPTED_SPLIT_BY_POP = {"humanoid": "half", "quadruped": "proximal"}

# Where each quadruped carries its leg actuators, stated per robot (not detected).
# `knee_drive`: `remotized` (knee actuator at the hip, driving via belt/rod) or
# `at_joint`. `hip_pitch` / `knee` name the segment class holding that actuator.
# `evidence` cites the thigh COM fraction or body masses. A fitted quadruped
# missing from this table stops the stage.
QUAD_ACTUATOR_HOSTS = {
    "anymal_b_description": dict(knee_drive="at_joint", hip_pitch="hip_link", knee="thigh",
        evidence="thigh COM at 0.80 of its length: the KFE drive is at the knee"),
    "anymal_c_description": dict(knee_drive="at_joint", hip_pitch="hip_link", knee="thigh",
        evidence="thigh COM at 0.77 of its length: the KFE drive is at the knee"),
    "anymal_d_description": dict(knee_drive="at_joint", hip_pitch="thigh", knee="thigh",
        evidence="hip body 0.37 kg against a 1.21 kg HFE drive, so that drive is in "
                 "the thigh body; thigh COM at 0.35 with both drives in it"),
    "deeprobotics_lite3": dict(knee_drive="remotized", hip_pitch="hip_link", knee="thigh",
        evidence="thigh COM at 0.13 of its length"),
    "deeprobotics_x30": dict(knee_drive="remotized", hip_pitch="hip_link", knee="thigh",
        evidence="thigh COM at 0.18 of its length"),
    "deeprobotics_m20": dict(knee_drive="at_joint", hip_pitch="hip_link", knee="thigh",
        evidence="thigh COM at 0.53 of its length: not a hip-mounted knee unit"),
    "deeprobotics_m20s": dict(knee_drive="at_joint", hip_pitch="hip_link", knee="thigh",
        evidence="thigh COM at 0.53 of its length: not a hip-mounted knee unit"),
    "magicdog": dict(knee_drive="remotized", hip_pitch="thigh", knee="thigh",
        evidence="hip body 0.36 kg against a 0.56 kg hip-pitch unit, so that unit is "
                 "in the thigh body; thigh COM at 0.16"),
    "unitree_a1": dict(knee_drive="remotized", hip_pitch="hip_link", knee="thigh",
        evidence="thigh COM at 0.14 of its length"),
    "unitree_go1": dict(knee_drive="remotized", hip_pitch="hip_link", knee="thigh",
        evidence="thigh COM at 0.14 of its length"),
    "unitree_aliengo": dict(knee_drive="remotized", hip_pitch="hip_link", knee="hip_link",
        evidence="thigh body 0.77 kg against a 0.82 kg knee unit, so that unit is in "
                 "the 2.14 kg hip body"),
    "unitree_a2": dict(knee_drive="remotized", hip_pitch="hip_link", knee="thigh",
        evidence="thigh COM at 0.20 of its length"),
    "unitree_go2": dict(knee_drive="remotized", hip_pitch="hip_link", knee="thigh",
        evidence="thigh COM at 0.14 of its length"),
    "unitree_b1": dict(knee_drive="remotized", hip_pitch="hip_link", knee="thigh",
        evidence="thigh COM at 0.15 of its length"),
    "unitree_b2": dict(knee_drive="remotized", hip_pitch="hip_link", knee="thigh",
        evidence="thigh COM at 0.16 of its length"),
    "unitree_laikago": dict(knee_drive="remotized", hip_pitch="hip_link", knee="thigh",
        evidence="thigh COM at 0.13 of its length"),
    "zsibot_zsl1": dict(knee_drive="remotized", hip_pitch="thigh", knee="thigh",
        evidence="hip body 0.45 kg against a 0.60 kg hip-pitch unit, so that unit is "
                 "in the thigh body; thigh COM at 0.11"),
    "dobot_rover_x1": dict(knee_drive="remotized", hip_pitch="thigh", knee="thigh",
        evidence="hip body 0.06 kg against a 0.53 kg hip-pitch unit, so that unit is "
                 "in the thigh body; thigh COM at 0.14"),
    "dfki_quad": dict(knee_drive="remotized", hip_pitch="hip_link", knee="thigh",
        evidence="thigh COM at 0.09 of its length"),
    "xiaomi_cyberdog2": dict(knee_drive="remotized", hip_pitch="hip_link", knee="thigh",
        evidence="thigh COM at 0.12 of its length"),
}
#: Joint role -> key in a `QUAD_ACTUATOR_HOSTS` entry.
QUAD_HOST_ROLE = {"q_hip": "hip_pitch", "q_knee": "knee"}


def actuator_host(seg: dict, role: str, hosts: dict | None, seg_by_idx: dict,
                  default: dict, robot: str) -> dict:
    """The segment that carries `role`'s actuator: the stated host class, found
    by walking up the leg from the joint's own segment."""
    if not hosts or role not in QUAD_HOST_ROLE:
        return default
    want = hosts[QUAD_HOST_ROLE[role]]
    cur = seg
    while cur is not None:
        if cur["seg_class"] == want:
            return cur
        cur = seg_by_idx.get(cur["parent_seg"])
    raise SystemExit(f"{robot}: QUAD_ACTUATOR_HOSTS puts the {role} actuator in a "
                     f"{want!r} segment, and the leg above that joint has none")

# Efforts above this are placeholders (some descriptions ship 1e9), not specs.
EFFORT_CAP_NM = 1000.0

# Implied actuator mass fraction above this means the effort column is not a
# peak-torque proxy; the robot is dropped from the structural fit.
MOTOR_FRAC_CEILING = 0.65

# Robots built from hobby smart servos rather than geared BLDC modules.
SERVO_ROBOTS = {"toddlerbot_description", "op3_mj_description", "sigmaban_description",
                "nao_v50", "romeo_description"}

# Published actuator line-ups: (label, peak_torque_Nm, mass_kg, source).
DISCLOSED = {
    "berkeley_humanoid_description": [
        ("5013",  9.7, 0.251, "Berkeley Humanoid paper, arXiv:2407.21781 Table I"),
        ("8513", 45.3, 0.756, "Berkeley Humanoid paper, arXiv:2407.21781 Table I"),
        ("8518", 62.6, 0.856, "Berkeley Humanoid paper, arXiv:2407.21781 Table I"),
        ("10413", 81.1, 1.011, "Berkeley Humanoid paper, arXiv:2407.21781 Table I"),
    ],
}

# Published per-joint actuators (effort column not consulted). `*` = fallback role.
DISCLOSED_BY_ROLE = {
    "op3_mj_description": {
        "_source": "ROBOTIS OP3 spec: 20x XM430-W350 (82 g) at every joint",
        "*": ("XM430-W350", 0.082),
    },
    "toddlerbot_description": {
        "_source": "ToddlerBot paper (arXiv:2502.00893) motor-selection section + "
                   "ROBOTIS e-Manual masses. XM430 on knee/ankle-pitch, 2XC430 "
                   "(dual-axis, 100 g for 2 DOF) on hip roll+pitch, XC330 elsewhere.",
        "knee": ("XM430-W350", 0.082),
        "ankle": ("XM430-W350", 0.082),
        "hip": ("2XC430-W250/2", 0.050),
        "*": ("XC330-M288", 0.023),
    },
}

# Actuator masses the vendor states directly, used as validation anchors rather
# than substituted into the decomposition (they cover only part of the robot).
ANCHORS = {
    "h1_description": {
        "note": "Unitree M107: 360 Nm peak, 189 Nm/kg -> 1.90 kg; URDF knee effort is 300 Nm",
        "tau_peak_Nm": 360.0, "mass_kg": 360.0 / 189.0,
    },
}


def power_fit(x: np.ndarray, y: np.ndarray) -> dict:
    """OLS in log-log space: y = coef * x^exp."""
    lx, ly = np.log(x), np.log(y)
    A = np.vstack([lx, np.ones_like(lx)]).T
    (exp, log_coef), res, *_ = np.linalg.lstsq(A, ly, rcond=None)
    pred = A @ [exp, log_coef]
    ss_res = float(np.sum((ly - pred) ** 2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    return {"exp": float(exp), "coef": float(math.exp(log_coef)),
            "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
            "sigma_log": float(np.std(ly - pred, ddof=2)) if len(x) > 2 else float("nan"),
            "n": int(len(x))}


def servo_law() -> dict:
    # Servos are not in the actuator catalogue; their rows live beside this stage.
    rows = json.loads((DATA / "servo_actuators.json").read_text())["rows"]
    fit = power_fit(np.array([r["tau_peak_Nm"] for r in rows]),
                    np.array([r["mass_kg"] for r in rows]))
    fit["source"] = "refit here over datasets/robot_descriptions/data/servo_actuators.json"
    return fit


def usable_effort(tau: float) -> bool:
    return math.isfinite(tau) and 0.0 < tau <= EFFORT_CAP_NM


def effort_quality(efforts: list[float]) -> str:
    """Describe the effort column. `uniform` is informational, not a defect:
    A1 and Berkeley Lite really do use one actuator at every joint."""
    if not efforts:
        return "missing"
    good = [e for e in efforts if usable_effort(e)]
    if not good:
        return "missing"
    if any(math.isfinite(e) and e > EFFORT_CAP_NM for e in efforts):
        return "placeholder"
    if len(good) < len(efforts):
        return "partial"
    if len(set(round(e, 6) for e in good)) == 1 and len(good) > 3:
        return "uniform"
    return "ok"


def pick_disclosed(lineup: list, tau: float):
    """Lightest lineup member whose peak torque covers the joint's effort limit."""
    covering = [a for a in lineup if a[1] >= tau]
    pool = covering or lineup
    return min(pool, key=lambda a: a[2])


def main() -> None:
    robots = json.loads((DATA / "segments_classified.json").read_text())
    laws = json.loads((TRENDS_DATA / "actuator_trends.json").read_text())
    mt = laws["mass_trend"]                         # the adopted Table I law
    geared = {"coef": mt["coef"], "exp": mt["tau_exp"], "r2": mt["r2"], "n": mt["n"]}
    servo = servo_law()
    print(f"geared law   m = {geared['coef']:.4f} * tau^{geared['exp']:.4f}   "
          f"(R2={geared['r2']:.3f}, n={geared['n']}, datasets/actuators)")
    print(f"servo law    m = {servo['coef']:.4f} * tau^{servo['exp']:.4f}   "
          f"(R2={servo['r2']:.3f}, n={servo['n']}, refit)\n")

    out = []
    for r in robots:
        efforts = [j["effort"] for s in r["segments"] for j in s["joints"]
                   if j["role"] not in ("aux",)]
        r["effort_quality"] = effort_quality(efforts)
        r["motor_class"] = "servo" if r["key"] in SERVO_ROBOTS else "geared"
        law = servo if r["motor_class"] == "servo" else geared
        lineup = DISCLOSED.get(r["key"])
        by_role = DISCLOSED_BY_ROLE.get(r["key"])
        r["motor_source"] = "disclosed" if (lineup or by_role) else "law"
        r["anchor"] = ANCHORS.get(r["key"])

        seg_by_idx = {s["idx"]: s for s in r["segments"]}
        hosts = QUAD_ACTUATOR_HOSTS.get(r["key"]) if r["is_quadruped"] else None
        r["knee_drive"] = hosts["knee_drive"] if hosts else None
        r["actuator_hosts"] = ({k: hosts[k] for k in ("hip_pitch", "knee", "evidence")}
                               if hosts else None)
        for s in r["segments"]:
            for key in ("m_motor_half", "m_motor_proximal", "m_motor_distal"):
                s.setdefault(key, {f: 0.0 for f in TAU_FACTORS})
            # Unknown actuators make the structural residual unknown, not motor-free.
            s.setdefault("n_unknown", {"half": 0, "proximal": 0, "distal": 0})

        for s in r["segments"]:
            parent = seg_by_idx.get(s["parent_seg"])
            proximal_tgt = parent if parent is not None else s
            for j in s["joints"]:
                if j["role"] in ("aux", "digit"):
                    j["m_motor"] = None      # not part of the limb mass budget
                    continue
                host_tgt = actuator_host(s, j["role"], hosts, seg_by_idx,
                                         proximal_tgt, r["key"])
                if by_role:
                    # Published assignment; tau sensitivity is degenerate.
                    label, mass = by_role.get(j["role"], by_role["*"])
                    j["effort_usable"] = True
                    j["motor_label"] = label
                    j["m_motor"] = mass
                    j["m_motor_by_factor"] = {f: mass for f in TAU_FACTORS}
                    for f in TAU_FACTORS:
                        s["m_motor_distal"][f] += mass
                        host_tgt["m_motor_proximal"][f] += mass
                        s["m_motor_half"][f] += 0.5 * mass
                        proximal_tgt["m_motor_half"][f] += 0.5 * mass
                    continue
                if not usable_effort(j["effort"]):
                    j["m_motor"] = None
                    j["effort_usable"] = False
                    s["n_unknown"]["distal"] += 1
                    host_tgt["n_unknown"]["proximal"] += 1
                    for tgt in (s, proximal_tgt):
                        tgt["n_unknown"]["half"] += 1
                    continue
                j["effort_usable"] = True
                per_factor = {}
                for f in TAU_FACTORS:
                    tau = j["effort"] * f
                    per_factor[f] = (pick_disclosed(lineup, tau)[2] if lineup
                                     else law["coef"] * tau ** law["exp"])
                j["m_motor"] = per_factor[ADOPTED_TAU_FACTOR]
                j["m_motor_by_factor"] = per_factor
                # Accumulate under all three splits.
                for f, m in per_factor.items():
                    s["m_motor_distal"][f] += m
                    host_tgt["m_motor_proximal"][f] += m
                    s["m_motor_half"][f] += 0.5 * m
                    proximal_tgt["m_motor_half"][f] += 0.5 * m

        # Structural residual under the adopted rule.
        f = ADOPTED_TAU_FACTOR
        split_adopted = ADOPTED_SPLIT_BY_POP[
            "quadruped" if r["is_quadruped"] else "humanoid"]
        r["adopted_split"] = split_adopted
        for s in r["segments"]:
            for split in ("half", "proximal", "distal"):
                known = s["n_unknown"][split] == 0
                s[f"m_struct_{split}"] = (s["mass"] - s[f"m_motor_{split}"][f]
                                          if known else None)
            s["motor_known"] = s["n_unknown"][split_adopted] == 0
            s["m_motor"] = s[f"m_motor_{split_adopted}"][f]
            s["m_struct"] = s[f"m_struct_{split_adopted}"]
            s["struct_frac"] = (s["m_struct"] / s["mass"]
                                if s["motor_known"] and s["mass"] > 0 else None)
        # Negative-residual counts per split: the evidence for the adopted split.
        r["split_negatives"] = {
            split: sum(1 for s in r["segments"]
                       if s["mass"] > 1e-6 and s[f"m_struct_{split}"] is not None
                       and s[f"m_struct_{split}"] < 0)
            for split in ("half", "proximal", "distal")}

        tot = r["total_mass_kg"]
        dofs = [j for s in r["segments"] for j in s["joints"]
                if j["role"] not in ("aux", "digit")]
        known_dofs = [j for j in dofs if j.get("effort_usable")]
        r["dof_known_frac"] = len(known_dofs) / len(dofs) if dofs else 0.0
        mot = sum(s["m_motor"] for s in r["segments"])
        r["motor_mass_kg"] = mot
        r["motor_frac"] = mot / tot if tot > 0 else None
        r["n_negative_struct"] = sum(1 for s in r["segments"]
                                     if s["mass"] > 1e-6 and s["m_struct"] is not None
                                     and s["m_struct"] < 0)
        # Usable only if nearly all actuators are priced and structure is left over.
        reasons = []
        if r["excluded"]:
            reasons.append(r["excluded"])
        if r["dof_known_frac"] < 0.8:
            reasons.append(f"only {r['dof_known_frac']:.0%} of DOF have usable effort limits")
        if (r["motor_frac"] or 0) > MOTOR_FRAC_CEILING:
            reasons.append(f"implied actuator fraction {r['motor_frac']:.0%} "
                           f"> {MOTOR_FRAC_CEILING:.0%} ceiling — effort column is not peak torque")
        r["fit_excluded"] = "; ".join(reasons) or None
        if r["is_quadruped"] and not r["fit_excluded"] and hosts is None:
            raise SystemExit(
                f"{r['key']}: a fitted quadruped needs a QUAD_ACTUATOR_HOSTS entry "
                f"stating its knee drive and which segments carry its hip-pitch and "
                f"knee actuators")
        # Whole-robot allometry needs no motor split, so only hard exclusions apply.
        r["size_excluded"] = r["excluded"]
        out.append(r)

    payload = {
        "rules": {
            "geared_law": geared, "servo_law": servo,
            "tau_factors": TAU_FACTORS, "adopted_tau_factor": ADOPTED_TAU_FACTOR,
            "adopted_split_by_population": ADOPTED_SPLIT_BY_POP,
            "quadruped_actuator_hosts": QUAD_ACTUATOR_HOSTS,
            "disclosed": {k: [list(a) for a in v] for k, v in DISCLOSED.items()},
        },
        "robots": out,
    }
    write_json(DATA / "decomposition.json", payload, indent=1)

    kept = [r for r in out if not r["excluded"]]
    print(f"{'robot':32s} {'tot':>7s} {'motor':>7s} {'frac':>6s} {'effort':>11s} "
          f"{'src':>9s} {'dof':>5s} neg")
    for r in sorted(kept, key=lambda x: -x["total_mass_kg"]):
        flag = "" if not r["fit_excluded"] else "  <- dropped"
        print(f"{r['key']:32s} {r['total_mass_kg']:7.2f} {r['motor_mass_kg']:7.2f} "
              f"{(r['motor_frac'] or 0) * 100:5.1f}% {r['effort_quality']:>11s} "
              f"{r['motor_source']:>9s} {r['dof_known_frac']:5.0%} "
              f"{r['n_negative_struct']:3d}{flag}")
    usable = [r for r in out if not r["fit_excluded"]]
    print(f"\n{len(usable)} robots usable for the structural fit")
    for r in sorted(out, key=lambda x: x["key"]):
        if r["fit_excluded"] and not r["excluded"]:
            print(f"  dropped {r['key']:30s} {r['fit_excluded']}")
    print(f"\nwrote {DATA / 'decomposition.json'}")


if __name__ == "__main__":
    sys.exit(main())
