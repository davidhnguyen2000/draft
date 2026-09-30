"""Freeze the sweep into the one committed record every §V number is read from.

    python scripts/train_all.py freeze

Reads the sweep's raw outputs (from `scripts/train_all.py` or `train_all.py fetch`):

    logs/lineup_eval/     360 battery cells: 3 designs x 10 seeds x 6 cells x 2 arms
    logs/lineup_walk/     flat-walk rollouts, one npz per (design, seed)
    logs/lineup_traces/   sprint traces, one per (design, seed)

and writes `experiments/results.json`: per-cell capability scalars, the curves
Fig. 8 draws, and actuator-limit statistics. `paper_numbers.py` and
`scripts/figures/fig8_capability.py` read only that file.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..paths import repo_root
from . import study
from . import utilization_report as UR

ROOT = repo_root()

R = study.STUDY
EVAL = ROOT / "logs" / R.eval_dir
WALK = ROOT / "logs" / R.walk_dir

ROBOTS = ("cheetah", "bear", "giraffe")
SEEDS = tuple(range(10))
#: arm -> (cell prefix, split prefix). `dr` is the same policies with every
#: replica's link inertias drawn from the twin study's measured error.
ARMS = {"nom": ("nom", "nomsplit"), "dr": ("dr", "drsplit")}
SURV = 0.8                     # a rung is HELD when this share of replicas survives
WALK_MS = 0.40
CLASSES = ("hip_roll", "hip_pitch", "knee")


def _rows(name: str) -> list[dict]:
    f = EVAL / name
    if not f.exists():
        raise SystemExit(f"missing {f.relative_to(ROOT)} — run scripts/train_all.py "
                         "evaluate, or scripts/train_all.py fetch --only eval")
    return json.loads(f.read_text())["rows"]


def _median_speed(x: dict) -> float:
    # Per-rung replica median (the paper's statistic; distributions are skewed).
    return x["speed_q_ms"][2] if x.get("speed_q_ms") else x["speed_mean_ms"]


def _fifty(dv: list[float], survive: list[float]) -> float:
    """Kick survived by half the replicas, interpolated across the 50% line."""
    for i in range(len(dv) - 1):
        if survive[i] >= 0.5 > survive[i + 1]:
            f = (survive[i] - 0.5) / (survive[i] - survive[i + 1])
            return dv[i] + f * (dv[i + 1] - dv[i])
    return max((a for a, b in zip(dv, survive) if b >= 0.5), default=float("nan"))


def _tallest(step: list[float], frac: list[float]) -> float:
    """Tallest rung at least half the replicas traverse (see evaluate.frac_traversed)."""
    return max((s for s, f in zip(step, frac) if f >= 0.5), default=float("nan"))


def cell(arm: str, robot: str, seed: int) -> dict:
    pfx, spfx = ARMS[arm]
    key = f"{robot}_s{seed}"
    sprint = sorted(_rows(f"{pfx}_{key}_sprint_speed.json"), key=lambda x: x["cmd_ms"])
    push = sorted(_rows(f"{pfx}_{key}_push_push.json"), key=lambda x: x["push_dv_ms"])
    flat = _rows(f"{pfx}_{key}_flat_flat.json")
    walk = min(flat, key=lambda x: abs(x.get("cmd_ms", 9.0) - WALK_MS))
    # pyramid_stairs_inv spawns in a pit (climbing out); pyramid_stairs on a mound.
    up = sorted(_rows(f"{spfx}_{key}_pyramid_stairs_inv.json"), key=lambda x: x["step_h_m"])
    down = sorted(_rows(f"{spfx}_{key}_pyramid_stairs.json"), key=lambda x: x["step_h_m"])
    step = [x["step_h_m"] for x in up]
    assert step == [x["step_h_m"] for x in down], f"{arm}/{key}: stair ladders differ"
    t_up = [x["frac_traversed"] for x in up]
    t_down = [x["frac_traversed"] for x in down]

    held = [x for x in sprint if x["survive_rate"] >= SURV]
    peak = max(held, key=_median_speed)
    dv = [x["push_dv_ms"] for x in push]
    surv = [x["survive_rate"] for x in push]
    return {
        "mass_kg": walk["mass_kg"],
        "leg_length_m": walk["leg_length_m"],
        "top_speed_ms": _median_speed(peak),
        # Share of electrical draw reaching the joints, at the design's top speed.
        "eta_at_top": peak["eta_q"][2],
        "push_dv_ms": _fifty(dv, surv),
        "step_m": _tallest(step, [0.5 * (a + b) for a, b in zip(t_up, t_down)]),
        "step_ascent_m": _tallest(step, t_up),
        "step_descent_m": _tallest(step, t_down),
        "curves": {
            "cmd": [x["cmd_ms"] for x in sprint],
            "v": [_median_speed(x) for x in sprint],
            "survive": [x["survive_rate"] for x in sprint],
            "eta": [x["eta_q"][2] for x in sprint],
            "step": step, "trav_up": t_up, "trav_down": t_down,
            "dv": dv, "push_survive": surv,
        },
    }


def _xml_limits(robot_xml: Path, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Per joint (effort limit from `actuatorfrcrange`, no-load speed from `<joint>_velocity_limit`)."""
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(robot_xml))
    eff = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j): float(m.jnt_actfrcrange[j][1])
           for j in range(m.njnt)}
    wnl = {}
    for i in range(m.nnumeric):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_NUMERIC, i)
        if n.endswith("_velocity_limit"):
            wnl[n[:-len("_velocity_limit")]] = float(m.numeric_data[m.numeric_adr[i]])
    return np.array([eff[n] for n in names]), np.array([wnl[n] for n in names])


def walk_limits(robot: str) -> dict:
    """Torque and speed as fractions of the design's limits at the 0.4 m/s walk, pooled over seeds."""
    tau, vel, names, xml = [], [], None, None
    for s in SEEDS:
        z = np.load(WALK / f"{robot}_s{s}.npz", allow_pickle=True)
        meta = json.loads(str(z["meta"]))
        names = meta["joint_names"]
        xml = ROOT / meta["robot_xml"]
        if not xml.exists():                      # recorded under another path
            xml = ROOT / "generated" / robot / "quadruped.xml"
        t0 = int(UR._SETTLE_FRAC * z["torque"].shape[0])
        tau.append(z["torque"][t0:])
        vel.append(z["joint_vel"][t0:])
    eff, wnl = _xml_limits(xml, names)
    tf = np.abs(np.concatenate(tau, 1)) / eff
    vf = np.abs(np.concatenate(vel, 1)) / wnl
    out = {}
    for c in CLASSES:
        idx = [i for i, n in enumerate(names) if c in n]
        out[c] = {"tau_p95": float(np.percentile(tf[..., idx], 95)),
                  "speed_p95": float(np.percentile(vf[..., idx], 95)),
                  "near_no_load": float((vf[..., idx] >= 0.95).mean())}
    return out


def sprint_limits(robot: str) -> dict:
    """The same statistics at each design's peak sprint command, pooled over seeds."""
    ds = []
    for rel in R.sprint_traces(robot, SEEDS):
        f = ROOT / "logs" / rel
        if not f.exists():
            raise SystemExit(f"missing {f.relative_to(ROOT)} — run scripts/train_all.py "
                             "traces")
        ds.append(UR.load(f))
    classes = ds[0]["classes"]

    def pool(key: str, idx=None) -> np.ndarray:
        return np.concatenate([(d[key] if idx is None else d[key][:, :, idx]).ravel()
                               for d in ds])

    out = {"speed_p95_all": float(np.percentile(pool("vel_frac"), 95)),
           "past_no_load_all": float(pool("beyond").mean())}
    for c in CLASSES:
        idx = [i for i, x in enumerate(classes) if x == c]
        t = pool("tau_frac", idx)
        out[c] = {"tau_p95": float(np.percentile(t, 95)),
                  "near_torque_limit": float((t > UR._SAT).mean()),
                  "speed_p95": float(np.percentile(pool("vel_frac", idx), 95))}
    return out


def main() -> int:
    global R, EVAL, WALK
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None,
                    help="default experiments/results.json")
    args = ap.parse_args()
    EVAL, WALK = ROOT / "logs" / R.eval_dir, ROOT / "logs" / R.walk_dir
    args.out = args.out or str(R.record)
    rec = {"robots": list(ROBOTS), "seeds": list(SEEDS), "arms": list(ARMS),
           "cells": {arm: {r: [cell(arm, r, s) for s in SEEDS] for r in ROBOTS}
                     for arm in ARMS},
           "walk_limits": {r: walk_limits(r) for r in ROBOTS},
           "sprint_limits": {r: sprint_limits(r) for r in ROBOTS}}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1))
    shown = out.resolve().relative_to(ROOT) if out.resolve().is_relative_to(ROOT) else out
    print(f"wrote {shown}  ({out.stat().st_size / 1e3:.0f} kB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
