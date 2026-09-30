#!/usr/bin/env python3
"""Is a design at its hardware's limit, or at its policy's?

Utilisation against each joint's own four-quadrant DC-motor envelope, from the
traces `collect_traces.py` writes:

    tau_avail(w) = min(effort_limit, tau_stall * (1 - |w| / w_NL))
    utilisation  = |tau| / tau_avail(|w|)

1.0 is on the envelope. Pooling joints hides a saturated knee, so the hottest
joint class is reported alongside; torque and speed fractions are reported apart.

    python -m draft.lineup.utilization_report --scenario sprint --by-joint
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np

from ..paths import repo_root

ROBOTS = ("cheetah", "bear", "giraffe")
#: Discarded head of each episode (the drop-in transient).
_SETTLE_FRAC = 0.2


def _joint_class(name: str) -> str:
    for k in ("hip_roll", "hip_pitch", "knee"):
        if k in name:
            return k
    return "other"


def load(path: Path) -> dict:
    z = np.load(path, allow_pickle=True)
    tau, vel = np.asarray(z["tau"]), np.asarray(z["vel"])       # (T, B, J)
    keep = ~np.asarray(z["fell"])
    t0 = int(_SETTLE_FRAC * tau.shape[0])
    tau, vel = tau[t0:, keep, :], vel[t0:, keep, :]
    stall = np.asarray(z["stall_torque"], float)
    wnl = np.asarray(z["no_load_speed"], float)
    eff = np.asarray(z["effort_limit"], float)
    # Available torque at the speed the joint is actually turning.
    wfrac = np.abs(vel) / np.maximum(wnl, 1e-9)
    avail = np.minimum(eff, stall * np.clip(1.0 - wfrac, 0.0, None))
    # Near/past no-load speed available torque is ~0: such samples are counted
    # (`beyond`) rather than divided by, and the ratio is clipped.
    util = np.where(avail > 0.02 * np.maximum(stall, 1e-9),
                    np.abs(tau) / np.maximum(avail, 1e-9), np.nan)
    return {
        "util": np.clip(util, 0.0, 1.5),
        "beyond": wfrac >= 0.95,
        "tau_frac": np.abs(tau) / np.maximum(eff, 1e-9),
        "vel_frac": np.abs(vel) / np.maximum(wnl, 1e-9),
        "classes": [_joint_class(n) for n in np.asarray(z["joint_names"])],
        "checkpoint": str(z["checkpoint"]),
        "n": int(tau.shape[0] * tau.shape[1]),
    }


#: A joint counts as torque-saturated above this fraction of its effort limit.
_SAT = 0.90


def _hot(d):
    """The joint class closest to its TORQUE limit, and how often it is there."""
    best = (None, -1.0, 0.0)
    for c in ("hip_roll", "hip_pitch", "knee"):
        idx = [i for i, x in enumerate(d["classes"]) if x == c]
        if not idx:
            continue
        t = d["tau_frac"][:, :, idx]
        frac = float((t > _SAT).mean())
        if frac > best[1]:
            best = (c, frac, float(np.percentile(t, 95)))
    return best


def _row(d, sel=None):
    idx = slice(None) if sel is None else [i for i, c in enumerate(d["classes"]) if c == sel]
    if isinstance(idx, list) and not idx:
        return None
    u, t, v, b = (d[k][:, :, idx].ravel()
                  for k in ("util", "tau_frac", "vel_frac", "beyond"))
    return dict(u50=np.nanpercentile(u, 50), u95=np.nanpercentile(u, 95),
                t95=np.percentile(t, 95), v95=np.percentile(v, 95),
                sat=float(np.nanmean(u > 0.9)), beyond=float(b.mean()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=None, help="trace directory")
    ap.add_argument("--scenario", default=None, help="only this scenario")
    ap.add_argument("--by-joint", action="store_true")
    args = ap.parse_args()

    root = repo_root()
    d = Path(args.dir) if args.dir else root / "logs" / "lineup_traces"
    files = sorted(glob.glob(str(d / "*.npz")))
    if not files:
        raise SystemExit(f"no traces in {d} — run scripts/analyze.py")

    print("utilisation = |tau| / tau_available(|omega|), on the design's own "
          "four-quadrant envelope.\n1.0 is ON the limit. Low p95 with a low speed "
          "means the POLICY is the constraint.\n")
    hdr = "%-9s %-12s %7s %7s %8s %8s %8s   %-9s %7s %8s" % (
        "design", "policy", "u50", "u95", "sat>0.9", "past wNL", "spd p95",
        "hot joint", "tau p95", "sat>0.9")
    print(hdr); print("-" * len(hdr))
    for r in ROBOTS:
        for f in files:
            stem = Path(f).stem
            if not stem.startswith(r + "_"):
                continue
            scen = stem[len(r) + 1:]
            if args.scenario and not scen.startswith(args.scenario):
                continue
            data = load(Path(f))
            row = _row(data)
            hc, hfrac, ht95 = _hot(data)
            print("%-9s %-12s %7.3f %7.3f %7.1f%% %7.1f%% %8.3f   %-9s %7.3f %7.1f%%" % (
                r, scen, row["u50"], row["u95"], 100 * row["sat"],
                100 * row["beyond"], row["v95"], hc or "-", ht95, 100 * hfrac))
            if args.by_joint:
                for c in ("hip_roll", "hip_pitch", "knee"):
                    jr = _row(data, c)
                    if jr:
                        print("%-9s   %-10s %7.3f %7.3f %7.1f%% %7.1f%% %8.3f   %-9s %7.3f" % (
                            "", c, jr["u50"], jr["u95"], 100 * jr["sat"],
                            100 * jr["beyond"], jr["v95"], "", jr["t95"]))
        print()
    print("u50/u95 pool all twelve joints and are clipped at 1.5; they skip samples "
          "past no-load\nspeed, which `past wNL` counts instead — there the motor "
          "cannot push and the ratio has\nno denominator. `hot joint` is the class "
          "nearest its TORQUE limit, which the pooled\nnumber hides: a quadruped "
          "runs on its knees. Read torque and speed apart — a QDD runs\nout of "
          "torque with its speed range untouched, a harmonic runs out of speed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
