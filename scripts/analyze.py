#!/usr/bin/env python3
"""Record a rollout and report whether the robot is actuator- or policy-limited.

Writes three plots:

    <out>/<robot>_<scenario>_actuators.png   every (speed, torque) sample against
                                             each joint's own envelope
    <out>/<robot>_<scenario>_task.png        commanded vs achieved speed, base
                                             height, and the contact pattern
    <out>/<robot>_<scenario>_rewards.png     each reward term over the episode

and prints the verdict.

    python scripts/analyze.py --robot cheetah --scenario sprint \
        --checkpoint logs/cheetah_velocity_s0/*/model_1799.pt

    python scripts/analyze.py --trace logs/lineup_traces/cheetah_sprint.npz
        # plot a recording you already have; needs no GPU

Scenarios are identical for every design: `walk` 0.4 m/s flat, `sprint`
2.5 m/s flat, `stairs` 0.10 m steps, `push` 0.4 m/s under 2.5 m/s kicks.
Recording needs mjlab and a GPU; plotting a `.npz` does not. The reward plot is
skipped if the installed mjlab does not expose per-step reward terms.
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

from draft.paths import repo_root


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", help="generated design directory, e.g. cheetah")
    ap.add_argument("--scenario", default="sprint",
                    help="absolute condition to record at (default: sprint)")
    ap.add_argument("--checkpoint", help="policy to roll out")
    ap.add_argument("--trace", help="skip recording; plot this .npz instead")
    ap.add_argument("--out", default=None, help="output directory for the plots")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    from draft import analysis

    if args.trace:
        trace_path = Path(args.trace)
    else:
        if not (args.robot and args.checkpoint):
            ap.error("--robot and --checkpoint are required unless --trace is given")
        # Checkpoint paths carry a timestamp, so accept a glob.
        hits = sorted(glob.glob(args.checkpoint))
        ckpt = hits[-1] if hits else args.checkpoint
        if not Path(ckpt).exists():
            ap.error(f"no checkpoint at {args.checkpoint}")

        from draft.lineup.collect_traces import collect
        from draft.lineup.scenarios import SCENARIOS
        if args.scenario not in SCENARIOS:
            ap.error(f"unknown scenario {args.scenario!r}; "
                     f"choose from {', '.join(SCENARIOS)}")
        trace_path = Path(args.out or "logs/analyze") / \
            f"{args.robot}_{args.scenario}.npz"
        collect(args.robot, args.scenario, ckpt, trace_path, args.device,
                with_rewards=True)

    trace = analysis.Trace.load(trace_path)
    out = Path(args.out or trace_path.parent)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{trace.robot or 'robot'}_{trace.scenario or 'rollout'}"

    made = [analysis.plot_actuators(trace, out / f"{stem}_actuators.png"),
            analysis.plot_task(trace, out / f"{stem}_task.png")]
    rew = analysis.plot_rewards(trace, out / f"{stem}_rewards.png")
    if rew is None:
        print("\n! no reward terms in this recording — re-record with "
              "--with-rewards, or this mjlab does not expose the per-step buffer")
    else:
        made.append(rew)

    s = analysis.summary(trace)
    print(f"\n=== {s['robot']} — {s['scenario']} "
          f"({s['seconds']:.0f} s x {s['replicas']} replicas) ===")
    print(f"  commanded {s['commanded_ms']:.2f} m/s   "
          f"achieved {s['achieved_ms']:.2f} m/s   "
          f"({s['tracking_error_ms']:+.2f})   fell {s['fell_frac']:.0%}")
    print(f"\n  {'joint':22s}{'tau %':>8s}{'speed %':>9s}{'envelope':>10s}"
          f"{'% near limit':>14s}")
    for r in analysis.per_joint(trace):
        print(f"  {r['joint']:22s}{r['tau_pct']:8.0f}{r['speed_pct']:9.0f}"
              f"{r['envelope']:10.2f}{100 * r['frac_near_limit']:13.1f}%")
    print(f"\n  {analysis.verdict(s)}")

    root = repo_root()
    print("\nwrote:")
    for p in made:
        try:
            print(f"  {p.relative_to(root)}")
        except ValueError:
            print(f"  {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
