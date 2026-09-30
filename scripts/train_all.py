#!/usr/bin/env python3
"""Run the paper's §V study: 3 designs x 10 seeds x 4 tasks.

120 trainings and 360 evaluation cells; needs a CUDA GPU for days. For a single
run use `scripts/train.py`.

    python scripts/train_all.py all --dry-run     # print every command, run none
    python scripts/train_all.py all               # the whole study, in order
    python scripts/train_all.py train --designs cheetah --seeds 0
    python scripts/train_all.py numbers           # needs no GPU; reads the record

Steps, in the order `all` runs them (each skips work already on disk):

    designs    build the three machines and check them against the pin
    train      120 runs: stage 1 flat, then stage 2 warm-started per seed
    evaluate   360 cells — every design on ONE absolute battery, nominal and
               with link inertias perturbed by the twin study's residual
    traces     sprint operating points at each design's peak command
    walk       30 flat-walk recordings, for the walk limits
    freeze     reduce all of it to experiments/results.json
    numbers    print every §V number from that record, and write Table V

Not part of `all`:

    fetch      download the released evaluation cells, walk recordings and
               sprint traces, so `freeze` and `numbers` run without training

Run configs are expanded from `experiments/lineup.yaml` into
`generated/lineup_configs/`. Machine fingerprints are checked against
`experiments/machines.json` before anything trains or scores. MuJoCo Warp is
not deterministic on GPU: retrained numbers land within the paper's ten-seed
ranges, not on its medians.
"""
from __future__ import annotations

import sys

STEPS = ("designs", "train", "evaluate", "traces", "walk",
         "freeze", "numbers", "fetch", "all")
#: Steps handled by `draft.lineup.sweep`.
SWEEP_STEPS = ("designs", "train", "evaluate", "traces", "walk")


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in STEPS:
        print(__doc__)
        print(f"\nusage: python scripts/train_all.py {{{'|'.join(STEPS)}}} [options]")
        return 2

    step, rest = sys.argv[1], sys.argv[2:]

    from draft.lineup import artifacts, freeze_results, paper_numbers, sweep

    def run_sweep(s: str) -> int:
        sys.argv = ["train_all.py", s, *rest]
        return sweep.main() or 0

    def run_freeze() -> int:
        sys.argv = ["train_all.py", *(rest if step == "freeze" else [])]
        return freeze_results.main() or 0

    def run_numbers() -> int:
        sys.argv = ["train_all.py"]
        return paper_numbers.main() or 0

    if step in SWEEP_STEPS:
        return run_sweep(step)
    if step == "freeze":
        return run_freeze()
    if step == "numbers":
        return run_numbers()
    if step == "fetch":
        # `fetch --only eval` downloads; `fetch pack|verify ...` reach the other two.
        sub = rest if rest and rest[0] in ("pack", "fetch", "verify") else ["fetch", *rest]
        sys.argv = ["train_all.py", *sub]
        return artifacts.main() or 0

    # all: one sweep pass over every step, so a dry run's planned trainings are
    # still known when evaluate asks for their checkpoints
    rc = run_sweep("all")
    if rc:
        return rc
    if "--dry-run" in rest:
        print("\n=== freeze, numbers ===\n(dry run: they would reduce logs/ to "
              "experiments/results.json and print every number)")
        return 0
    if any(a in rest for a in ("--designs", "--seeds", "--tasks", "--arms")):
        print("\n=== freeze, numbers ===\nskipped: freeze reduces the whole study "
              "(3 designs x 10 seeds), and this run was narrowed")
        return 0
    return run_freeze() or run_numbers()


if __name__ == "__main__":
    sys.exit(main())
