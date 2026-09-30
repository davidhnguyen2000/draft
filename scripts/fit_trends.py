#!/usr/bin/env python3
"""Refit every trend the generator uses, from the raw surveys (§III).

Fits 114 catalogued actuators and 49 robot descriptions; the stages run here
are the only writers of `src/draft/trends/data/`.

    python scripts/fit_trends.py                  # refit everything
    python scripts/fit_trends.py --only actuators # just the actuator trends
    python scripts/fit_trends.py --dry-run        # print the stages, run nothing

Two pipelines, each numbered in dependency order:

  actuators           the catalogue -> Table I
    1 validate        check the hand-maintained YAML before anything fits it
    2 build           one consistent peak-power rule over every entry
    3 catalogue       the population statistics §III-A reports
    4 fit             the four relations: reduction, mass, volume, rotor inertia
    5 leave-one-out   hold each entry out, refit, predict it: the LOO column

  robot descriptions  49 published robots -> Table II
    1 fetch           clone the survey population (~36 repos; see --skip-fetch)
    2 parse           every description -> segments with mass and inertia
    3 classify        each segment -> an anatomical class
    4 attribute       remove the actuator mass a link carries, leaving structure
    5 fit             structural mass per class against segment length
    6 leave-one-out   hold each ROBOT out and predict its total mass

Descriptions stages 1-2 need the corpus on disk; their output
`data/segments.json` is committed, so `--skip-fetch` skips both and starts at
stage 3. A full run checks the fitted population against the paper's. After a
refit, `pytest tests/test_trends.py` checks Tables I and II.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from draft.paths import repo_root

REPO = repo_root()

PIPELINES = {
    "actuators": (
        REPO / "datasets" / "actuators",
        [("01_validate.py", "check the hand-maintained catalogue"),
         ("02_build_dataset.py", "one consistent peak-power rule"),
         ("03_catalogue_stats.py", "the population statistics"),
         ("04_fit_trends.py", "the four actuator relations (Table I)"),
         ("05_leave_one_out.py", "held-out prediction (the LOO column)")],
    ),
    "robots": (
        REPO / "datasets" / "robot_descriptions",
        [("01_fetch_descriptions.py", "clone the survey population [corpus]"),
         ("02_parse_descriptions.py", "descriptions -> segments [corpus]"),
         ("03_classify_segments.py", "segments -> anatomical classes"),
         ("04_attribute_actuators.py", "remove actuator mass, leaving structure"),
         ("05_fit_trends.py", "structural mass per class (Table II)"),
         ("06_leave_one_out.py", "hold each ROBOT out, predict its total mass")],
    ),
}


def run(cwd: Path, script: str, dry: bool) -> int:
    rel = (cwd / "scripts" / script).relative_to(REPO)
    print(f"\n{'=' * 72}\n== {rel}\n{'=' * 72}", flush=True)
    if dry:
        return 0
    r = subprocess.run([sys.executable, str(Path("scripts") / script)], cwd=cwd)
    if r.returncode != 0:
        print(f"\n!! {rel} exited {r.returncode} — stopping, because every stage "
              f"after this one reads what it was supposed to write.")
    return r.returncode


#: The population behind Table II (§III-C): 29 humanoids and 20 quadrupeds.
PAPER_POPULATION = {"n_usable_humanoids": 29, "n_usable_quadrupeds": 20}


def check_population(names: list[str]) -> int:
    """Fail if the robots refit used fewer robots than the paper (partial corpus)."""
    import json

    if "robots" not in names:
        return 0
    path = REPO / "src" / "draft" / "trends" / "data" / "link_trends.json"
    try:
        prov = json.loads(path.read_text())["provenance"]
    except Exception as exc:
        print(f"\ncould not read the fitted population back: {exc}")
        return 0

    got = {k: prov.get(k) for k in PAPER_POPULATION}
    short = {k: (got[k], want) for k, want in PAPER_POPULATION.items()
             if isinstance(got[k], int) and got[k] < want}
    if not short:
        print("\npopulation checks out: "
              + ", ".join(f"{k.replace('n_usable_', '')} {got[k]}"
                          for k in sorted(got) if got[k] is not None))
        return 0

    print("\n" + "!" * 72)
    print("REFIT OVER A PARTIAL CORPUS — what was just written is NOT the paper's "
          "Table II.")
    for k, (n, want) in sorted(short.items()):
        print(f"  {k.replace('n_usable_', '')}: fitted on {n}, "
              f"the paper surveyed {want}")
    print("\nThe corpus is not all on disk. Restore the shipped coefficients:\n"
          "    git checkout src/draft/trends/data/link_trends.json \\\n"
          "                 datasets/robot_descriptions/data/\n"
          "then either fetch the whole population with\n"
          "    python scripts/setup_data.py --all\n"
          "or refit from the parsed `segments.json` this repository carries:\n"
          "    python scripts/fit_trends.py --only robots --skip-fetch")
    print("!" * 72)
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=sorted(PIPELINES), default=None,
                    help="run one pipeline rather than both")
    ap.add_argument("--skip-fetch", action="store_true",
                    help="skip the two stages that need the 49-robot corpus on "
                         "disk, and refit from the parsed `segments.json` this "
                         "repository already carries")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the stages in order and run none of them")
    args = ap.parse_args()

    names = [args.only] if args.only else list(PIPELINES)
    for name in names:
        cwd, stages = PIPELINES[name]
        print(f"\n\n######## {name} ########")
        for i, (script, why) in enumerate(stages, start=1):
            if args.skip_fetch and "[corpus]" in why:
                print(f"  skip {i}. {script:28s} {why}  (--skip-fetch)")
                continue
            rc = run(cwd, script, args.dry_run)
            if rc:
                return rc

    if not args.dry_run:
        rc = check_population(names)
        print("\n\nRefitted. The coefficients now in src/draft/trends/data/ are what "
              "every generated robot will be priced against.\n"
              "Check them against the paper:  pytest tests/test_trends.py")
        return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
