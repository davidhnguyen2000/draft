#!/usr/bin/env python3
"""``scripts/setup_data.py`` — download the vendor robot descriptions.

No robot model ships with Draft; each belongs to its vendor under its own licence.

    python scripts/setup_data.py              the four twin targets — three repositories
    python scripts/setup_data.py --all        plus the full survey population (~36 repos)
    python scripts/setup_data.py --list       what is present and what is missing

Clones are sparse and blobless. An unreachable source is reported and skipped; the exit code is non-zero only if ``--only`` sources were
requested and none arrived.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from draft.sources import TWIN_SOURCES, Result, fetch_many, present


def _print_table(results: list[Result]) -> None:
    for r in results:
        mark = "ok  " if r.ok else "SKIP"
        note = r.detail if not r.ok else ("already present" if r.already else r.detail)
        print(f"  {mark}  {r.key:22s} {note}")


def _list() -> int:
    print("Twin targets (paper §IV-F) — `scripts/setup_data.py`\n")
    missing = 0
    for s in TWIN_SOURCES:
        p = present(s)
        missing += p is None
        print(f"  {'present' if p else 'MISSING':8s}  {s.key:22s} {s.provides}")
        if p:
            print(f"            {p}")
    print("\nSurvey population (the structural fits) — `python scripts/setup_data.py --all`")
    print("  ~36 repositories, listed in")
    print("  datasets/robot_descriptions/scripts/01_fetch_descriptions.py")
    if missing:
        print(f"\n{missing} of {len(TWIN_SOURCES)} twin sources missing; "
              f"run `scripts/setup_data.py` to get them.")
    return 0


def _fetch_survey() -> int:
    """Fetch the full survey population by running dataset stage 1, which owns that list."""
    from draft.paths import repo_root
    stage = repo_root() / "datasets" / "robot_descriptions"
    print("\n[survey] the full fit population, via "
          "datasets/robot_descriptions/scripts/01_fetch_descriptions.py")
    print("[survey] this clones ~36 repositories; failures are recorded in "
          "data/manifest.json and skipped\n", flush=True)
    r = subprocess.run([sys.executable, "scripts/01_fetch_descriptions.py"], cwd=stage)
    if r.returncode != 0:
        print("\n[survey] the stage exited non-zero. Whatever it did fetch is still "
              "usable; re-run it to retry the rest.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true",
                    help="also fetch the full survey population the structural "
                         "trends are fitted on (~36 repositories)")
    ap.add_argument("--list", action="store_true",
                    help="show what is present and what is missing, fetch nothing")
    ap.add_argument("--only", nargs="+", metavar="KEY",
                    help=f"fetch only these ({', '.join(s.key for s in TWIN_SOURCES)})")
    ap.add_argument("--timeout", type=int, default=900,
                    help="seconds per repository (default: 900)")
    args = ap.parse_args()

    if args.list:
        return _list()

    wanted = TWIN_SOURCES
    if args.only:
        by_key = {s.key: s for s in TWIN_SOURCES}
        unknown = [k for k in args.only if k not in by_key]
        if unknown:
            print(f"unknown source(s): {', '.join(unknown)}. "
                  f"Known: {', '.join(by_key)}", file=sys.stderr)
            return 2
        wanted = tuple(by_key[k] for k in args.only)

    print(f"[twins] {len(wanted)} source(s) — sparse, blobless, XML only\n")
    results = fetch_many(wanted, timeout=args.timeout)
    _print_table(results)

    ok = [r for r in results if r.ok]
    bad = [r for r in results if not r.ok]
    print(f"\n{len(ok)}/{len(results)} available.")
    if bad:
        print("Could not fetch: " + ", ".join(r.key for r in bad))
        print("A vendor may have moved or removed the repository. Everything else "
              "still works; the twins that needed it will say they were skipped.")

    if args.all:
        _fetch_survey()

    # A partial fetch is not a failure.
    return 1 if (args.only and not ok) else 0

