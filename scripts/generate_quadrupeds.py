#!/usr/bin/env python3
"""Generate the paper's quadrupeds (cheetah, bear, giraffe).

Each design is an override in ``experiments/quadruped_variants/`` applied to
``src/draft/robots/quadruped/parameters.yaml``, written to ``generated/<name>/``,
then checked against the fingerprints pinned in ``experiments/machines.json``.

  python scripts/generate_quadrupeds.py              # all three; exit 1 on a mismatch
  python scripts/generate_quadrupeds.py --only bear
  python scripts/generate_quadrupeds.py --pin        # re-pin after a deliberate change
"""

from __future__ import annotations

import argparse

import yaml

from draft.generation.generator import RobotGenerator
from draft.lineup import machines
from draft.paths import repo_root

_REPO = repo_root()


def generate_design(name: str):
    """Generate ``generated/<name>/`` from ``experiments/quadruped_variants/<name>.yaml``."""
    cfg = machines.VARIANTS_DIR / f"{name}.yaml"
    if not cfg.exists():
        raise FileNotFoundError(f"No design config: {cfg} (have: {machines.designs()})")
    override = yaml.safe_load(cfg.read_text()) or {}
    out = machines.xml_path(name).parent
    out.mkdir(parents=True, exist_ok=True)
    RobotGenerator(machines.ROBOT_DIR).generate(out, params_override=override)
    # The RL entity reads motor limits from `parameters_from_gui.yaml` beside
    # the XML; this writes the design's override there.
    (out / "parameters_from_gui.yaml").write_text(cfg.read_text())
    print(f"  {name:8s} -> {out.relative_to(_REPO)}/quadruped.xml")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=machines.designs(), default=None,
                    help="Generate only these designs instead of the whole lineup.")
    ap.add_argument("--pin", action="store_true",
                    help="Rewrite experiments/machines.json from the machines just built "
                         "instead of checking them. Needs the whole lineup.")
    args = ap.parse_args()
    if args.pin and args.only:
        ap.error("--pin rewrites every machine's fingerprint; drop --only")

    names = args.only or machines.designs()
    print(f"Generating {len(names)} quadruped design(s): {', '.join(names)}")
    for name in names:
        generate_design(name)

    pin = machines.PIN.relative_to(_REPO)
    if args.pin:
        machines.pin(names)
        print(f"\npinned the machines in {pin}")
        return 0
    bad = machines.check(names)
    if bad:
        print(f"\nmachines DO NOT match {pin}:")
        for b in bad:
            print(f"  {b}")
        return 1
    print(f"\nmachines match {pin}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
