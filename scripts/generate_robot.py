#!/usr/bin/env python3
"""Compile one robot from its parameter table and tree, headless (the editor's
Generate button without the browser).

    python scripts/generate_robot.py --robot quadruped
    python scripts/generate_robot.py --robot humanoid --output-dir generated/tall
    python scripts/generate_robot.py --robot quadruped \
        --overrides experiments/quadruped_variants/bear.yaml

`--robot` names any folder under `src/draft/robots/`. Output: the MJCF,
`scene.xml`, `parameters_resolved.yaml` and `feasibility_report.yaml`.
For the paper's three quadrupeds use `scripts/generate_quadrupeds.py`.
"""
from __future__ import annotations

import sys

from draft.cli.generate import main

if __name__ == "__main__":
    sys.exit(main())
