#!/usr/bin/env python3
"""Open the browser design editor: parameters beside a live 3D robot.

    python scripts/robot_tuner.py
    python scripts/robot_tuner.py --port 9000 --viser-port 9001

Derived numbers are computed by the same code as the generator. **Generate**
overwrites the selected folder under `generated/` with the MJCF, scene,
`parameters_resolved.yaml` and `feasibility_report.yaml`. Overwriting a paper
machine (`generated/<robot>/`) with a different design makes `scripts/train_all.py`
refuse it, since it fingerprints those against `experiments/machines.json`.
"""
from __future__ import annotations

import sys

from draft.cli.editor import main

if __name__ == "__main__":
    sys.exit(main())
