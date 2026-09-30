#!/usr/bin/env python3
"""Fetch the vendor robot descriptions the twins are measured against.

No robot model ships with Draft; each is cloned sparse, pinned to the commit
the published numbers were measured at. A missing upstream is skipped, not fatal.

    python scripts/setup_data.py              # the four twin targets
    python scripts/setup_data.py --all        # plus the full survey population
    python scripts/setup_data.py --list       # what is present, what is missing

Needed only before `scripts/make_twins.py`.
"""
from __future__ import annotations

import sys

from draft.cli.fetch import main

if __name__ == "__main__":
    sys.exit(main())
