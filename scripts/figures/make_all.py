#!/usr/bin/env python3
"""Redraw the paper's scripted figures into `figures/`.

    python scripts/figures/make_all.py

Figs. 1, 3 and 7 are drawn by hand. `README.md` here lists each figure's inputs.
"""
from __future__ import annotations

import runpy
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGES = ["fig2_link_primitives.py", "fig4_actuator_trends.py",
          "fig5_twin_dynamics.py", "fig6_quadruped_lineup.py",
          "fig8_capability.py"]


def main() -> int:
    failed = []
    for name in STAGES:
        print(f"\n=== {name}")
        sys.argv = [str(HERE / name)]
        try:
            runpy.run_path(str(HERE / name), run_name="__main__")
        except SystemExit as e:
            if e.code:
                failed.append(name)
        except Exception:
            # One figure's missing input must not cost every figure after it.
            traceback.print_exc()
            failed.append(name)
    print(f"\nfigures in {HERE.parents[1] / 'figures'}")
    if failed:
        print(f"not drawn: {', '.join(failed)} (scripts/figures/README.md lists "
              f"what each needs first)")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
