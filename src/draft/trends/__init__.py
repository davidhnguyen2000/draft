"""The fitted trends, and the feasibility checks they imply (paper §IV).

`feasibility.py`: the regressions (`MotorTrends`, `LinkTrends`) and the
feasibility gate. `motor_solve.py`: expands an actuator class into the full
spec. `motor_catalog.py`: resolves a class anchored on a catalogue part.

The coefficients in `data/` ship inside the package so the generator and the
paper's numbers read the same files; only the stages under `datasets/` write them.
"""

from .feasibility import (  # noqa: F401
    ACTUATOR_TRENDS_JSON, CATALOG_JSON, LINK_TRENDS_JSON,
    FeasibilityViolation, LinkTrends, MotorTrends,
    link_fits, motor_fits,
)
