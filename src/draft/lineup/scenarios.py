"""Fixed operating points, in absolute units, at which each design is traced.

Unlike the battery in ``evaluate.py`` (a ladder reduced to scalars), a scenario
is one condition that ``collect_traces.py`` records and ``utilization_report.py``
reduces to actuator-limit statistics.

mjlab spawns on top of a ``pyramid_stairs`` mound and at the bottom of a
``pyramid_stairs_inv`` pit, hence ``stairs_down`` / ``stairs_up``.
"""
from __future__ import annotations

#: policy: task whose policy runs it; cmd (m/s); terrain (step m, None = flat);
#: push (kick m/s, None = none); only: pin to one stair sub-terrain.
SCENARIOS: dict[str, dict] = {
    "walk":        {"policy": "flat",     "cmd": 0.40, "terrain": None, "push": None},
    "sprint":      {"policy": "velocity", "cmd": 2.50, "terrain": None, "push": None},
    "stairs":      {"policy": "terrain",  "cmd": 0.40, "terrain": 0.10, "push": None},
    "push":        {"policy": "push",     "cmd": 0.40, "terrain": None, "push": 2.50},
    "stairs_down": {"policy": "terrain",  "cmd": 0.40, "terrain": 0.10,
                    "push": None, "only": "pyramid_stairs"},
    "stairs_up":   {"policy": "terrain",  "cmd": 0.40, "terrain": 0.10,
                    "push": None, "only": "pyramid_stairs_inv"},
}
