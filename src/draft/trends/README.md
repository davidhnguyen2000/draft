# `draft.trends`

Trends fitted to 114 catalogued actuators and 49 published robot descriptions,
and the feasibility checks that run on every generated model.

| file | what it owns |
|---|---|
| `feasibility.py` | `MotorTrends`, `LinkTrends`, and every refusal |
| `motor_solve.py` | the two numbers a design states about an actuator class → the full set the generator needs |
| `motor_catalog.py` | anchoring a class on a named catalogue part |
| `data/` | the fitted coefficients |

```
data/actuator_trends.json    the four fitted actuator relations (Table I)
data/actuator_catalog.json   the 114 modules, for anchoring a class on a real part
data/link_trends.json        structural density, allometry, mass fractions (Table II)
```

These files ship inside the package so an installed Draft reads the same
coefficients the paper reports. The stages under `datasets/*/scripts/` are their
only writers.

A class beyond its family's frontier, or two consecutive actuators sharing
space, raises `FeasibilityViolation` and nothing is written.
`allow_hypothetical: true` and `allow_motor_overlap: true` downgrade these to
warnings in `feasibility_report.yaml`.

`armature` is the loosest number the generator emits; quote reflected inertia at
about 1.56×.

Details: [`docs/fitted-trends.md`](../../../docs/fitted-trends.md),
[`docs/actuator-dataset.md`](../../../docs/actuator-dataset.md),
[`docs/robot-dataset.md`](../../../docs/robot-dataset.md).
