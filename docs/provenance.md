# Where each number in a generated robot comes from

## The model's own report

Every generated folder carries `feasibility_report.yaml`:

```shell
python scripts/generate_robot.py --robot humanoid --output-dir generated/h --fixed-output-dir
cat generated/h/feasibility_report.yaml
```

Per actuator class it records which two quantities the design stated
(`designed_from`), what was solved from them, the distance to the frontier
(`frontier_tau_max_Nm`) and anything the trends overrode (`overridden_by_law`).
`segment_densities` lists every structural density applied and the measured
band it came from. `mass_composition_check` scores the robot's actuator and
trunk mass fractions against the population. `parameters_resolved.yaml` beside
it is the complete design as generated.

## The tests

```shell
pytest             # fast suite
pytest --runslow   # plus the four twin rebuilds
```

`tests/test_trends.py` pins every coefficient in the paper's Tables I and II.
`tests/test_feasibility.py` checks that the gate refuses beyond-frontier
designs and that trend-owned values cannot be set. `tests/test_twins.py`
rebuilds the four twins and pins each mass ratio and the 1.10× geometric mean.

## The datasets

[`actuator-dataset.md`](actuator-dataset.md) and
[`robot-dataset.md`](robot-dataset.md) give each fit's method, leave-one-out
results and limitations. The coefficients the generator reads live in
`src/draft/trends/data/`, written only by the stages under `datasets/*/scripts/`.

## What is asserted, not fitted

- **Gearbox friction and damping.** The catalogue publishes neither.
- **Rotor inertia** is fitted, but through a derived radius; treat `armature` as
  accurate to about 1.56×. See `MotorTrends` in `src/draft/trends/feasibility.py`.
