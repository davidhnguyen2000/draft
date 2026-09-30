# `draft.twins`

Digital twins of four Unitree platforms (G1, H2, Go2, B2): each is generated
from the vendor's geometry and declared torque and speed alone, then scored
against the vendor's masses and dynamics. Total mass predicts at 1.10× geometric
mean fold error.

| entry point | |
|---|---|
| `python scripts/make_twins.py` | build, adapt, measure and render all four into `generated/twins/` |
| `synthesize.py:build_twin(target, out)` | Python |
| `pytest --runslow tests/test_twins.py` | pins every ratio |

`targets.py` is the registry, `measure.py` the shared measurement schema,
`synthesize.py` the measurement → twin fixed point, `dynamics_parity.py` the
M(q)/g(q) comparison.

Results, limits and module table: [`docs/twins.md`](../../../docs/twins.md).
