# Architecture

## Layout

| path | what it is |
|---|---|
| `src/draft/generation/` | The compiler. `generator.py:RobotGenerator` loads a robot directory, resolves `${expr}` (`expr_eval.py`), applies the trends and the packing gate (`packing.py`), and emits a compiled MuJoCo `MjSpec` (`spec_builder.py`). `tree_ops.py` handles mirroring and motor classes; `mesh_spec.py` and `layered_mesh.py` loft the root body. Format: [`robot-format.md`](robot-format.md). |
| `src/draft/trends/` | The fitted trends and feasibility checks. `feasibility.py` holds `MotorTrends`, `LinkTrends` and every refusal; `motor_solve.py` turns the two numbers a design states about an actuator into the full set; `motor_catalog.py` anchors a class on a real part. Coefficients live in `trends/data/`. |
| `src/draft/robots/` | The shipped specifications (`humanoid`, `quadruped`), discovered automatically. |
| `src/draft/twins/` | Four Unitree robots rebuilt from their specs and scored. See [`twins.md`](twins.md). |
| `src/draft/tasks/` | The quadruped RL tasks. See [`tasks.md`](tasks.md). |
| `src/draft/lineup/` | The paper's training and evaluation sweep, driven by `scripts/train_all.py`. See [`reproducing.md`](reproducing.md). |
| `src/draft/editor/` | The browser editor behind `scripts/robot_tuner.py`. |
| `datasets/` | The actuator catalogue and robot-description survey, with the numbered stages that fit the trends. |
| `experiments/` | Design variants, the sweep specification and the frozen results. |
| `scripts/` | Command-line entry points; `scripts/figures/` has one script per paper figure. |
| `tests/` | `pytest` (a few seconds); `--runslow` adds the four twin rebuilds. |

Output goes to `generated/` and `logs/`, both git-ignored.

## Generation, in order

1. `RobotGenerator` reads `parameters.yaml` and `tree.yaml` and sizes each actuator class from the trends.
2. `${expr}` placeholders resolve against the parameter table. `<joint>_mot: L` expands into that joint's motor radius, length, effort, velocity and armature.
3. A `mesh.yaml`, if present, lofts the root body from stacked cross-sections.
4. `SpecEmitter` builds the `MjSpec`, mirrors symmetric subtrees and compiles. The packing and feasibility gates run here, and a failing model is never written.

Output goes to `generated/<robot>_<timestamp>/` unless `--output-dir` is given.
