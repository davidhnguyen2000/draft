# Draft — a parametric design tool for robot exploration

A robot is two files: a flat table of numeric parameters, and a kinematic tree
whose numeric fields are arithmetic expressions over that table. The engine
compiles the pair into a simulation-ready MJCF. Whatever the design does not
state is filled in from trends fitted to 114 catalogued actuators and 49
published robot descriptions, and the assembled machine is then checked against
what that hardware can actually do.

This file is a pointer. The documentation lives in [`docs/`](docs/), split by
topic, and each package states its own claim at the top of its own README.

| if you are about to… | read |
|---|---|
| find your way around | [`docs/architecture.md`](docs/architecture.md) |
| change a density, a drawn radius, a packing rule, anything a trend owns | [`docs/fitted-trends.md`](docs/fitted-trends.md) |
| touch the RL tasks or compare two designs | [`docs/tasks.md`](docs/tasks.md) |
| quote a number, or wonder where one came from | [`docs/provenance.md`](docs/provenance.md) |
| run or write code here | [`docs/contributing.md`](docs/contributing.md) |
| fetch the vendor descriptions, or wonder why a twin skipped | [`docs/fetching-robots.md`](docs/fetching-robots.md) |
| write a `tree.yaml` | [`docs/robot-format.md`](docs/robot-format.md) |
| recreate a result from the paper | [`docs/reproducing.md`](docs/reproducing.md) |
| redraw a figure from the paper | [`scripts/figures/README.md`](scripts/figures/README.md) |

[`docs/README.md`](docs/README.md) is the full index.

## Things that will bite you

- **The package is installed**, with
  `pip install -e ".[editor,twins,figures,dev]" -c constraints.txt` (add `rl` on
  a CUDA machine); then `from draft.trends import ...`. Do not add `src/` to
  `sys.path`; `tests/test_structure.py` forbids it.
- **`generated/` is not committed.** Every task config trains
  `generated/cheetah/quadruped.xml`, so run `python scripts/generate_quadrupeds.py`
  first on a fresh checkout. Training logs to Weights & Biases: `wandb login`,
  or `WANDB_MODE=offline`.
- **`draft.paths.repo_root()` is the ONE place the checkout root is resolved**,
  and `draft.robots.robot_dir(name)` the one place a robot's directory is.
  A wrong root does not raise; it silently skips a twin.
- **The fitted coefficients live in `src/draft/trends/data/`, inside the
  package.** The stages under `datasets/*/scripts/` are their only writers. That
  is what makes "the pipeline and the reported numbers cannot drift apart"
  structural rather than a convention.
- **They are trends, not laws.** Regressions over a finite sample of real
  hardware, not statements about what is physically possible. The one genuine
  law in the chain is the airgap relation `τ_rotor ∝ r²L`, which is imposed by
  physics rather than fitted.
- **A design cannot opt out of the feasibility gate**, and values it states for
  a quantity the trends own are ignored and recorded as overridden.
- **No robot model ships here.** Every vendor description is fetched, never
  committed — `scripts/setup_data.py`, pinned to the commit the published numbers
  were measured at. A test refuses to let a `.urdf` or `.stl` be committed.
- **The paper's sweep is not uniform in `terrain_scan_fix`.** Its terrain
  cells used the corrected torso scan and its other cells did not, so every
  config states the flag explicitly. The default is `all` (corrected), and
  `train_all.py` refuses to score a config that omits it.
- **Run the tests.** `pytest` takes a few seconds; `pytest --runslow` adds the
  four twin rebuilds. `tests/test_trends.py` pins every number in the paper's
  Table I and Table II and `tests/test_paper_numbers.py` every number in §V, so
  if you move a coefficient you will hear about it.

## What is and is not in the paper

`generation/`, `trends/`, `robots/`, `twins/`, `tasks/quadruped/` and `lineup/`
(the §V sweep) are the paper. `editor/` is not, but it is how a design is
normally built.

Nothing else is here. The paper's humanoid appears as a generation example and
as a twin target, never as a policy.
