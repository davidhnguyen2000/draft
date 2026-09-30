# Contributing

## Setup

```shell
conda env create -f environment.yml && conda activate draft
export PYTHONNOUSERSITE=1    # keep ~/.local packages out of the env
pip install -e ".[editor,twins,figures,dev]" -c constraints.txt   # add ,rl on a CUDA Linux box
pytest              # a few seconds
pytest --runslow    # adds the four twin rebuilds
```

`constraints.txt` pins the versions the paper's numbers were produced with;
without it pip takes the newest, which the tests still pass on but which is not
what was measured. Install `.[rl]` only on a machine with a CUDA GPU; everything
else runs on a CPU.
Training logs to Weights & Biases, so run `wandb login` once first.

## Conventions

- **Import the installed package** (`from draft.trends import ...`). Do not add
  `src/` to `sys.path`; `tests/test_structure.py` forbids it.
- **Resolve paths in one place.** `draft.paths.repo_root()` for the checkout
  root, `draft.robots.robot_dir(name)` for a robot's directory. A wrong root
  does not raise, it silently skips data.
- **Fitted coefficients** live in `src/draft/trends/data/`, written only by the
  stages under `datasets/*/scripts/`. Figure scripts read data; they never fit.
  `tests/test_trends.py` and `tests/test_paper_numbers.py` pin the published
  numbers, so moving a coefficient fails the suite.
- **New robots** are a folder under `src/draft/robots/<name>/` with
  `parameters.yaml` and `tree.yaml`, no Python. See
  [`robot-format.md`](robot-format.md).
- **No special cases.** Code should not branch on a particular robot or run; put
  such choices in a config that states them (see `terrain_scan_fix` in
  `draft.tasks.quadruped.base_env`).
- **No vendor models in git.** A test refuses any committed `.urdf`, `.stl`,
  `.dae`, `.obj` or `.xacro`.
- Preferred libraries: Pinocchio for rigid-body dynamics, viser for 3D.
- Comment *why*, not what.
