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

## macOS CPU setup

Conda is only an interpreter provider here. On macOS use Python 3.11 in an
isolated environment; the system Python and the Linux/CUDA setup above do not
need to change. With [uv](https://docs.astral.sh/uv/):

```shell
uv venv --python 3.11 .venv
source .venv/bin/activate
export PYTHONNOUSERSITE=1
uv pip install -e ".[editor,dev]" -c constraints.txt
python scripts/generate_robot.py --robot quadruped
python scripts/generate_robot.py --robot humanoid
python scripts/robot_tuner.py         # http://localhost:8081; scene on :8080
```

The `editor` extra includes `mjviser`, which the viewer imports directly. It
can be installed without `figures`, `twins`, or the CUDA/RL stack. For the CPU
paper checks, extend the same environment:

```shell
uv pip install -e ".[editor,twins,figures,dev]" -c constraints.txt
python scripts/fit_trends.py --skip-fetch
python scripts/generate_quadrupeds.py
python scripts/setup_data.py
python scripts/make_twins.py --steps build adapt inertia
pytest --runslow
python scripts/train_all.py numbers
```

`constraints.txt` keeps the measured NumPy 2.2.5/SciPy 1.15.3 versions on Linux.
On macOS it selects NumPy 2.3.5 and SciPy 1.16.3 (both require Python >=3.11):

- SciPy 1.15.3's arm64 PROPACK extension failed to load on macOS 27.0.1 with
  `section '__DATA/__thread_bss' has a zero-fill section type, but offset field
  is not zero`. This stopped trimesh's connected-component calculation and
  both generators. SciPy 1.15.2 on Python 3.10 and 1.15.3 on Python 3.11 also
  failed; 1.16.3 loaded successfully. See the
  [SciPy wheel issue](https://github.com/scipy/scipy/issues/25635).
- NumPy 2.2.5 emitted divide-by-zero, overflow and invalid-value `matmul`
  warnings on the tested Apple M5 Pro. NumPy 2.3.5 eliminated them while the
  model fingerprints and mass tests remained unchanged. NumPy documents the
  [Accelerate warning fix](https://numpy.org/doc/2.4/release/2.3.1-notes.html)
  starting in 2.3.1. No floating-point warning filters are applied.

Validated on an arm64 Apple M5 Pro, macOS 27.0.1, Python 3.11.17, MuJoCo
3.8.0, Pinocchio 4.0.0, viser 1.0.27 and mjviser 0.0.14. The humanoid and
quadruped compiled and ran in MuJoCo; both rendered in Chrome, and parameter
edits and Generate rebuilt them. All three paper designs matched
`experiments/machines.json`. The trends refit left the committed data unchanged;
twin/vendor mass ratios were G1 1.1146, H2 0.9090, Go2 1.0105 and B2 0.8412
(GMFE 1.1017). The CPU dynamics comparison at 300 poses reproduced Table III's
rounded medians and sample variances. See [reproducing.md](reproducing.md) for
figure prerequisites and the distinction between the committed RL record and
new GPU experiments.

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
