# Reproducing the paper

Everything runs on one machine. There are three depths for the locomotion
results (§V); the trends and twins (§III–IV) need no GPU at any depth.

| depth | needs | recreates |
|---|---|---|
| 1. From the committed record | this checkout | every §V number, Table V, Fig. 8 |
| 2. From the released archives | a download, no GPU | the committed record, from raw evaluation cells |
| 3. From scratch | one CUDA GPU (`pip install -e ".[rl]" -c constraints.txt`, then `wandb login`) | the 120 trainings and 360 evaluation cells |

## Trends and twins

```shell
pytest                               # pins every coefficient in Tables I and II
python scripts/fit_trends.py --skip-fetch   # refit from the committed parsed survey
python scripts/fit_trends.py         # the same from the raw descriptions (clones ~4.4 GB first)
python scripts/setup_data.py         # the four vendor descriptions, pinned
python scripts/make_twins.py         # build, adapt, measure inertia error, render (Fig. 5)
python scripts/figures/make_all.py   # every scripted figure
```

Rendering drives a headless Chrome, so Chrome or Chromium must be installed
(`make_twins.py --steps render --mujoco` avoids it).

## §V, depth 1: from the committed record

```shell
python scripts/train_all.py numbers       # every §V number; writes figures/tab_capability.tex
python scripts/figures/fig8_capability.py
pytest tests/test_paper_numbers.py
```

`experiments/results.json` holds, per arm, design and seed, the capability
scalars, the curves Fig. 8 draws and the actuator-limit statistics.

## §V, depth 2: from the released archives

**Not yet available.** The archives below have not been published, and
`experiments/artifacts.json`, the manifest `fetch` checks them against, will
ship with them; until then `fetch` stops with "no manifest". Depth 1 needs none
of it.

| archive | contents |
|---|---|
| `eval` | 360 evaluation cells |
| `walk` | 30 recorded flat walks |
| `traces` | sprint operating points at each design's peak command |
| `checkpoints` | the final policy of each of the 120 runs (no optimizer state) |

```shell
python scripts/train_all.py fetch --only eval walk traces   # verifies sha256, unpacks into logs/
python scripts/train_all.py freeze                          # rewrites experiments/results.json
python scripts/train_all.py numbers
```

To re-score the released policies on a GPU without retraining:

```shell
python scripts/train_all.py fetch --only checkpoints
python scripts/train_all.py designs
python scripts/train_all.py evaluate --force
python scripts/train_all.py walk --force
python scripts/train_all.py traces
```

## §V, depth 3: from scratch

```shell
python scripts/train_all.py designs         # build the three machines (seconds)
python scripts/train_all.py all --dry-run   # print every remaining command
python scripts/train_all.py all             # designs, train, evaluate, traces, walk, freeze, numbers
```

Each step skips work already on disk, so an interrupted run resumes. The run
configs are written from the machines, so a dry run needs `designs` first.
`--designs`, `--seeds` and `--tasks` narrow the sweep steps; a narrowed `all`
skips `freeze` and `numbers`, which reduce the whole study. Expect about 50
GPU-hours.

- **Machines.** `experiments/quadruped_variants/{cheetah,bear,giraffe}.yaml`
  over `src/draft/robots/quadruped/parameters.yaml`, built into
  `generated/<design>/`. `experiments/machines.json` pins a fingerprint of each
  compiled machine (its physics, not its XML text, so a MuJoCo upgrade that only
  reorders the file still matches), and the sweep refuses to train or score
  against a mismatch. `python scripts/generate_quadrupeds.py` builds and checks
  them; a change to the trends or generator that moves a machine needs `--pin`
  and a retrain.
- **Training.** The 120 configs are expanded from `experiments/lineup.yaml`.
  Stage 1 trains a flat base policy per design and seed; stage 2 warm-starts
  velocity, terrain and push from it. Every config states `terrain_scan_fix`
  explicitly, and the sweep refuses one that does not.
- **Evaluation.** Every design is scored on one battery in absolute units (m/s,
  m of step, m/s of kick): flat, sprint, stairs and push, each on the nominal
  machine and with inertias drawn from the twin measurement (see
  [`tasks.md`](tasks.md)).
- **Environment.** The `rl` extra pins the RL stack (`mjlab`, `mujoco`,
  `mujoco-warp`, `rsl-rl-lib`, `torch`, `wandb`) to the versions the paper's
  policies were trained with. The first training step records the commit and
  `pip freeze` in `logs/`.
- **Logging.** As in mjlab, every run logs to Weights & Biases and uploads its
  checkpoints, so run `wandb login` once before training.

MuJoCo Warp is not deterministic on GPU, so retrained numbers will fall inside
the paper's ten-seed ranges rather than on its medians.

## Joint-limit analysis (Table VI)

```shell
python scripts/analyze.py --robot cheetah --scenario sprint --checkpoint <ckpt>
python scripts/analyze.py --trace <recording>.npz   # plot an existing recording, no GPU
```
