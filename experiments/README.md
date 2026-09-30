# `experiments/`

The paper's §V study and what it produced.

| file | what it is |
|---|---|
| `lineup.yaml` | The study as one specification: three designs, ten seeds, four tasks, and per-task budgets. `scripts/train_all.py` expands it into 120 run configurations under `generated/lineup_configs/`. |
| `quadruped_variants/*.yaml` | The three designs (cheetah, bear, giraffe), as overrides on `src/draft/robots/quadruped/parameters.yaml`. |
| `machines.json` | A fingerprint of each compiled machine (`generated/<design>/`) and a hash of every input that built it. `scripts/generate_quadrupeds.py` checks it; the sweep refuses to train or score a mismatch. Rewritten only by `scripts/generate_quadrupeds.py --pin`. |
| `results.json` | The frozen §V record, read by `scripts/train_all.py numbers`, Fig. 8 and `tests/test_paper_numbers.py`. |
| `twin_inertia_mismatch.json` | The twins' measured inertia error, which the perturbed evaluation draws from. Written by `scripts/make_twins.py --steps inertia`. |

`tests/test_lineup_configs.py` pins the expansion of `lineup.yaml`.
