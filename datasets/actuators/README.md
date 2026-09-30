# datasets/actuators

A catalogue of 114 integrated rotary actuators (33 QDD, 33 MidGear, 48 Harmonic)
and the stages that fit Draft's actuator trends to it. `data/actuators/*.yaml` is
edited by hand; `data/active_flags.json` excludes rows from the fits.

Run from this directory, in order:

```shell
python scripts/01_validate.py         # check hand edits
python scripts/02_build_dataset.py    # -> data/derived/actuators.json
python scripts/03_catalogue_stats.py  # -> src/draft/trends/data/actuator_catalog.json
python scripts/04_fit_trends.py       # -> src/draft/trends/data/actuator_trends.json
python scripts/05_leave_one_out.py    # -> data/derived/loo_prediction.json
```

Field reference: [`data/actuators/README.md`](data/actuators/README.md).
Methodology and results: [`docs/actuator-dataset.md`](../../docs/actuator-dataset.md).
