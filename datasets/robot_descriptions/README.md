# datasets/robot_descriptions

Structural mass trends for links and trunks, fitted from the inertials of
official URDF/MJCF descriptions (29 humanoids and 20 quadrupeds after the
motor/structure split). Descriptions are fetched at a pinned commit, never
committed: [`docs/fetching-robots.md`](../../docs/fetching-robots.md).

Run from this directory, in order. Stage 1 needs the network and clones about
4.4 GB on first run; `data/segments.json` is committed, so you can start at
stage 3.

```shell
python scripts/01_fetch_descriptions.py   # -> data/manifest.json
python scripts/02_parse_descriptions.py   # -> data/segments.json
python scripts/03_classify_segments.py    # -> data/segments_classified.json
python scripts/04_attribute_actuators.py  # -> data/decomposition.json
python scripts/05_fit_trends.py           # -> src/draft/trends/data/link_trends.json
python scripts/06_leave_one_out.py        # -> data/loo_prediction.json
```

Methodology and results: [`docs/robot-dataset.md`](../../docs/robot-dataset.md).
