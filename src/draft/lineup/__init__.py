"""The paper's §V locomotion study: three designs, ten seeds, four tasks.

Sweep driver, evaluation battery, frozen record and the numbers read from it.
The environment is :mod:`draft.tasks.quadruped`.

    study               run names, log dirs, record paths
    machines            the three machines' pinned fingerprints
    sweep               the driver: designs -> train -> evaluate -> traces -> walk
    evaluate            one absolute battery, in m/s and m, for every design
    freeze_results      raw cells -> the one committed record
    paper_numbers       that record -> every number §V quotes
    collect_traces      a rollout's per-joint time series at one pinned condition
    utilization_report  those traces -> how close a joint runs to its envelope
    record_walk         a flat walk, for the walk limits
    artifacts           pack/fetch the released archives
    scenarios           the absolute conditions a design is traced at

`scripts/train_all.py` is the thin entry point over all of it.
"""
