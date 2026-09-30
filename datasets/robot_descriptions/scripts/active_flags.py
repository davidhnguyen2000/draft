"""Side-car ACTIVE flags: rows that stay in the dataset but are not used.

Excluded rows are listed with a reason in `data/active_flags.json` rather than
deleted, so exclusions stay auditable; the flags are applied at load time.

File format (`data/active_flags.json`):

    {
      "schema":   "active-flags/1",
      "dataset":  "datasets/actuators/data/actuators/*.yaml",
      "key":      "name",
      "updated":  "2026-08-13",
      "inactive": {"<key>": "<why it is not used>", ...}
    }

An absent file or empty `inactive` map means every row is active. Duplicated in
datasets/actuators/ and datasets/robot_descriptions/ so each runs standalone.
"""

import json
import os

SCHEMA = "active-flags/1"


def load_inactive(path, expect_dataset=None):
    """Return {key: reason} for every row flagged inactive. Missing file -> {}."""
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        doc = json.load(f)
    schema = doc.get("schema")
    if schema != SCHEMA:
        raise ValueError(f"{path}: expected schema {SCHEMA!r}, found {schema!r}")
    if expect_dataset and doc.get("dataset") != expect_dataset:
        raise ValueError(f"{path}: flags are for {doc.get('dataset')!r}, not {expect_dataset!r}")
    inactive = doc.get("inactive") or {}
    if not isinstance(inactive, dict):
        raise ValueError(f"{path}: 'inactive' must be an object of key -> reason")
    return {k: (v or "flagged inactive") for k, v in inactive.items()}


def split(records, key_field, inactive, label="rows"):
    """Split `records` on the flags; return (active, dropped). Warns on unknown keys."""
    active, dropped = [], []
    for r in records:
        (dropped if r[key_field] in inactive else active).append(r)
    if dropped:
        print(f"active_flags: {len(dropped)} of {len(records)} {label} flagged inactive")
        for r in dropped:
            print(f"  - {r[key_field]}: {inactive[r[key_field]]}")
    seen = {r[key_field] for r in records}
    unknown = sorted(set(inactive) - seen)
    if unknown:
        print(f"active_flags: WARNING {len(unknown)} flagged key(s) match no row: {', '.join(unknown)}")
    return active, dropped
