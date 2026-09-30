# Getting the robot descriptions

No vendor robot model ships with Draft; each belongs to its vendor under its own
licence. The repository stores only where to fetch each one.

```shell
python scripts/setup_data.py            # the four twin targets (three repositories)
python scripts/setup_data.py --all      # plus the full survey population (~36 repos)
python scripts/setup_data.py --list     # what is present and what is missing
python scripts/setup_data.py --only unitree_ros_all
```

| source | for | lands in |
|---|---|---|
| `unitree_ros` | G1, Go2 and B2 URDFs | `~/.cache/robot_descriptions/` |
| `mujoco_menagerie` | G1 and Go2 MJCFs, for rendering | `~/.cache/robot_descriptions/` |
| `unitree_ros_all` | H2 | `datasets/robot_descriptions/cache_extra/` |
| `--all` | the survey the structural trends are fitted on | both |

Clones are sparse and blobless, limited to the paths needed, and **pinned to the
commit the published numbers were measured at**. Meshes are fetched too: twin
measurement fits geometry to them, and a clone without meshes produces a
slightly different twin.

## When a source disappears

A missing source is reported and skipped, never fatal:

| what happened | result |
|---|---|
| repository deleted, renamed or private | `SKIP not found at <url>`, and the rest continue |
| file moved inside the repository | reported, partial clone removed |
| pinned commit gone (force-push) | cloned at HEAD with a warning that numbers may differ |
| no network | reported per source |

`setup_data.py` exits non-zero only when `--only` sources all fail.
`scripts/make_twins.py` reports which targets it skipped and why.
