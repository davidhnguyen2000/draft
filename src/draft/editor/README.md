# `draft.editor`

The browser editor. Not evaluated in the paper, but the normal way to build a
design.

```shell
python scripts/robot_tuner.py    # open the page it prints; :8081 by default
```

One page: the robot on the left (a viser scene, also reachable on its own port)
and the parameters on the right. Every number shown — actuator mass, envelope,
reflected inertia, frontier distance, the full feasibility report — is computed
by the same `draft.trends` code the generator runs, on every edit.

The page shows:

- the actuator classes as one editable table, with a torque–speed plot;
- the four fitted actuator relations with this design's classes marked on them;
- a mass breakdown by part;
- each feasibility check as a band with the design's value marked;
- a marker on every field that differs from the shipped robot.

**New from defaults** starts from the robot's preset: `presets/g1.yaml` (the
Unitree G1 twin) for the humanoid, `presets/cheetah.yaml` for the quadruped.

**Save…** builds the design into `generated/<robot>_<name>/`, an ordinary
generated robot that `scripts/train.py` can point at. **Generate** overwrites
the selected folder under `generated/`.

## Layout

- `core/` is headless. `schema.py` classifies every parameter as free,
  calibrated or trend-owned (hence whether it gets a widget); `session.py` holds
  the design being edited; `fits.py` draws the actuator relations.
- `viewer.py` is the viser scene and physics loop.
- `web/` serves the page and its websocket on one port. The browser sends
  `{patch, key, value}` and receives the whole state back, so every number is
  solved in one place.
