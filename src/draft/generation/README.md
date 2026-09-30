# `draft.generation`

The compiler: reads a robot directory (`parameters.yaml`, `tree.yaml`, optional
`mesh.yaml` and `modules/*.xml`) and writes a checked MJCF. It carries no body
plan of its own.

| entry point | |
|---|---|
| `python scripts/generate_robot.py --robot <name>` | CLI |
| `generator.py:RobotGenerator(dir).generate(out, params_override=...)` | Python |

| module | job |
|---|---|
| `generator.py` | load, size actuators, resolve, prune, gate, write |
| `expr_eval.py` | `${expr}` resolver |
| `tree_ops.py` | mirroring, motor-class expansion, link inset and drawn radius |
| `spec_builder.py` | resolved tree → `MjSpec` → compiled model |
| `mesh_spec.py`, `layered_mesh.py`, `root_mesh.py` | `mesh.yaml` loft |
| `mounting.py` | outlines and edge placement helpers |
| `packing.py` | actuator and structure overlap gates |
| `mjcf_assets.py` | loads models from disk, bypassing MuJoCo's mesh cache |

Format reference: [`docs/robot-format.md`](../../../docs/robot-format.md).
