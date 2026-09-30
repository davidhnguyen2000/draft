# `draft.robots` — the shipped specifications

**The claim:** the engine carries no body plan of its own. A humanoid and a
quadruped were each added by writing declarative files into a directory, with no
engine code touched — which is the §III claim that a machine nobody has built is
a set of these files rather than a change to the compiler.

A robot **is** its directory:

```
<name>/
  parameters.yaml   a flat table of numbers: link lengths, densities,
                    joint ranges, motor classes
  tree.yaml         the kinematic tree, whose numeric fields are arithmetic
                    expressions over that table
  mesh.yaml         optional: the root body as stacked cross-sections to loft,
                    for the parts a cylinder approximates badly
```

No Python. Discovery is "every directory with a parameter table and a tree", so
adding one needs no registry entry — `scripts/generate_robot.py`, the editor and the task
registry all find it. Ask for one by name (`draft.robots.robot_dir("quadruped")`)
rather than rebuilding the path, so it resolves the same from a checkout and from
an installed wheel.

| robot | what it is |
|---|---|
| `humanoid` | 29 DoF. The §IV-F twin targets (Unitree G1, H2) are built from this tree. |
| `quadruped` | 12 DoF. The base for Go2 and B2, and for the three designs §V trains — Cheetah, Bear and Giraffe, each an override file in `experiments/quadruped_variants/`. |

Both state a FIXED tree and vary only by parameters. Variable topology — a part
*count* that is itself a parameter, through `for_each` — and attaching a whole
MJCF file as a `module:` both exist in the engine and are used by no robot here:
they are the hooks for a morphology layer above this one.

The full format reference is
[`docs/robot-format.md`](../../../docs/robot-format.md).
