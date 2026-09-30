# `draft.tasks` — §V, making design tradeoffs legible

**The claim:** three quadrupeds from one specification, trained under an
identical curriculum, acquire measurably different capabilities — and no two
axes rank them the same way.

A task is a package that exposes `make_env_cfg(play=False, **kwargs)` and
`make_runner_cfg()` from its `__init__.py` and carries a `config.yaml`. Register
it in `tasks.yaml` and `scripts/train.py` needs no edit. `package:` there is
relative to this package, so a task resolves from an installed wheel as well as
from a checkout.

```
quadruped/            the flat base policy: velocity tracking on gently uneven ground
quadruped/velocity/   commanded speed ramped toward each design's motor-limited ceiling
quadruped/terrain/    procedural rough terrain, absolute height ceilings
quadruped/push/       mid-episode velocity kicks ramped to each design's stability ceiling
```

The three challenge tasks share the base observation and action space
**exactly**, which is what permits warm-starting from the flat policy and what
makes each curriculum's endpoint read directly as a capability number rather than
as a training artefact.

Two properties of the generator make the comparison fair rather than merely
different, and both are worth knowing before changing anything here:

- **Gait-shaping rewards are Froude-scaled by leg length**, which the generator
  knows exactly rather than measuring off a mesh, so a long-legged design is not
  rewarded for longer strides simply because it is large.
- **PD gains scale with each joint's peak torque** (`k_p = 1.43·τ_peak`,
  `k_d = 0.071·τ_peak`), so a given tracking error commands the same share of
  available torque on every design, and a design that swaps in a stronger
  actuator is not handed a controller that uses less of it.

Which robot a task trains on comes from its run config's `env_kwargs.robot_xml`,
not from a flag. `task_geometry_xml` lets a pair of robots be scored on a task
defined by the *same* machine, which is what makes a twin/original comparison a
measurement of the robot rather than of the task.

`gait_metrics.py` sits at this level rather than inside `quadruped/` because it
needs nothing but numpy, and the figure scripts and tests that read it should not
have to import mjlab.

The registry holds quadruped tasks only. The paper trains quadrupeds, and its
humanoid appears as a generation example and as a twin target, never as a
policy — so there is no humanoid task here to mistake for one.
