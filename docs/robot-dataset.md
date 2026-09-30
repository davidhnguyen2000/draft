# The robot-description dataset

`datasets/robot_descriptions/` measures the mass of everything in a robot that
is not an actuator: links and the trunk. It works from the per-link
inertials in official URDF and MJCF descriptions. The actuator trends in
[`actuator-dataset.md`](actuator-dataset.md) price the motors, and this dataset
prices the rest. The generator's use of these fits is in
[`fitted-trends.md`](fitted-trends.md), and the robot file format is in
[`robot-format.md`](robot-format.md).

The central problem is that a link mass in a description is structure *plus*
the actuators bolted into it. Structure scales with geometry and actuators scale
with torque, so every fit here works on the structure left after the actuators
are subtracted.

## Population

118 descriptions parse. Two populations are fitted, and they are never mixed:

| | motor/structure split | allometry | span |
|---|---|---|---|
| humanoids | 29 | 44 | 16–85 kg |
| quadrupeds | 20 | 24 | 8.8–75 kg |

- **Humanoids:** adam_lite, agibot_a3_ultra, agibot_x1, agibot_x2_ultra,
  azureloong, berkeley_humanoid, booster_k1, booster_t1, dr02_std, elf2,
  engineai_t800, gr1, h1, h1_2, kuavo_s45, kuavo_s53, limx_oli_d03,
  limx_oli_d04, limx_oli_d04_hand, n1, noetix_e1, noetix_n2, tienkung2_pro,
  tienkung3, tienkung_lite, tienkung_pro, unitree_h2, unitree_h2_plus, zq_sa01.
- **Quadrupeds:** anymal_b, anymal_c, anymal_d, deeprobotics_lite3,
  deeprobotics_m20, deeprobotics_m20s, deeprobotics_x30, dfki_quad,
  dobot_rover_x1, magicdog, unitree_a1, unitree_a2, unitree_aliengo, unitree_b1,
  unitree_b2, unitree_go1, unitree_go2, unitree_laikago, xiaomi_cyberdog2,
  zsibot_zsl1.

The allometry needs only a mass and a size. It therefore also keeps robots
whose effort column cannot support the split: Spot, Mini Cheetah, Barkour vB,
Unitree AS2, and the humanoids that fail the actuator-fraction check.

**Exclusions** are recorded with a reason in `data/active_flags.json` or in
stage 3's `EXCLUDE` table. Nothing is dropped silently.

| rule | examples |
|---|---|
| humanoid first released before 2023 (quadrupeds exempt) | Valkyrie, TALOS, NAO, Romeo, OP3, iCub, JAXON, JVRC-1 |
| hydraulic | Atlas DRC/v4, HyQ, HyQReal |
| motor mass sits away from the joint (linkage or remote drive) | Cassie, Minitaur |
| ball-screw linear actuation | Apptronik Apollo |
| wheeled, or no leg chain | B2W, Go2W, WL P311, Astribot S1, Galbot One |
| not vendor hardware (synthetic, hobby, 3D-printed, CHAMP configs) | MuJoCo humanoid, Pupper, ODRI Solo |
| duplicate of a chassis already in the set, or an MJCF twin of a URDF | PNDbotics Adam variants, `*_mj_description` |
| effort column missing or a placeholder (≤ 0, non-finite, > 1000 N·m) | ergoCub, GENE.01, Spot, Mini Cheetah |
| implied actuator fraction > 65% (split only) | G1, AgiBot A3, XBot, Unitree AS2 |

**Fetching.** Stage 1 pulls every `humanoid`, `biped` and `quadruped` entry of
the [`robot_descriptions`](https://github.com/robot-descriptions/robot_descriptions.py)
catalogue, plus sparse clones (`EXTRA_REPOS`) for platforms the catalogue lacks.
Each row is pinned to a commit in `data/manifest.json`, and no model is
committed. How sources are fetched and pinned, and what happens when one
disappears, is in [`fetching-robots.md`](fetching-robots.md). Two cautions:

- `robot_descriptions` leaves its cache sparsely checked out. Stage 1 then
  writes `NO PATH` rows, and stage 2 skips them without an error. Run
  `git -C ~/.cache/robot_descriptions/<repo> sparse-checkout disable`, then
  `grep 'NO PATH'` the stage-1 output before trusting stage 5.
- Stage 1 downloads no meshes, because no fit reads geometry.

## Pipeline

Run from `datasets/robot_descriptions/`, or all at once with
`python scripts/fit_trends.py --only robots`. Stage 1 needs the network, and its
first run clones about 4.4 GB. Stages 2–6 take seconds.

| stage | writes | does |
|---|---|---|
| `01_fetch_descriptions.py` | `data/manifest.json` | resolves each description on disk: path, repo, commit, licence |
| `02_parse_descriptions.py` | `data/segments.json` | loads with Pinocchio, merges joints closer than 2 cm into one segment, records mass, COM, inertia, joints and their limits, length, and the inertia-equivalent box, cylinder and `tip_len`. Reads no meshes |
| `03_classify_segments.py` | `data/segments_classified.json` | assigns segment classes and applies exclusions. Prints unmapped joints |
| `04_attribute_actuators.py` | `data/decomposition.json` | splits motor and structure mass (rules below) |
| `05_fit_trends.py` | `src/draft/trends/data/link_trends.json`, `data/segment_rows.json` | fits both populations: humanoids at top level, quadrupeds under `quadruped` |
| `06_leave_one_out.py` | `data/loo_prediction.json` | leave-one-robot-out prediction of total mass |

## Segment classes and attribution

A segment is named after the joint at its **proximal** end. The body the hip
joints drive is the thigh, and the body the knee drives is the shank, so a
segment's `length` is its physical link length.

- **Humanoid classes:** `pelvis`, `torso`, `head`, `thigh`, `shank`, `foot`,
  `upper_arm`, `forearm`, `hand`, and the short links between two joints of the
  same role: `hip_`, `knee_`, `ankle_`, `shoulder_`, `elbow_`, `wrist_`, `neck_link`
  and `waist_link`. Each is a link like any other, and its structural mass is the
  same residual. In a 3-DOF hip whose axes sit a few centimetres apart, only the
  segment whose child is the knee counts as a thigh. The others are hip links.
- **Quadruped classes:** `pelvis` (the body), `hip_link`, `thigh`, `shank`.
  `QUAD_ROLES` maps each vendor's joint names onto these, and a robot's own
  `ROBOT_ROLES` entry takes precedence. A 12-DOF quadruped must classify as
  exactly `{hip_link: 4, pelvis: 1, thigh: 4, shank: 4}`.

Stages 4 and 5 price each joint's actuator and fit what is left in four steps:

1. **Disclosed line-up.** If the builder publishes its actuators (Berkeley
   Humanoid), each joint gets the lightest member whose peak torque covers the
   joint's effort limit.
2. **Class trend.** Otherwise `m = 0.0653·τ^0.666`, the adopted `mass_trend` of
   `actuator_trends.json` (Table I, n = 103). The five servo-built robots in
   `SERVO_ROBOTS` are priced instead by a fit over the 18 Robotis servos in
   `data/servo_actuators.json`.
   The URDF effort limit is taken at face value, so actuator mass is a lower
   bound. The sweep over `TAU_FACTORS` = 1.0 / 1.25 / 1.5 moves the humanoid
   actuator fraction from 54% to 62% to 70%.
3. **Charge it to a segment.** Humanoids use `half` (50/50 between the proximal
   and distal segments). A quadruped leg actuator is charged to the segment that
   carries it, and that is **stated per robot** in `QUAD_ACTUATOR_HOSTS`, never
   inferred: a remotized knee's actuator rides at the hip, inside the thigh or
   the hip body, and a description does not say which. Each entry gives the knee
   drive (`remotized` or `at_joint`), the host class of the hip-pitch and knee
   actuators, and the evidence read from the description (where the thigh's
   centre of mass sits along it; a body lighter than the actuator it would
   otherwise hold). MagicDog, ZSL-1, Rover X1 and ANYmal D carry the hip-pitch
   unit in the thigh body; Aliengo carries its knee unit in the hip body. A
   fitted quadruped with no entry stops the stage. `split_negatives` records the
   negative-residual count under `half`, `proximal` and `distal` as a check.
4. **Fit the knee the generator has.** The generated quadruped's knee is
   remotized, so the quadruped `thigh` and `shank` trends are fitted only over
   the 15 robots marked `remotized` (`GENERATOR_KNEE_DRIVE` in stage 5). ANYmal
   B, C and D and M20/M20S carry the knee drive at the knee and are left out of
   those two classes only.

A segment whose structural residual is not positive is dropped. On the
generator's own models, where the true split is known, the estimator recovers
actuator mass to within 1.0% (humanoid) and 0.6% (quadruped).

## Fitted trends

**Structural effective density**, `m_struct / V_box` in kg/m³. This is the
density a member's radius is first solved at, and the trunk's price. The
paper's Table II is the power law `m = cL^e` (`m_struct_vs_L` in
`link_trends.json`), which sets each member's mass; `lambda_linear` keeps the
same fit with e held at 1 for comparison. The humanoid trunk is fit
only over robots that describe a head separately.

| class | humanoid n | median | mean ± SD | quadruped n | median | mean ± SD |
|---|---|---|---|---|---|---|
| trunk | 13 | 716 | 823 ± 300 | 20 | 654 | 752 ± 539 |
| thigh | 54 | 782 | 873 ± 472 | 60 | 345 | 454 ± 409 |
| shank | 50 | 533 | 641 ± 553 | 60 | 644 | 748 ± 331 |
| head | 16 | 301 | 316 ± 114 | — | — | — |
| upper arm | 30 | 448 | 612 ± 549 | — | — | — |
| forearm | 36 | 784 | 671 ± 482 | — | — | — |
| hand | 28 | 571 | 830 ± 834 | — | — | — |
| foot | 42 | 301 | 312 ± 231 | — | — | — |

With their actuators left in, humanoid arm links measure 1370–1620 kg/m³, two to
three times their structural values. Trunk is `torso` (humanoid), `pelvis` (quadruped).

**Whole robot** (`whole_robot` in `link_trends.json`):

| quantity | humanoid | quadruped |
|---|---|---|
| actuator mass fraction, median (p10–p90) | 54% (46–64%), n = 29 | 45% (28–59%), n = 20 |
| trunk mass fraction (pelvis + torso + head) | 41% (32–49%) | 47% (42–55%) |
| allometry | `m = 37.1·H^1.33`, H = neutral-pose vertical joint span 0.50–1.66 m; exponent CI [1.12, 1.54], R² 0.79, σ_log 0.20, n = 44 | `m = 136·L_leg^2.47`, L_leg = thigh + shank 0.29–0.74 m; CI [2.03, 2.91], R² 0.85, σ_log 0.27, n = 24 |
| thigh / shank λ = m_struct/L, median | 9.34 / 2.61 kg/m | 1.93 / 1.20 kg/m |
| distal taper of λ, thigh → shank (the span the generator adopts) | 0.21 (CI 0.17–0.25) | 0.56 (CI 0.52–0.60) |

A quadruped's vertical joint span is not a size: as a predictor of mass it gives
R² 0.08. Humanoid per-segment mass fractions, against Winter (2009) for humans:
thigh 8.6% vs 10.0%, shank 5.2% vs 4.65%, foot 1.8% vs 1.45%. The trunk is 41%
of a robot and 58% of a human.

## Validation (`06_leave_one_out.py`)

Each robot is held out in turn and the trends are refit on the rest. Its total
mass is then predicted from the inputs the generator takes, and compared with
the mass in its own description. Actuator mass comes from the actuator trend,
which is never refit here.

| predictor | inputs | humanoids (29) | quadrupeds (20) | humanoid trends on quadrupeds |
|---|---|---|---|---|
| `null` | population median | 1.42× | 1.77× | 2.20× |
| `allometry` | size | 1.19× | 1.43× | 3.74× (bias 0.29) |
| `struct_L` | link lengths + torques, `m = cL^e` | 1.14× | 1.19× | 1.35× |
| `struct_V` | segment volumes + torques | 1.27× | 1.21× | 1.36× |
| `struct_LV` | lengths, trunk volume + torques — what the generator does | **1.22×**, all but one within 2× | **1.18×**, all within 2× | 1.38× |
| `act_budget` | torques ÷ median actuator fraction | 1.11× | 1.29× | 1.30× |

- Fitting the two body plans separately is what gets the quadrupeds from 1.38×
  to 1.18×. Pricing the trunk by volume costs the humanoids 1.14× → 1.22× and
  buys the quadrupeds 1.19× → 1.18×. On quadrupeds the humanoid allometry fails outright, which shows the
  suite is capable of failing.
- The dimensionless constants hold out well: actuator fraction 1.11× and trunk
  fraction 1.16×.
- The ranking uses fold error, not R². Restricted to robots of 20 kg or more,
  allometry's R²(log) falls from 0.72 to 0.44 while its GMFE improves from 1.19×
  to 1.16×.

## Adding a robot

1. If the robot is in the `robot_descriptions` catalogue under a
   humanoid/biped/quadruped tag, it is already pulled. Otherwise add it to
   `EXTRA_REPOS` in stage 1 and `EXTRA_MODELS` in stage 2. Take the newest complete
   revision of each product, one per product, and check the per-link masses to
   rule out a chassis already in the set.
2. Run stages 1–3 and read the unmapped-joints report. Add a `ROBOT_ROLES`
   entry for numbered joint names rather than letting the generic patterns
   guess.
3. Run stage 4 and check `effort_quality` and the implied actuator fraction. If
   the builder publishes its line-up, add it to `DISCLOSED` with the source.
4. Re-run stages 5 and 6, then `pytest`. `tests/test_trends.py` pins the
   Table II medians.

## Limitations

- A URDF effort limit is an operating limit chosen by the author, typically
  0.6–1.0× of the actuator's peak. Actuator mass is therefore a lower bound and
  structure an upper bound.
- λ is the weakest trend family. Thigh p10–p90 is 4.4–13.9 kg/m, and it inherits
  both the attribution error and differences in how authors split a hip between
  hip links and thigh. λ for `pelvis` and `torso` is meaningless, because their
  length is the distance to a nearby hip.
- The humanoid allometry describes a 16–85 kg band and must not be extrapolated
  below it.
- The feasibility bands built from these fits rule out the impossible without
  certifying the plausible. The 95% allometry band is ×1.47 wide for humanoids
  and ×1.69 for quadrupeds, so a 20–40% mass error passes.
- The quadruped population is thin and concentrated by vendor: Unitree and
  DeepRobotics supply 11 of 24. A public sweep found few more, because most
  quadruped makers ship an SDK and no model.
- Descriptions are validated against other descriptions, not against weighed
  hardware.
