"""The four Unitree platforms the twins are built against: G1, H2, Go2, B2.

A platform qualifies if it has an official description pinned in
`datasets/robot_descriptions/data/manifest.json` with real effort/velocity
limits, and a usable MuJoCo model. G1 and H2 match the parametric tree's
3-waist / 6-leg / 7-arm layout exactly (H2 adds a 2-DOF neck).

Each role map sends joint names (regex, first match wins) to the canonical
roles in `measure.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


CACHE = Path.home() / ".cache" / "robot_descriptions"
MENAGERIE = CACHE / "mujoco_menagerie"
from draft.paths import repo_root
# Extra clones for platforms the `robot_descriptions` catalog lacks (e.g. H2).
_EXTRA = repo_root() / "datasets/robot_descriptions" / "cache_extra"


@dataclass(frozen=True)
class Target:
    key: str
    label: str
    category: str                 # 'humanoid' | 'quadruped'
    urdf: Path                    # measured from this; authoritative for mass and limits
    mjcf: Path | None             # rendered from this
    #: RL adapter source when `mjcf` is unusable (H2's shipped MJCF is closed-loop).
    rl_source: Path | None = None
    role_map: list = field(default_factory=list)  # [(regex, canonical role)], first match wins
    vendor: str = ""
    notes: str = ""
    citation: str = ""
    # Roles the parametric tree has but this platform lacks.
    missing_roles: tuple = ()


# ── humanoid role maps ────────────────────────────────────────────────────────

_G1_ROLES = [
    (r"waist_yaw", "torso_yaw"), (r"waist_roll", "torso_roll"),
    (r"waist_pitch", "torso_pitch"),
    (r"hip_pitch", "hip_pitch"), (r"hip_roll", "hip_roll"), (r"hip_yaw", "hip_yaw"),
    (r"knee", "knee"),
    (r"ankle_pitch", "ankle_pitch"), (r"ankle_roll", "ankle_roll"),
    (r"shoulder_pitch", "shoulder_pitch"), (r"shoulder_roll", "shoulder_roll"),
    (r"shoulder_yaw", "shoulder_yaw"), (r"elbow", "elbow"),
    # Wrist roll/yaw are deliberately crossed: Unitree names wrist axes with arms
    # forward, the tree with arms hanging, so the along-forearm axis is Unitree's
    # `wrist_roll` and the tree's `wrist_yaw`. Roles are physical, not labels.
    (r"wrist_yaw", "wrist_roll"), (r"wrist_pitch", "wrist_pitch"),
    (r"wrist_roll", "wrist_yaw"),
]

# H2: G1's naming plus a 2-DOF neck. Roles are by axis, so its roll-then-pitch
# ankle order needs no special case.
_H2_ROLES = _G1_ROLES + [(r"head_(pitch|yaw)", "neck")]

# ── quadruped role maps ───────────────────────────────────────────────────────

_UNITREE_QUAD_ROLES = [
    (r"_hip_joint", "hip_roll"), (r"_thigh_joint", "hip_pitch"),
    (r"_calf_joint", "knee"), (r"_foot", "aux"),
]
# Unused by REGISTRY; kept for adding an ANYmal target.
_ANYMAL_ROLES = [(r"_HAA", "hip_roll"), (r"_HFE", "hip_pitch"), (r"_KFE", "knee")]

# ── our own generated models ──────────────────────────────────────────────────

GENERATED_HUMANOID_ROLES = [
    (r"torso_yaw", "torso_yaw"), (r"torso_roll", "torso_roll"),
    (r"torso_pitch", "torso_pitch"),
    (r"hip_pitch", "hip_pitch"), (r"hip_roll", "hip_roll"), (r"hip_yaw", "hip_yaw"),
    (r"knee", "knee"),
    (r"ankle_pitch", "ankle_pitch"), (r"ankle_roll", "ankle_roll"),
    (r"shoulder_pitch", "shoulder_pitch"), (r"shoulder_roll", "shoulder_roll"),
    (r"shoulder_yaw", "shoulder_yaw"), (r"elbow", "elbow"),
    (r"wrist_yaw", "wrist_yaw"), (r"wrist_pitch", "wrist_pitch"),
    (r"wrist_roll", "wrist_roll"),
]
GENERATED_QUADRUPED_ROLES = [
    (r"hip_roll", "hip_roll"), (r"hip_pitch", "hip_pitch"), (r"knee", "knee"),
]


REGISTRY: list[Target] = [
    Target(
        key="g1", label="Unitree G1", category="humanoid", vendor="Unitree",
        # rev 1.0 (33.34 kg, hip roll 139 N·m @ 20 rad/s) matches the menagerie
        # MJCF; the manifest pins the older g1_29dof.urdf for the population fit.
        urdf=CACHE / "unitree_ros/robots/g1_description/g1_29dof_rev_1_0.urdf",
        # MJX variant: per-joint armature, realistic PD gains and corrected hip
        # limits, where plain g1.xml uses placeholders.
        mjcf=MENAGERIE / "unitree_g1/g1_mjx.xml",
        role_map=_G1_ROLES,
        notes="29-DOF layout identical to the parametric tree — the clean case. "
              "Measured from the hardware-validated MJX model.",
        citation="Unitree unitree_ros / mujoco_menagerie unitree_g1 (g1_mjx)",
    ),
    Target(
        key="h2", label="Unitree H2", category="humanoid", vendor="Unitree",
        urdf=_EXTRA / "unitree_ros_all/robots/h2_description/H2.urdf",
        # COLLADA URDF (same physics as H2.urdf, painted meshes) instead of the
        # closed-loop `H2_loop.xml`.
        mjcf=_EXTRA / "unitree_ros_all/robots/h2_description/H2_dae.urdf",
        # Same file as `mjcf`, so the trained and plotted H2 are one description.
        rl_source=_EXTRA / "unitree_ros_all/robots/h2_description/H2_dae.urdf",
        role_map=_H2_ROLES,
        notes="G1's layout at full size, plus a 2-DOF neck the parametric tree "
              "has no equivalent for (its head is fixed; the adapter welds the "
              "neck). The shipped MJCF models the knee and ankle as closed "
              "loops, so both the swap pair and the render are built from the "
              "URDF, which describes the same machine as a direct-drive tree — "
              "the twin's linkages are direct drives too, and comparing a loop "
              "model against them would measure the linkage rather than the "
              "robot.",
        citation="Unitree unitree_ros (h2_description)",
    ),
    # ANYmal C is excluded: its hip-pitch axis is offset in 3D from the
    # abduction axis, which the two-orthogonal-link leg cannot express. It
    # remains in the fit population.
    Target(
        key="go2", label="Unitree Go2", category="quadruped", vendor="Unitree",
        urdf=CACHE / "unitree_ros/robots/go2_description/urdf/go2_description.urdf",
        mjcf=MENAGERIE / "unitree_go2/go2.xml",
        role_map=_UNITREE_QUAD_ROLES,
        citation="Unitree unitree_ros / mujoco_menagerie unitree_go2",
    ),
    Target(
        key="b2", label="Unitree B2", category="quadruped", vendor="Unitree",
        urdf=CACHE / "unitree_ros/robots/b2_description/urdf/b2_description.urdf",
        # Unitree's MJCF, used for geometry and rendering only: it drops the
        # URDF's head/imu/radar links (35.9 vs 41.4 kg trunk).
        mjcf=CACHE / "unitree_ros/robots/b2_description_mujoco/xml/b2.xml",
        role_map=_UNITREE_QUAD_ROLES,
        notes="Official MJCF from Unitree's `b2_description_mujoco` package; "
              "mass and joint limits measured from the URDF, which carries the "
              "head and sensor links the MJCF drops.",
        citation="Unitree unitree_ros b2_description / b2_description_mujoco",
    ),
]

BY_KEY = {t.key: t for t in REGISTRY}


def resolve(keys: list[str] | None = None) -> list[Target]:
    if not keys:
        return list(REGISTRY)
    return [BY_KEY[k] for k in keys]
