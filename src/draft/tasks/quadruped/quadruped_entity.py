"""Parametric quadruped EntityCfg factory.

Replaces the XML's motor actuators with ``DcMotorActuatorCfg`` groups (linear
torque-speed curve from stall torque to no-load speed ``omega_NL``, read from the
generator's ``<joint>_velocity_limit`` numerics). PD gains scale with peak torque:
``kp = KP_PER_NM * effort``, ``kd = KD_PER_NM * effort``. Without ``omega_NL``
it falls back to flat-clamp position actuators and warns.

Expects 12 joints ``{fl,fr,rl,rr}_{hip_roll,hip_pitch,knee}`` and foot spheres
``{fl,fr,rl,rr}_lower_leg_link_sphere``.
"""

from __future__ import annotations

import math
import warnings
from collections import defaultdict
from pathlib import Path

import mujoco

# Geometry and speed limits that need no simulator, so a laptop can check them.
from draft.tasks.quadruped.speed_limits import (  # noqa: F401
    REF_LEG_LENGTH, _resolve_robot_xml, estimate_top_speed,
    measure_quadruped_leg_length)

from mjlab.actuator import BuiltinPositionActuatorCfg, DcMotorActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

# Initial trunk height at the reference leg length, m; scaled with leg length.
REF_INIT_HEIGHT: float = 0.45

_LEGS = ("fl", "fr", "rl", "rr")

# PD gain scaling — calibrated so L-class motor (70 Nm) gives kp=100, kd=5.
KP_PER_NM: float = 100.0 / 70.0
KD_PER_NM: float = 5.0 / 70.0

# Only the foot spheres collide (avoids overflowing contacts on hfield terrain);
# falls are detected from state in the env, not from contacts.
_TERRAIN_COLLISION = CollisionCfg(
    geom_names_expr=(r".*_lower_leg_link_sphere",),  # foot contact spheres (×4)
    contype=1,
    conaffinity=1,
    condim=3,
    priority=1,
    disable_other_geoms=True,
)


# Heuristic foot-tip-speed -> body-speed factor for a trot.
_TROT_EFFICIENCY: float = 0.25


def measure_max_body_speed(xml_path: Path | str) -> float:
    """Heuristic max body speed: knee no-load speed x shank length x _TROT_EFFICIENCY.

    Reads ``parameters_from_gui.yaml`` next to the XML; without it, falls back to
    ``1.35 * sqrt(s)``.
    """
    import yaml
    xml_path = _resolve_robot_xml(Path(xml_path))
    params_yaml = xml_path.parent / "parameters_from_gui.yaml"
    if not params_yaml.exists():
        leg = measure_quadruped_leg_length(xml_path)
        return 1.35 * math.sqrt(leg / REF_LEG_LENGTH)
    params = yaml.safe_load(params_yaml.read_text())
    knee_cls = params.get("knee_mot", "L")
    knee_vel = float(params.get(f"{knee_cls}_motor_velocity", 35.0))
    lower_leg = float(params.get("lower_leg_link_length", REF_LEG_LENGTH / 2))
    return knee_vel * lower_leg * _TROT_EFFICIENCY


def measure_max_joint_speed(xml_path: Path | str) -> float:
    """Largest no-load joint speed (rad/s) declared by the model.

    Used to size anything in rad/s (e.g. observation noise) per design.
    """
    xml_path = _resolve_robot_xml(Path(xml_path))
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    limits = _velocity_limits_from_xml(model)
    vals = [v for v in limits.values() if v > 0.0]
    return max(vals) if vals else 35.0


def measure_foot_sphere_radius(xml_path: Path | str) -> float:
    """Return the foot sphere radius from the fl_lower_leg_link_sphere geom.

    Read from the compiled model so ``class`` defaults (e.g. in models adapted by
    ``draft.twins.rl_adapter``) are applied.
    """
    xml_path = _resolve_robot_xml(Path(xml_path))
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "fl_lower_leg_link_sphere")
    if gid < 0:
        raise ValueError(f"fl_lower_leg_link_sphere geom not found in {xml_path}")
    return float(model.geom_size[gid, 0])


#: Swing room (rad) the nominal stance keeps on both sides of every joint; must
#: exceed the action scale (0.175 rad) so every joint can move both ways.
_STANCE_HEADROOM: float = 0.20

#: Crouch depth (knee flexion, rad) the parametric designs were tuned at.
_NOMINAL_CROUCH: float = 1.0


def _soft_range(model: mujoco.MjModel, joint: str,
                factor: float = 0.9) -> tuple[float, float]:
    """The joint range the ``dof_pos_limits`` reward actually leaves usable."""
    j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
    lo, hi = (float(x) for x in model.jnt_range[j])
    mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo) * factor
    return mid - half, mid + half


def _stance_height(model: mujoco.MjModel, data: mujoco.MjData, k: float) -> float:
    """Torso height above the lowest foot, in the symmetric crouch of depth ``k``."""
    data.qpos[:] = 0.0
    if model.nq >= 7:
        data.qpos[3] = 1.0
    for leg in _LEGS:
        for role, val in (("hip_pitch", 0.5 * k), ("knee", -k), ("hip_roll", 0.0)):
            j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{leg}_{role}")
            data.qpos[model.jnt_qposadr[j]] = val
    mujoco.mj_forward(model, data)
    feet = []
    for leg in _LEGS:
        g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{leg}_lower_leg_link_sphere")
        feet.append(float(data.geom_xpos[g][2] - model.geom_size[g][0]))
    torso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    return float(data.xpos[torso][2]) - min(feet)


def default_stance(xml_path: Path | str,
                   stance_xml: Path | str | None = None) -> dict[str, float]:
    """The nominal crouch for this robot: ``knee = -k``, ``hip_pitch = k/2``.

    ``k`` is ``_NOMINAL_CROUCH`` clamped so every joint keeps
    ``_STANCE_HEADROOM`` from its soft limits (e.g. a Go2's knee range forces it).
    """
    # The stance is the action-space zero, so compared robots must share it;
    # ``stance_xml`` adds a model whose limits also constrain it.
    models = [mujoco.MjModel.from_xml_path(str(_resolve_robot_xml(Path(x))))
              for x in ({xml_path} if stance_xml is None else {xml_path, stance_xml})]

    # Feasible for EVERY model in the set: intersect their bounds.
    k_min, k_max = 0.0, float("inf")
    for model in models:
        knee_lo, knee_hi = _soft_range(model, "fl_knee")
        hip_lo, hip_hi = _soft_range(model, "fl_hip_pitch")
        k_min = max(k_min, -knee_hi + _STANCE_HEADROOM, 2.0 * (hip_lo + _STANCE_HEADROOM))
        k_max = min(k_max, -knee_lo - _STANCE_HEADROOM, 2.0 * (hip_hi - _STANCE_HEADROOM))
    xml_path = _resolve_robot_xml(Path(xml_path))
    if k_max <= k_min:
        k = max(0.5 * (k_min + k_max), 0.0)
        warnings.warn(
            f"{xml_path.name}: knee/hip ranges leave no room for a stance with "
            f"{_STANCE_HEADROOM} rad of headroom at both ends; using k={k:.2f}.",
            UserWarning, stacklevel=2)
    else:
        k = float(min(max(_NOMINAL_CROUCH, k_min), k_max))
    return {".*hip_pitch": 0.5 * k, ".*knee": -k, ".*hip_roll": 0.0}


def _build_robot_spec(xml_path: Path, geom_group: int | None = None) -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(xml_path))
    for act in list(spec.actuators):
        spec.delete(act)
    if geom_group is not None:
        # Move robot geoms out of group 0 so terrain ray-casts do not hit them.
        for geom in spec.geoms:
            geom.group = geom_group
    return spec


def _velocity_limits_from_xml(model: mujoco.MjModel) -> dict[str, float]:
    """Read per-joint no-load speed (omega_NL) from '<joint>_velocity_limit' numerics."""
    suffix = "_velocity_limit"
    out: dict[str, float] = {}
    for i in range(model.nnumeric):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_NUMERIC, i)
        if name and name.endswith(suffix):
            adr = int(model.numeric_adr[i])
            out[name[: -len(suffix)]] = float(model.numeric_data[adr])
    return out


def _actuator_cfgs_from_xml(
    xml_path: Path,
) -> tuple[DcMotorActuatorCfg | BuiltinPositionActuatorCfg, ...]:
    """Derive DcMotorActuatorCfg groups (torque-speed model) from the robot XML.

    Peak torque sets kp, kd, saturation_effort and effort_limit; joints with
    identical parameters are merged. Efforts come from ``<actuator>`` elements,
    else from joint ``actuatorfrcrange``.
    """
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    vel_limits = _velocity_limits_from_xml(model)

    joints: list[tuple[str, float]] = []
    if model.nu > 0:
        for i in range(model.nu):
            # target_names_expr matches joint names, not actuator names
            jnt_id = int(model.actuator_trnid[i, 0])
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id)
            if not name:
                continue
            joints.append((name, float(model.actuator_forcerange[i, 1])))
    else:
        # No explicit actuators — derive from joints using actuatorfrcrange.
        for i in range(model.njnt):
            if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
                continue
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if not name:
                continue
            joints.append((name, float(model.jnt_actfrcrange[i, 1])))

    have_dc = bool(joints) and all(n in vel_limits and e > 0.0 for n, e in joints)

    if not have_dc:
        warnings.warn(
            "No-load speeds (omega_NL) missing or non-positive efforts in XML; "
            "regenerate the model to enable the DcMotor torque-speed curve. "
            "Falling back to flat-clamp position actuators.",
            UserWarning,
            stacklevel=2,
        )
        pos_groups: dict[tuple[float, float, float], list[str]] = defaultdict(list)
        for name, effort in joints:
            kp = round(KP_PER_NM * effort, 4)
            kd = round(KD_PER_NM * effort, 4)
            pos_groups[(kp, kd, effort)].append(name)
        return tuple(
            BuiltinPositionActuatorCfg(
                target_names_expr=tuple(names),
                stiffness=kp,
                damping=kd,
                effort_limit=effort if effort > 0.0 else None,
            )
            for (kp, kd, effort), names in pos_groups.items()
        )

    # (kp, kd, effort, velocity) -> joint names
    dc_groups: dict[tuple[float, float, float, float], list[str]] = defaultdict(list)
    for name, effort in joints:
        kp = round(KP_PER_NM * effort, 4)
        kd = round(KD_PER_NM * effort, 4)
        dc_groups[(kp, kd, effort, round(vel_limits[name], 4))].append(name)
    return tuple(
        DcMotorActuatorCfg(
            target_names_expr=tuple(names),
            stiffness=kp,
            damping=kd,
            effort_limit=effort,
            saturation_effort=effort,
            velocity_limit=velocity,
        )
        for (kp, kd, effort, velocity), names in dc_groups.items()
    )


def make_quadruped_entity_cfg(
    xml_path: Path | str,
    init_height: float | None = None,
    stance_xml: Path | str | None = None,
    geom_group: int | None = None,
) -> EntityCfg:
    """Create an EntityCfg for the parametric quadruped from its robot XML.

    Args:
        xml_path:    Path to the robot XML (or a scene XML that includes it).
        init_height: Initial root z. None scales REF_INIT_HEIGHT by leg length.
        stance_xml:  Extra model constraining the stance; see default_stance().
        geom_group:  Put every robot geom in this group. None keeps the XML's.
    """
    xml_path = _resolve_robot_xml(Path(xml_path))
    if init_height is None:
        leg_length = measure_quadruped_leg_length(xml_path)
        init_height = REF_INIT_HEIGHT * (leg_length / REF_LEG_LENGTH)
    actuators = _actuator_cfgs_from_xml(xml_path)
    captured = xml_path

    return EntityCfg(
        spec_fn=lambda: _build_robot_spec(captured, geom_group),
        articulation=EntityArticulationInfoCfg(
            actuators=actuators,
            soft_joint_pos_limit_factor=0.9,
        ),
        init_state=EntityCfg.InitialStateCfg(
            pos=(0.0, 0.0, init_height),
            joint_pos=default_stance(xml_path, stance_xml),
            joint_vel={".*": 0.0},
        ),
        collisions=(_TERRAIN_COLLISION,),
    )
