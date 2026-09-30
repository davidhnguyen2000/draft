"""How fast a design can go, from its own dimensions.

Kept free of mjlab so ``tests/test_speed_shaping.py`` runs without a GPU stack.
``quadruped_entity`` re-exports these names.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import mujoco
import numpy as np

#: Reference leg length the Froude fallbacks are stated against.
REF_LEG_LENGTH = 0.44

#: The velocity task's command ceiling, m/s, shared by every design. Set far
#: above any design's reach so it never caps the measurement.
COMMON_V_CEILING = 15.0

#: Headroom on the speed-shaping scale: the forward-speed bonus is
#: ``clamp(v, 0, v_cap) / v_cap``, so ``v_cap`` must sit above what the design
#: already reaches.
SHAPE_HEADROOM = 1.5


def _resolve_robot_xml(xml_path: Path) -> Path:
    """If xml_path is a scene file that includes a robot xml, resolve the robot xml."""
    text = xml_path.read_text()
    m = re.search(r'<include\s+file="([^"]+)"', text)
    if m:
        return xml_path.parent / m.group(1)
    return xml_path


def measure_quadruped_leg_length(xml_path: Path | str) -> float:
    """Return thigh + shank length by reading fl_lower_leg_link and fl_foot from the model.

    Equals upper_leg_link_length + lower_leg_link_length. Accepts scene XMLs.
    """
    xml_path = _resolve_robot_xml(Path(xml_path))
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    lower_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "fl_lower_leg_link")
    foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "fl_foot")
    # body_pos[lower_id] is the static offset from fl_upper_leg_link → equals upper_leg_link_length
    # site_pos[foot_id]  is the static offset from fl_lower_leg_link → equals lower_leg_link_length
    thigh = float(np.linalg.norm(model.body_pos[lower_id]))
    shank = float(np.linalg.norm(model.site_pos[foot_id]))
    return thigh + shank


#: Two speed limits; a design is bound by whichever is lower: the knee motor's
#: foot-tip speed, and a Froude number of about 1.3. Both constants are
#: calibrated on three designs' measured top speeds, so treat them as a scale for
#: sizing a reward, not as a claim about what a quadruped can do.
_ACT_SPEED_FRAC = 0.74          # of omega_NL * L_shank
_FROUDE_SPEED_COEF = 1.13       # of sqrt(g * L_leg), i.e. Fr ~ 1.27


def estimate_top_speed(xml_path: Path | str) -> float:
    """Top body speed to shape the speed reward against: min(actuator, Froude) limit.

    Separate from ``measure_max_body_speed``, which still sets ``ctx.v_max``.
    """
    import yaml
    xml_path = _resolve_robot_xml(Path(xml_path))
    leg = measure_quadruped_leg_length(xml_path)
    froude = _FROUDE_SPEED_COEF * math.sqrt(9.81 * leg)

    params_yaml = xml_path.parent / "parameters_from_gui.yaml"
    if not params_yaml.exists():
        return froude
    params = yaml.safe_load(params_yaml.read_text())
    knee_cls = params.get("knee_mot", "L")
    knee_vel = float(params.get(f"{knee_cls}_motor_velocity", 35.0))
    shank = float(params.get("lower_leg_link_length", leg / 2))
    return min(_ACT_SPEED_FRAC * knee_vel * shank, froude)


