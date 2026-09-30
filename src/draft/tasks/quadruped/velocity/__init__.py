"""Top-speed challenge task for the parametric quadruped.

The base env with the velocity command ramped, in absolute m/s, up to the shared
``speed_limits.COMMON_V_CEILING``, so every design is asked for the same speeds.
On this task only, the stride-limiting gait rewards are loosened and a
forward-speed bonus is added; those are sized to the design's own
``estimate_top_speed``, not the command ceiling. The ramp relies on
``scripts/train.py`` zeroing the step counter on a warm start.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.reward_manager import RewardTermCfg

from draft.tasks.quadruped.base_env import make_base_env_cfg
from draft.tasks.quadruped.speed_limits import (COMMON_V_CEILING,
                                                SHAPE_HEADROOM,
                                                estimate_top_speed)
from draft.tasks.quadruped.curriculums import EpisodeSpeedMetric
from draft.tasks.quadruped.runner_cfg import make_runner_cfg  # noqa: F401  (task API)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv




def _forward_speed_bonus(env: "ManagerBasedRlEnv", v_cap: float) -> torch.Tensor:
    """Body speed toward the command, clamped to v_cap and normalised by it.

    Active only when |cmd_x| > 0.5 * v_cap, so slow commands are not pushed to sprint.
    """
    from mjlab.entity import Entity
    asset: Entity = env.scene["robot"]
    cmd_x = env.command_manager.get_command("twist")[:, 0]
    v_x = asset.data.root_link_lin_vel_b[:, 0]
    aligned = v_x * torch.sign(cmd_x)                       # speed toward command
    bonus = torch.clamp(aligned, min=0.0, max=v_cap) / v_cap
    gate = (cmd_x.abs() > 0.5 * v_cap).float()
    return bonus * gate


def make_env_cfg(*, play: bool = False, robot_xml, task_geometry_xml=None,
                 **base) -> ManagerBasedRlEnvCfg:
    """The speed task: one ramp on commanded velocity.

    ``clamp_command_to_vmax`` has no effect here; the ramp overwrites ``lin_vel_x``.

    Other kwargs pass straight to ``make_base_env_cfg``, which rejects unknown keys.
    """
    cfg, ctx = make_base_env_cfg(play=play, robot_xml=robot_xml,
                                 task_geometry_xml=task_geometry_xml, **base)
    if play:
        return cfg

    # Only the command ramps (the base has no curriculum; the pop states that).
    cfg.curriculum.pop("terrain_levels", None)

    # Command (demand) uses the shared ceiling; reward shaping uses this design's
    # own estimate, since terms sized off an unreachable speed go flat.
    v_max = COMMON_V_CEILING
    v_shape = SHAPE_HEADROOM * estimate_top_speed(task_geometry_xml or robot_xml)
    # ── Command ramp: absolute m/s, the same four rungs for every design ──
    # common_step_counter advances by num_steps_per_env (24) per iteration.
    lat, yaw = 0.4, 0.4  # modest absolute lateral / yaw commands (speed is the axis)
    stages = [(0, 0.4), (400, 0.6), (900, 0.8), (1400, 1.0)]
    # Seed the command range with the first rung so no design starts on the
    # base's per-design range before the curriculum first runs.
    twist = cfg.commands["twist"]
    twist.ranges.lin_vel_x = (-stages[0][1] * v_max, stages[0][1] * v_max)
    twist.ranges.lin_vel_y = (-lat, lat)
    twist.ranges.ang_vel_z = (-yaw, yaw)
    command_vel = ctx.base_curriculum["command_vel"]
    command_vel.params["velocity_stages"] = [
        {"step": it * 24,
         "lin_vel_x": (-f * v_max, f * v_max),
         "lin_vel_y": (-lat, lat),
         "ang_vel_z": (-yaw, yaw)}
        for it, f in stages
    ]
    cfg.curriculum["command_vel"] = command_vel

    # ── Loosen the stride-capping gait shaping on this task so fast gaits pay ──
    # Tracking tolerance sized to the design's own speed estimate.
    cfg.rewards["track_linear_velocity"].params["std"] = 0.4 * v_shape
    # Allow longer flight phases.
    cfg.rewards["air_time"].params["threshold_max"] = 0.5
    cfg.rewards["air_time"].weight = 0.3
    # Weaker smoothing penalty so the legs may cycle quickly.
    cfg.rewards["action_rate_l2"].weight = -0.01
    # Forward-speed bonus up to the design's own shaping speed.
    cfg.rewards["forward_speed"] = RewardTermCfg(
        func=_forward_speed_bonus, weight=0.6, params={"v_cap": v_shape}
    )

    cfg.curriculum["episode_speed"] = CurriculumTermCfg(func=EpisodeSpeedMetric, params={})
    return cfg
