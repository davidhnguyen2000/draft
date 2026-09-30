"""Push-robustness challenge task for the parametric quadruped.

The flat base plus mid-episode velocity pushes on a fixed ramp. To make designs
comparable, the kick (``_PUSH_BASE_RANGE``), the ramp (``_PUSH_STAGES``) and the
command range (``_PUSH_COMMAND_RANGE``) are absolute and shared by every design.
The push interval, episode length and gait reward widths still scale with
sqrt(leg length), as in the base.
"""

from __future__ import annotations


from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg

from draft.tasks.quadruped.base_env import make_base_env_cfg
from draft.tasks.quadruped.curriculums import EpisodeSpeedMetric, PushScheduleCurriculum
from draft.tasks.quadruped.runner_cfg import make_runner_cfg  # noqa: F401  (task API)


#: The scale=1 kick, in absolute m/s and rad/s: ``BaseCtx.push_base_range`` at
#: ``s_sqrt == 1``.
_PUSH_BASE_RANGE: dict[str, tuple[float, float]] = {
    "x":     (-0.5, 0.5),
    "y":     (-0.5, 0.5),
    "z":     (-0.4, 0.4),
    "roll":  (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw":   (-0.78, 0.78),
}

#: Fixed ramp, as (iteration, multiplier); ends at 5x (+-2.5 m/s in x and y).
_PUSH_STAGES = [(0, 1.0), (400, 2.0), (800, 3.5), (1200, 5.0)]

#: Command range during push training, in absolute m/s and rad/s: base_env's
#: ranges at ``s_sqrt == 1`` with no v_max clamp.
_PUSH_COMMAND_RANGE = {
    "lin_vel_x": (-1.0, 1.0),
    "lin_vel_y": (-0.6, 0.6),
    "ang_vel_z": (-0.5, 0.5),
}

#: Tracking tolerance, m/s: half the command range, as in base_env.
_PUSH_TRACK_STD = 0.5


def make_env_cfg(*, play: bool = False, **base) -> ManagerBasedRlEnvCfg:
    """The push task: one ramp on kick magnitude, everything else the base's.

    Other kwargs pass straight to ``make_base_env_cfg``, which rejects unknown keys.
    """
    cfg, ctx = make_base_env_cfg(play=play, **base)
    if play:
        return cfg

    # Only the push ramps (the base has no curriculum; the pop states that).
    cfg.curriculum.pop("terrain_levels", None)

    # One absolute command range for every design, replacing the base's.
    twist = cfg.commands["twist"]
    twist.ranges.lin_vel_x = _PUSH_COMMAND_RANGE["lin_vel_x"]
    twist.ranges.lin_vel_y = _PUSH_COMMAND_RANGE["lin_vel_y"]
    twist.ranges.ang_vel_z = _PUSH_COMMAND_RANGE["ang_vel_z"]
    # Tolerance follows the common command range.
    cfg.rewards["track_linear_velocity"].params["std"] = _PUSH_TRACK_STD

    # Push at the shared absolute baseline; the schedule below ramps it.
    push_event = ctx.push_event
    push_event.params["velocity_range"] = dict(_PUSH_BASE_RANGE)
    # More frequent than the base; scaled by sqrt(s) like gait-cycle time.
    push_event.interval_range_s = (1.0 * ctx.s_sqrt, 3.0 * ctx.s_sqrt)
    cfg.events["push_robot"] = push_event

    # Curriculum: one fixed absolute ramp, the same for every design.
    cfg.curriculum["push_schedule"] = CurriculumTermCfg(
        func=PushScheduleCurriculum,
        params={
            "event_name": "push_robot",
            "base_range": dict(_PUSH_BASE_RANGE),
            # common_step_counter advances by num_steps_per_env (24) per iteration.
            "stages": [{"step": it * 24, "scale": sc} for it, sc in _PUSH_STAGES],
            "ema_alpha": 0.05,
        },
    )
    cfg.curriculum["episode_speed"] = CurriculumTermCfg(func=EpisodeSpeedMetric, params={})
    return cfg
