"""Curriculum and metric terms for the parametric quadruped tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.curriculum_manager import CurriculumTermCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


class PushDifficultyCurriculum:
    """Adaptive push: scale ``base_range`` up or down to hold an EMA fall rate.

    Each design settles at its own ceiling, so this is not used for comparing
    designs (the push task uses :class:`PushScheduleCurriculum`). Params:
    ``event_name``, ``base_range``, ``target_fall_rate`` (0.15), ``scale_step``
    (0.01), ``min_scale`` (0.5), ``max_scale`` (3.0), ``ema_alpha`` (0.05).
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
        p = cfg.params
        self._event_name: str = p["event_name"]
        self._base_range: dict[str, tuple[float, float]] = p["base_range"]
        self._target_fall_rate: float = p.get("target_fall_rate", 0.15)
        self._scale_step: float = p.get("scale_step", 0.01)
        self._min_scale: float = p.get("min_scale", 0.5)
        self._max_scale: float = p.get("max_scale", 3.0)
        self._ema_alpha: float = p.get("ema_alpha", 0.05)

        self._scale: float = 1.0
        # Initialise EMA at target so the first few episodes don't over-correct.
        self._fall_rate_ema: float = self._target_fall_rate

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor,
        **_,
    ) -> dict[str, torch.Tensor]:
        # Fall = any non-timeout termination.
        time_outs = env.termination_manager.time_outs[env_ids]
        batch_fall_rate = (~time_outs).float().mean().item()

        # Smooth with EMA.
        self._fall_rate_ema = (
            self._ema_alpha * batch_fall_rate
            + (1.0 - self._ema_alpha) * self._fall_rate_ema
        )

        # Adjust multiplier toward keeping fall rate near target.
        if self._fall_rate_ema < self._target_fall_rate:
            self._scale = min(self._scale + self._scale_step, self._max_scale)
        else:
            self._scale = max(self._scale - self._scale_step, self._min_scale)

        self._apply(env)

        return {
            "scale": torch.tensor(self._scale),
            "fall_rate": torch.tensor(self._fall_rate_ema),
        }

    def reset(self, env_ids: torch.Tensor) -> None:
        pass  # Multiplier is global across all envs, not per-episode state.

    def _apply(self, env: ManagerBasedRlEnv) -> None:
        s = self._scale
        new_range = {k: (lo * s, hi * s) for k, (lo, hi) in self._base_range.items()}
        env.event_manager.get_term_cfg(self._event_name).params["velocity_range"] = new_range


class PushScheduleCurriculum:
    """Ramp push magnitude on a fixed schedule, identical for every design.

    Every design is shoved equally hard at every point in training. The fall rate
    is logged but does not steer anything. Params: ``event_name``, ``base_range``
    (absolute, scale=1), ``stages`` (``[{"step", "scale"}, ...]``; the last stage
    whose env ``step`` has been reached wins), ``ema_alpha`` (0.05).
    """

    def __init__(self, cfg: CurriculumTermCfg, env: "ManagerBasedRlEnv"):
        p = cfg.params
        self._event_name: str = p["event_name"]
        self._base_range: dict[str, tuple[float, float]] = p["base_range"]
        self._stages: list[dict] = sorted(p["stages"], key=lambda st: st["step"])
        self._ema_alpha: float = p.get("ema_alpha", 0.05)
        if not self._stages or self._stages[0]["step"] != 0:
            raise ValueError("PushScheduleCurriculum needs a stage at step 0 — "
                             "without one the first iterations run at whatever "
                             "the event cfg happened to be built with.")
        self._scale: float = float(self._stages[0]["scale"])
        self._fall_rate_ema: float = 0.0

    def __call__(
        self,
        env: "ManagerBasedRlEnv",
        env_ids: torch.Tensor,
        **_,
    ) -> dict[str, torch.Tensor]:
        # Driven by training progress only. scripts/train.py zeroes the counter on
        # a warm start and keeps it on --resume.
        for stage in self._stages:
            if env.common_step_counter >= stage["step"]:
                self._scale = float(stage["scale"])

        # Reported, not acted on.
        time_outs = env.termination_manager.time_outs[env_ids]
        batch_fall_rate = (~time_outs).float().mean().item()
        self._fall_rate_ema = (
            self._ema_alpha * batch_fall_rate
            + (1.0 - self._ema_alpha) * self._fall_rate_ema
        )

        self._apply(env)

        return {
            "scale": torch.tensor(self._scale),
            "fall_rate": torch.tensor(self._fall_rate_ema),
        }

    def reset(self, env_ids: torch.Tensor) -> None:
        pass  # The schedule is global across all envs, not per-episode state.

    def _apply(self, env: "ManagerBasedRlEnv") -> None:
        s = self._scale
        new_range = {k: (lo * s, hi * s) for k, (lo, hi) in self._base_range.items()}
        env.event_manager.get_term_cfg(self._event_name).params["velocity_range"] = new_range


class EpisodeSpeedMetric:
    """Log horizontal body speed of envs that just finished an episode.

    Emits ``mean_speed``, ``max_speed`` and ``cruising_speed`` (mean over
    timed-out episodes only, i.e. falls excluded).
    """

    def __init__(self, cfg: CurriculumTermCfg, env: "ManagerBasedRlEnv") -> None:
        pass  # Stateless — no per-env bookkeeping needed.

    def __call__(
        self,
        env: "ManagerBasedRlEnv",
        env_ids: torch.Tensor,
        **_,
    ) -> dict[str, torch.Tensor]:
        from mjlab.entity import Entity
        asset: Entity = env.scene["robot"]
        # Horizontal speed in body frame (x = forward, y = lateral).
        vel_xy = asset.data.root_link_lin_vel_b[env_ids, :2]
        speed = torch.norm(vel_xy, dim=1)

        # Separate falls (non-timeout terminations) from successful timeouts.
        time_outs = env.termination_manager.time_outs[env_ids]
        cruising = speed[time_outs].mean() if time_outs.any() else speed.mean()

        return {
            "mean_speed":     speed.mean(),
            "max_speed":      speed.max(),
            "cruising_speed": cruising,
        }

    def reset(self, env_ids: torch.Tensor) -> None:
        pass
