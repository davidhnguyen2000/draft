"""Flat-ground quadruped task: the base policy the challenge tasks warm-start from.

``make_base_env_cfg`` with no curriculum. Point a challenge task's
``warmstart_checkpoint`` at this policy's checkpoint.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg

from draft.tasks.quadruped.base_env import make_base_env_cfg


def make_env_cfg(**base) -> ManagerBasedRlEnvCfg:
    """The base environment, with no curriculum on top.

    All kwargs pass straight to ``make_base_env_cfg``, which rejects unknown keys.
    """
    cfg, _ctx = make_base_env_cfg(**base)
    return cfg
