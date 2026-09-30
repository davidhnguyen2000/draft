"""Custom MDP terms for the parametric quadruped tasks.

Only the feet collide (for sim speed), so a sunk trunk never registers contact:
``base_height_exp`` rewards nominal trunk height, and the ``root_height_*``
terminations detect a collapse from state rather than from contacts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def base_height_exp(
    env: "ManagerBasedRlEnv",
    target_height: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """``exp(-((z - target) / std)^2)`` on world trunk z (ground at z=0).

    Assumes flat ground at z=0.
    """
    asset = env.scene[asset_cfg.name]
    z = asset.data.root_link_pos_w[:, 2]
    return torch.exp(-torch.square((z - target_height) / std))


def root_height_above_feet_below(
    env: "ManagerBasedRlEnv",
    minimum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when trunk height above the mean foot height is below ``minimum_height``.

    Terrain-independent, unlike :func:`root_height_below`: standing in a
    depression is not a fall. Feet come from ``asset_cfg.site_names``.
    """
    asset = env.scene[asset_cfg.name]
    root_z = asset.data.root_link_pos_w[:, 2]
    feet_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
    return (root_z - feet_z.mean(dim=-1)) < minimum_height


def root_height_below(
    env: "ManagerBasedRlEnv",
    minimum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the trunk root falls below ``minimum_height`` (world z)."""
    asset = env.scene[asset_cfg.name]
    return asset.data.root_link_pos_w[:, 2] < minimum_height
