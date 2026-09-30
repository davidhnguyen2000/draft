"""Rough-terrain challenge task for the parametric quadruped.

The base env with a harder, stair-biased terrain generator and a terrain-level
curriculum starting at level 0; no extra pushes, fixed command range. Height
ceilings are absolute (unless ``terrain_scale_with_size``), so the level
reached is comparable across designs.
"""

from __future__ import annotations

from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.velocity import mdp
from mjlab.terrains.config import ROUGH_TERRAINS_CFG

from draft.tasks.quadruped.base_env import make_base_env_cfg
from draft.tasks.quadruped.runner_cfg import make_runner_cfg  # noqa: F401  (task API)

# Terrain-height ceilings at max curriculum difficulty, in metres; multiplied by
# the design's s only when ``terrain_scale_with_size`` is set (as in the base).
_MAX_STEP_HEIGHT = 0.50
_MAX_NOISE = 0.30
_MAX_WAVE_AMP = 0.50


def _apply_ceiling(st, hfield_resolution: float, s: float = 1.0):
    """Cap the height-type parameters of a sub-terrain and coarsen its hfield."""
    overrides: dict = {}
    if hasattr(st, "step_height_range"):
        overrides["step_height_range"] = (st.step_height_range[0], _MAX_STEP_HEIGHT * s)
    if hasattr(st, "noise_range"):
        overrides["noise_range"] = (st.noise_range[0], _MAX_NOISE * s)
    if hasattr(st, "amplitude_range"):
        overrides["amplitude_range"] = (st.amplitude_range[0], _MAX_WAVE_AMP * s)
    if hasattr(st, "horizontal_scale"):
        overrides["horizontal_scale"] = hfield_resolution
    return replace(st, **overrides) if overrides else st


def make_env_cfg(*, play: bool = False, terrain_scale_with_size: bool = False,
                 **base) -> ManagerBasedRlEnvCfg:
    """The step-height axis: one ramp on terrain level.

    ``terrain_mix`` has no effect: this task replaces the generator with its own
    stair-biased mix.

    Other kwargs pass straight to ``make_base_env_cfg``, which rejects unknown keys.
    """
    cfg, ctx = make_base_env_cfg(
        play=play, terrain_scale_with_size=terrain_scale_with_size, **base)
    # This task's own ceilings and step floor honour the flag too.
    ts = ctx.s if terrain_scale_with_size else 1.0

    # hfield cell ≥ 2×foot radius so a foot sphere never spans >1 cell (avoids
    # MuJoCo's per-pair contact overflow for large feet).
    hfield_scale = max(0.1, 2.0 * ctx.foot_radius)

    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "generator"
    gen = replace(ROUGH_TERRAINS_CFG)
    gen.curriculum = True
    gen.sub_terrains = {
        name: _apply_ceiling(st, hfield_scale, ts)
        for name, st in gen.sub_terrains.items()
    }
    # Bias toward stairs and raise the step-height floor, so even low levels
    # demand lifting the body over discrete obstacles.
    _HEIGHT_WEIGHTS = {  # sub-terrain -> proportion (renormalised by the generator)
        "pyramid_stairs": 0.30, "pyramid_stairs_inv": 0.30,
        "wave_terrain": 0.15, "random_rough": 0.10,
        "hf_pyramid_slope": 0.05, "hf_pyramid_slope_inv": 0.05, "flat": 0.05,
    }
    _STEP_FLOOR = 0.10 * ts  # smallest step (absolute unless the flag is set)
    reweighted: dict = {}
    for name, st in gen.sub_terrains.items():
        ov: dict = {}
        if name in _HEIGHT_WEIGHTS and hasattr(st, "proportion"):
            ov["proportion"] = _HEIGHT_WEIGHTS[name]
        if "stairs" in name and hasattr(st, "step_height_range"):
            ov["step_height_range"] = (_STEP_FLOOR, st.step_height_range[1])
        reweighted[name] = replace(st, **ov) if ov else st
    gen.sub_terrains = reweighted
    cfg.scene.terrain.terrain_generator = gen
    cfg.scene.terrain.max_init_terrain_level = 0
    cfg.sim.nconmax = None  # auto-allocate contacts for rough terrain

    # Walking off the terrain patch ends the episode; terrain-level curriculum.
    cfg.terminations["out_of_terrain_bounds"] = TerminationTermCfg(
        func=mdp.out_of_terrain_bounds
    )

    if not play:
        # Terrain-difficulty curriculum (reuse the stock terrain_levels term).
        cfg.curriculum["terrain_levels"] = ctx.base_curriculum["terrain_levels"]
    else:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.terminations.pop("out_of_terrain_bounds", None)

    return cfg
