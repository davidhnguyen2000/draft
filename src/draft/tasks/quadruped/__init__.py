"""The parametric quadruped task family.

Re-exports are lazy so mjlab-free modules (e.g. ``speed_limits``) import
without mjlab installed.
"""

import importlib

_LAZY = {
    "make_env_cfg": ".env",
    "make_quadruped_entity_cfg": ".quadruped_entity",
    "QuadrupedLocomotionRunnerCfg": ".runner_cfg",
    "make_runner_cfg": ".runner_cfg",
}

__all__ = list(_LAZY)


def __getattr__(name: str):
    if name in _LAZY:
        return getattr(importlib.import_module(_LAZY[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + __all__)
