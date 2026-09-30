"""Task registry, read from ``tasks.yaml``.

Each task package exposes ``make_env_cfg(**kwargs)`` and ``make_runner_cfg()``;
the runner cfg implements ``apply_overrides(dict)`` and ``build_rsl_rl_cfg()``.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


_TASKS_YAML = Path(__file__).resolve().parent / "tasks.yaml"


@dataclass
class TaskEntry:
    name: str
    path: Path
    module_name: str
    _module: Any = None

    def load(self):
        if self._module is None:
            self._module = importlib.import_module(self.module_name)
        return self._module

    def make_env_cfg(self, **kwargs):
        return self.load().make_env_cfg(**kwargs)

    def make_runner_cfg(self):
        return self.load().make_runner_cfg()

    def env_class(self):
        """The task's ``ENV_CLASS`` if it defines one, else ``ManagerBasedRlEnv``."""
        from mjlab.envs import ManagerBasedRlEnv
        return getattr(self.load(), "ENV_CLASS", ManagerBasedRlEnv)

    def default_config_yaml(self) -> Path:
        return self.path / "config.yaml"


def _load_registry() -> dict[str, TaskEntry]:
    """Read tasks.yaml into TaskEntry objects (``package:`` is relative to this package)."""
    if not _TASKS_YAML.exists():
        raise FileNotFoundError(f"Task registry not found: {_TASKS_YAML}")
    with _TASKS_YAML.open() as f:
        data = yaml.safe_load(f) or {}
    raw_tasks = data.get("tasks", {})
    if not raw_tasks:
        raise ValueError(f"No tasks defined in {_TASKS_YAML}")

    here = Path(__file__).resolve().parent
    registry: dict[str, TaskEntry] = {}
    for name, spec in raw_tasks.items():
        rel = spec["package"]
        path = (here / rel).resolve()
        if not path.is_dir():
            raise FileNotFoundError(
                f"Task {name!r} names package {rel!r}, which is not a directory "
                f"under {here}")
        registry[name] = TaskEntry(
            name=name,
            path=path,
            module_name="draft.tasks." + rel.replace("/", "."),
        )
    return registry


TASKS: dict[str, TaskEntry] = _load_registry()


def get_task(name: str) -> TaskEntry:
    if name not in TASKS:
        available = ", ".join(sorted(TASKS)) or "<none>"
        raise KeyError(f"Unknown task '{name}'. Available: {available}")
    return TASKS[name]


def list_tasks() -> list[str]:
    return sorted(TASKS)
