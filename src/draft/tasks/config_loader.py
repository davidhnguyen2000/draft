"""Per-run YAML config loading.

Top-level keys: ``task`` (required), ``env_kwargs`` (forwarded to
``make_env_cfg``), ``reward_overrides`` (term name -> weight). Any other key is a
runner-cfg override, validated by the runner cfg's ``apply_overrides``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


_KNOWN_TOP_LEVEL = {"task", "env_kwargs", "reward_overrides"}


def load_run_config(path: str | Path) -> dict[str, Any]:
    """Load a per-run YAML config and split it into task, env_kwargs, reward and runner overrides."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    data = {k: v for k, v in data.items() if v is not None}

    if "task" not in data:
        raise ValueError(f"Config {path} is missing required 'task' field.")

    runner_overrides = {
        k: v for k, v in data.items() if k not in _KNOWN_TOP_LEVEL
    }
    return {
        "task": data["task"],
        "env_kwargs": data.get("env_kwargs", {}) or {},
        "reward_overrides": data.get("reward_overrides", {}) or {},
        "runner_overrides": runner_overrides,
    }


def find_latest_checkpoint(log_dir: str | Path, run_name: str) -> Path:
    """Return the highest-numbered model_*.pt from the most recent run folder."""
    run_root = Path(log_dir) / run_name
    if not run_root.is_dir():
        raise FileNotFoundError(f"No runs found at {run_root}")

    run_dirs = [d for d in run_root.iterdir() if d.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(f"No run subdirectories found in {run_root}")

    latest_run = max(run_dirs, key=lambda d: d.stat().st_mtime)

    checkpoints = list(latest_run.glob("model_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No model_*.pt checkpoints found in {latest_run}")

    return max(checkpoints, key=lambda p: int(p.stem.split("_")[1]))


def require_robot_xml(env_kwargs: dict[str, Any]) -> None:
    """Stop with the command that builds the robot, rather than a traceback.

    `generated/` is not committed, so on a fresh checkout the robot a run config
    names does not exist yet.
    """
    for key in ("robot_xml", "task_geometry_xml"):
        xml = env_kwargs.get(key)
        if xml and not Path(xml).exists():
            raise SystemExit(
                f"{key} {xml} does not exist; generated/ is not committed, so build it first:\n"
                "  python scripts/generate_quadrupeds.py            # the paper's cheetah, bear, giraffe\n"
                "  python scripts/generate_robot.py --robot quadruped --output-dir <dir> --fixed-output-dir\n"
                "or point env_kwargs.robot_xml in the run config at a design you have built.")
