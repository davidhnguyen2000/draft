"""The §V sweep's 120 run configurations, expanded from ``experiments/lineup.yaml``.

Each run follows from (design, task, seed) plus the design's measured command
ceiling (``estimate_top_speed``). ``tests/test_lineup_configs.py`` pins the
expansion's digest.

    from draft.lineup import configs
    runs = configs.expand()                  # 120 RunConfig
    configs.materialise(runs, out_dir)       # ...as YAML, for scripts/train.py
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from ..paths import repo_root

SPEC = repo_root() / "experiments" / "lineup.yaml"


@dataclass(frozen=True)
class RunConfig:
    """One training run: exactly what `scripts/train.py --config` consumes."""

    run_name: str
    task: str
    seed: int
    stage: int
    max_iterations: int
    #: The flat policy this run warm-starts from, or None for a stage-1 run.
    warmstart_from: Optional[str]
    env_kwargs: dict = field(default_factory=dict)
    common: dict = field(default_factory=dict)

    def to_yaml_dict(self) -> dict:
        """The mapping a run's YAML file holds, in the order it is written."""
        out: dict[str, Any] = {
            "task": self.task,
            "run_name": self.run_name,
            "seed": self.seed,
            "device": self.common["device"],
            "logger": self.common["logger"],
            "wandb_project": self.common["wandb_project"],
            "n_envs": self.common["n_envs"],
            "max_iterations": self.max_iterations,
            "save_interval": self.common["save_interval"],
            "env_kwargs": dict(self.env_kwargs),
        }
        return out


def load_spec(path: Path | None = None) -> dict:
    return yaml.safe_load((path or SPEC).read_text())


def _v_max(design: str) -> float:
    """The design's estimated top speed, measured on its own machine (lazy MuJoCo import)."""
    from ..tasks.quadruped.speed_limits import estimate_top_speed

    xml = repo_root() / "generated" / design / "quadruped.xml"
    if not xml.exists():
        raise SystemExit(
            f"missing {xml.relative_to(repo_root())} — run "
            "`python scripts/generate_quadrupeds.py` first. "
            "The command ceiling is measured on the machine, so the machine has "
            "to exist before a config naming it can be written.")
    return estimate_top_speed(xml)


def expand(spec: dict | None = None, *, v_max: dict[str, float] | None = None,
           designs: list[str] | None = None) -> list[RunConfig]:
    """Every run of the study, ordered (design, task, seed).

    `v_max` overrides the measurement, so tests need no generated model.
    `designs` restricts the expansion, so only those machines need to exist.
    """
    spec = spec or load_spec()
    common = spec["common"]
    runs: list[RunConfig] = []

    for design in spec["designs"]:
        if designs is not None and design not in designs:
            continue
        top = v_max[design] if v_max else _v_max(design)
        xml = f"generated/{design}/quadruped.xml"

        for name, t in spec["tasks"].items():
            for seed in spec["seeds"]:
                env: dict[str, Any] = {
                    "robot_xml": xml,
                    "terrain_scale_with_size": common["terrain_scale_with_size"],
                    "dimensional_weights": common["dimensional_weights"],
                }
                if t.get("clamp"):
                    # Round only after scaling; rounding `top` first changes values.
                    env["v_max_override"] = round(top * t.get("v_max_scale", 1.0), 2)
                    env["clamp_command_to_vmax"] = True
                env["terrain_scan_fix"] = t["scan_fix"]

                runs.append(RunConfig(
                    run_name=f"{design}_{name}_s{seed}",
                    task=t["task"],
                    seed=seed,
                    stage=t["stage"],
                    max_iterations=t["iterations"],
                    # Warm-start from this seed's own flat policy, keeping seeds independent.
                    warmstart_from=(None if t["stage"] == 1
                                    else f"{design}_flat_s{seed}"),
                    env_kwargs=env,
                    common=common,
                ))
    return runs


def materialise(runs: list[RunConfig], out_dir: Path) -> list[Path]:
    """Write each run's YAML for `scripts/train.py --config` (build output, not committed)."""
    out_dir = Path(out_dir)
    written = []
    for r in runs:
        p = out_dir / f"stage{r.stage}" / f"{r.run_name}.yaml"
        p.parent.mkdir(parents=True, exist_ok=True)
        header = (f"# {r.run_name} — expanded from experiments/lineup.yaml.\n"
                  f"# Generated file: edit the specification, not this.\n\n")
        p.write_text(header + yaml.safe_dump(r.to_yaml_dict(), sort_keys=False))
        written.append(p)
    return written


def digest(runs: list[RunConfig]) -> str:
    """A stable hash of what every run receives, independent of spec formatting."""
    payload = [
        {"run_name": r.run_name, "task": r.task, "seed": r.seed,
         "max_iterations": r.max_iterations, "warmstart_from": r.warmstart_from,
         "env_kwargs": {k: r.env_kwargs[k] for k in sorted(r.env_kwargs)}}
        for r in sorted(runs, key=lambda r: r.run_name)
    ]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def by_name(runs: list[RunConfig]) -> dict[str, RunConfig]:
    return {r.run_name: r for r in runs}
