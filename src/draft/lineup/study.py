"""Where the §V study's inputs and outputs live.

The study is one sweep over three fixed machines: their models, run names,
perturbed-arm spec and frozen record. The committed files are under
``experiments/``; everything the sweep writes goes under ``generated/`` and
``logs/``. A rerun on changed machines is a different study, and
``experiments/machines.json`` refuses it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..paths import repo_root

ROOT = repo_root()


@dataclass(frozen=True)
class Study:
    #: The inertia-mismatch spec the ``dr`` arm draws from, relative to the repo.
    dr_spec: str
    #: Sprint-trace command per design; empty derives it in sweep.py.
    sprint_peak: dict = field(default_factory=dict)

    @staticmethod
    def robot_xml(robot: str) -> str:
        """The machine: ``generated/<robot>/quadruped.xml``."""
        return f"generated/{robot}/quadruped.xml"

    @staticmethod
    def run(robot: str, task: str, seed: int) -> str:
        """The run name, which is also its directory under ``logs/``."""
        return f"{robot}_{task}_s{seed}"

    # ── log directories, relative to logs/ ───────────────────────────────────
    eval_dir = "lineup_eval"
    walk_dir = "lineup_walk"
    trace_dir = "lineup_traces"

    def sprint_traces(self, robot: str, seeds) -> list[str]:
        """The trace files ``freeze_results.sprint_limits`` pools, relative to logs/."""
        return [f"{self.trace_dir}/{robot}_sprintpeak_s{s}.npz" for s in seeds]

    # ── committed files ──────────────────────────────────────────────────────
    @property
    def machines(self) -> Path:
        """The fingerprint of each machine and its inputs; the sweep refuses a mismatch."""
        return ROOT / "experiments" / "machines.json"

    @property
    def record(self) -> Path:
        """Everything §V reports, reduced from the raw cells."""
        return ROOT / "experiments" / "results.json"

    @property
    def manifest(self) -> Path:
        """Size, checksum and url of each released archive."""
        return ROOT / "experiments" / "artifacts.json"


#: The study the paper reports.
STUDY = Study("experiments/twin_inertia_mismatch.json")
