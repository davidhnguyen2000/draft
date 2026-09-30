#!/usr/bin/env python3
"""Reproduce the locomotion sweep (paper §V; see ``study.py``) on one machine.

3 designs (cheetah, bear, giraffe) x 10 seeds (0-9) x 4 tasks:

  flat                        stage 1, cold start, 1200 iterations
  velocity / terrain / push   stage 2, each warm-started from THAT seed's own
                              flat policy, 1800 / 2000 / 1500 iterations

All runs use 4096 environments x 24 steps. That is 120 trainings (195,000 PPO
iterations in total), then 360 evaluation cells (3 x 10 x 6 cells x 2 arms,
nominal and inertia-perturbed), 30 sprint traces and 30 walk recordings.
The paper's training took about 12 wall-hours on 4 GPUs running at once, so
expect roughly 50 GPU-hours on one GPU.

Each step is a subprocess (``scripts/train.py``, ``draft.lineup.evaluate``,
``collect_traces``, ``record_walk``); run configs are expanded from
``experiments/lineup.yaml`` (``draft.lineup.configs``).

Re-running is safe: finished runs and existing outputs are skipped (unless
``--force``), and a partial run resumes for the iterations it still owes.

  python scripts/train_all.py all --dry-run --designs cheetah --seeds 0
  python scripts/train_all.py train --tasks flat
  python scripts/train_all.py evaluate --arms nom
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from collections import Counter
from pathlib import Path

import yaml

from ..paths import repo_root
from . import machines, study

REPO = repo_root()
PY = sys.executable
LOGS = REPO / "logs"

DESIGNS = ("cheetah", "bear", "giraffe")
SEEDS = tuple(range(10))
TASKS = ("flat", "velocity", "terrain", "push")
ARMS = {"nom": [], "dr": ["--dr-inertia", "--dr-seed", "0"]}

#: (policy task, battery, output label, extra flags). The stair splits score
#: mound and pit separately.
EVAL_CELLS = (
    ("flat", "flat", "flat_flat", []),
    ("velocity", "speed", "sprint_speed", []),
    ("terrain", "terrain", "stairs_terrain", []),
    ("push", "push", "push_push", []),
    ("terrain", "terrain", "pyramid_stairs", ["--stairs-only", "pyramid_stairs"]),
    ("terrain", "terrain", "pyramid_stairs_inv", ["--stairs-only", "pyramid_stairs_inv"]),
)


class Sweep:
    #: Where the expanded run configurations are written (build output).
    CONFIG_ROOT = REPO / "generated" / "lineup_configs"

    def __init__(self, args):
        self.a = args
        self.R = study.STUDY
        self.missing: list[str] = []      # prerequisites we could not satisfy
        self.planned: set[str] = set()    # runs a dry-run pretends to have trained
        self._config_dir: Path | None = None

    def config_dir(self) -> Path:
        """Expand the specification once per process and return the folder.

        Not cached across processes: the ceiling is measured on the current machine.
        """
        if self._config_dir is None:
            from . import configs
            out = self.CONFIG_ROOT
            runs = configs.expand(designs=self.a.designs)
            configs.materialise(runs, out)
            print(f"# expanded {len(runs)} run configs -> {out.relative_to(REPO)}"
                  f"  (digest {configs.digest(runs)[:12]})", flush=True)
            self._config_dir = out
        return self._config_dir

    # ── plumbing ─────────────────────────────────────────────────────────────
    def run(self, cmd: list[str], why: str) -> None:
        print(f"\n# {why}\n{shlex.join(['python', *cmd[1:]])}", flush=True)
        if self.a.dry_run:
            return
        if "--out" in cmd:
            (REPO / cmd[cmd.index("--out") + 1]).parent.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "MJLAB_HEADLESS": "1"}
        if sys.platform.startswith("linux"):     # EGL is headless Linux only
            env.update(MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl")
        r = subprocess.run(cmd, cwd=REPO, env=env)
        if r.returncode:
            raise SystemExit(f"step failed (exit {r.returncode}): {why}")

    def skip(self, out: Path, what: str) -> bool:
        if out.exists() and not self.a.force:
            print(f"  skip {what}: {out.relative_to(REPO)} exists")
            return True
        return False

    def lacking(self, msg: str) -> None:
        print(f"  !! {msg}")
        self.missing.append(msg)

    def config(self, robot: str, task: str, seed: int) -> Path:
        stage = "stage1" if task == "flat" else "stage2"
        return self.config_dir() / stage / f"{self.R.run(robot, task, seed)}.yaml"

    @staticmethod
    def newest(run: str) -> tuple[Path, int] | None:
        """Newest ``logs/<run>/*/model_*.pt`` by mtime, with its iteration."""
        pts = sorted(LOGS.glob(f"{run}/*/model_*.pt"), key=lambda p: p.stat().st_mtime)
        return (pts[-1], int(re.search(r"model_(\d+)", pts[-1].name)[1])) if pts else None

    def checkpoint(self, robot: str, task: str, seed: int, for_what: str) -> str | None:
        """Path of a finished policy, or None (recorded as missing)."""
        run = self.R.run(robot, task, seed)
        cfg = self.config(robot, task, seed)
        if not cfg.exists():
            self.lacking(f"{cfg.relative_to(REPO)} missing; cannot {for_what} — "
                         f"check that experiments/lineup.yaml names this run.")
            return None
        budget = yaml.safe_load(cfg.read_text())["max_iterations"]
        got = self.newest(run)
        if got and got[1] >= budget - 1:
            return str(got[0].relative_to(REPO))
        if self.a.dry_run and run in self.planned:
            return f"logs/{run}/<timestamp>_{run}/model_{budget - 1}.pt"
        state = f"only reached iteration {got[1]} of {budget}" if got else "has no checkpoint"
        self.lacking(f"{run} {state}; cannot {for_what}. Run: train_all.py train "
                     f"--designs {robot} --seeds {seed} --tasks {task}")
        return None

    def cells(self, tasks=TASKS):
        for robot in self.a.designs:
            for seed in self.a.seeds:
                for task in tasks:
                    yield robot, seed, task

    def machines_match(self) -> bool:
        """True if every machine this step loads matches its pinned fingerprint."""
        bad = machines.check(self.a.designs)
        if bad and self.a.dry_run and all(b.endswith("is missing") for b in bad):
            # Run configs are written from the machines (each one's command
            # ceiling is measured on its XML), so a dry run cannot print the
            # later steps until they exist. Building them takes seconds.
            self.lacking("the machines are not built yet, and the run configs are "
                         "written from them: run `python scripts/train_all.py designs` "
                         "(seconds), then this dry run again")
            return False
        for b in bad:
            self.lacking(b)
        if bad:
            self.lacking(f"the pinned machines are fingerprinted in "
                         f"{self.R.machines.relative_to(REPO)}; rebuild them with "
                         f"train_all.py designs --force.")
        return not bad

    def record_environment(self) -> None:
        """Once: the commit and every installed package, beside the logs."""
        out = LOGS / "lineup_environment.txt"
        if self.a.dry_run or out.exists():
            return
        def sh(*cmd):
            r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
            return (r.stdout or r.stderr).strip()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(f"python: {sys.version}\n"
                       f"git HEAD: {sh('git', 'rev-parse', 'HEAD')}\n"
                       f"git status --porcelain:\n{sh('git', 'status', '--porcelain')}\n\n"
                       f"pip freeze:\n{sh(PY, '-m', 'pip', 'freeze')}\n")
        print(f"  recorded {out.relative_to(REPO)}")

    def sprint_peak(self, robot: str) -> float | None:
        """The design's sprint command: stated, or the modal peak rung of the nominal speed ladder."""
        if self.R.sprint_peak:
            return self.R.sprint_peak[robot]
        from . import freeze_results as FR
        peaks = []
        for seed in SEEDS:
            f = LOGS / self.R.eval_dir / f"nom_{robot}_s{seed}_sprint_speed.json"
            if not f.exists():
                continue
            held = [x for x in json.loads(f.read_text())["rows"]
                    if x["survive_rate"] >= FR.SURV]
            if held:
                peaks.append(max(held, key=FR._median_speed)["cmd_ms"])
        return Counter(peaks).most_common(1)[0][0] if peaks else None

    # ── steps ────────────────────────────────────────────────────────────────
    def designs(self) -> None:
        todo = [r for r in self.a.designs
                if self.a.force or not (REPO / self.R.robot_xml(r)).exists()]
        for r in set(self.a.designs) - set(todo):
            print(f"  skip design {r}: {self.R.robot_xml(r)} exists")
        if todo:
            self.run([PY, "scripts/generate_quadrupeds.py", "--only", *todo],
                     f"build the machines generated/<robot>/ for {', '.join(todo)}")

    def train(self) -> None:
        # All stage-1 runs first, so every stage-2 run finds its flat policy.
        stage1 = [t for t in self.a.tasks if t == "flat"]
        stage2 = [t for t in self.a.tasks if t != "flat"]
        for tasks in (stage1, stage2):
            for robot, seed, task in self.cells(tasks):
                self.train_one(robot, seed, task)

    def train_one(self, robot: str, seed: int, task: str) -> None:
        cfg_path = self.config(robot, task, seed)
        if not cfg_path.exists():
            self.lacking(f"{cfg_path.relative_to(REPO)} missing — "
                         f"check that experiments/lineup.yaml names this run.")
            return
        cfg = yaml.safe_load(cfg_path.read_text())
        run, budget = cfg["run_name"], cfg["max_iterations"]
        got = self.newest(run)
        if got and got[1] >= budget - 1 and not self.a.force:
            print(f"  skip train {run}: {got[0].relative_to(REPO)} reached its budget of {budget}")
            return
        cmd = [PY, "scripts/train.py", "--config", str(cfg_path.relative_to(REPO))]
        overrides = {"device": self.a.device} if self.a.device else {}
        if got and not self.a.force:
            # rsl_rl runs max_iterations MORE on resume, so pass only what is owed.
            overrides["max_iterations"] = budget - got[1]
            cmd += ["--resume", str(got[0].relative_to(REPO))]
            why = f"resume {run} at iteration {got[1]} of {budget}"
        elif task == "flat":
            why = f"train {run} (stage 1, cold start)"
        else:
            flat = self.checkpoint(robot, "flat", seed, f"warm-start {run}")
            if flat is None:
                return
            cmd += ["--warmstart", flat]
            why = f"train {run} (stage 2, warm start from seed {seed}'s own flat policy)"
        if overrides:
            derived = LOGS / run / "sweep_config.yaml"
            if not self.a.dry_run:
                derived.parent.mkdir(parents=True, exist_ok=True)
                derived.write_text(f"# {cfg_path.relative_to(REPO)} with {overrides}, "
                                   f"written by scripts/train_all.py\n"
                                   + yaml.safe_dump({**cfg, **overrides}, sort_keys=False))
            cmd[3] = str(derived.relative_to(REPO))
            why += f"; config overrides {overrides}"
        self.run(cmd, why)
        self.planned.add(run)

    def evaluate(self) -> None:
        out_dir = LOGS / self.R.eval_dir
        for robot in self.a.designs:
            for seed in self.a.seeds:
                for arm in self.a.arms:
                    # The perturbed arm passes the study's own inertia spec.
                    arm_flags = ARMS[arm] + (["--dr-spec", self.R.dr_spec] if arm == "dr" else [])
                    for policy, battery, label, extra in EVAL_CELLS:
                        if policy not in self.a.tasks:
                            continue
                        name = (f"{arm}split_{robot}_s{seed}_{label}" if extra
                                else f"{arm}_{robot}_s{seed}_{label}")
                        out = out_dir / f"{name}.json"
                        if self.skip(out, f"eval {name}"):
                            continue
                        ckpt = self.checkpoint(robot, policy, seed, f"score {name}")
                        if ckpt is None:
                            continue
                        # terrain_scan_fix changes the observation and the study is not
                        # uniform in it, so read it from the run's config, never default.
                        cfg_p = self.config(robot, policy, seed)
                        ek = yaml.safe_load(cfg_p.read_text()).get("env_kwargs", {})
                        if "terrain_scan_fix" not in ek:
                            raise SystemExit(
                                f"{cfg_p}: env_kwargs.terrain_scan_fix is not stated. "
                                "Every config must state it outright — see "
                                "draft.tasks.quadruped.base_env.TERRAIN_SCAN_FIXES.")
                        fix = ek["terrain_scan_fix"]
                        self.run([PY, "-m", "draft.lineup.evaluate",
                                  "--robot", robot,
                                  "--battery", battery, "--checkpoint", ckpt,
                                  "--out", str(out.relative_to(REPO)), *extra, *arm_flags,
                                  "--terrain-scan-fix", fix,
                                  *self.device_flag()],
                                 f"eval {name}")

    def traces(self) -> None:
        if "velocity" not in self.a.tasks:
            return
        for robot in self.a.designs:
            peak = self.sprint_peak(robot)
            if peak is None and not self.a.dry_run:
                self.lacking(f"no nominal sprint evaluation for {robot} under "
                             f"logs/{self.R.eval_dir}/, so its sprint command is unknown. "
                             f"Run: python scripts/train_all.py evaluate --arms nom "
                             f"--tasks velocity --designs {robot}")
                continue
            cmd_ms = "<modal peak rung>" if peak is None else str(peak)
            for seed in self.a.seeds:
                out = LOGS / self.R.trace_dir / f"{robot}_sprintpeak_s{seed}.npz"
                if self.skip(out, "sprint trace"):
                    continue
                ckpt = self.checkpoint(robot, "velocity", seed, f"trace {out.name}")
                if ckpt is None:
                    continue
                self.run([PY, "-m", "draft.lineup.collect_traces",
                          "--robot", robot,
                          "--scenario", "sprint", "--checkpoint", ckpt,
                          "--cmd", cmd_ms, "--out", str(out.relative_to(REPO)),
                          *self.device_flag()],
                         f"sprint trace {robot} seed {seed} at {cmd_ms} m/s")

    def walk(self) -> None:
        if "flat" not in self.a.tasks:
            return
        for robot in self.a.designs:
            for seed in self.a.seeds:
                out = LOGS / self.R.walk_dir / f"{robot}_s{seed}.npz"
                if self.skip(out, "walk recording"):
                    continue
                ckpt = self.checkpoint(robot, "flat", seed, f"record {out.name}")
                if ckpt is None:
                    continue
                self.run([PY, "-m", "draft.lineup.record_walk", "--robot", robot,
                          "--robot-xml", self.R.robot_xml(robot),
                          "--checkpoint", ckpt, "--n-envs", "64", "--env-seed", "1234",
                          "--out", str(out.relative_to(REPO)), *self.device_flag()],
                         f"walk recording {robot} seed {seed}")

    def device_flag(self) -> list[str]:
        return ["--device", self.a.device] if self.a.device else []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=("designs", "train", "evaluate", "traces", "walk", "all"))
    ap.add_argument("--designs", nargs="+", choices=DESIGNS, default=list(DESIGNS))
    ap.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    ap.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS),
                    help="train these tasks; evaluate/traces/walk only the cells that read them")
    ap.add_argument("--arms", nargs="+", choices=tuple(ARMS), default=list(ARMS),
                    help="evaluation arms: nom (nominal) and dr (inertia-perturbed)")
    ap.add_argument("--device", default=None,
                    help="e.g. cuda:0. Default: each script's own (config `device: cuda`, eval cuda:0)")
    ap.add_argument("--dry-run", action="store_true", help="print the commands, run nothing")
    ap.add_argument("--force", action="store_true",
                    help="redo finished trainings from scratch and overwrite existing outputs")
    args = ap.parse_args()

    sw = Sweep(args)
    steps = ("designs", "train", "evaluate", "traces", "walk") if args.step == "all" else (args.step,)
    for step in steps:
        print(f"\n=== {step} ===")
        if step != "designs" and not sw.machines_match():
            break
        if step == "train":
            sw.record_environment()
        getattr(sw, step)()
    if sw.missing:
        print(f"\n{len(sw.missing)} step(s) could not run because a prerequisite is missing:")
        for m in sw.missing:
            print(f"  {m}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
