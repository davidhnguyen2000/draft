"""Record a flat-walk rollout of one policy to an npz, for the walk-limit statistics.

The condition is identical for every checkpoint: fixed env seed (``--env-seed``),
spawn heading and offset pinned to 0, ``--n-envs`` replicas (default 64).
``train_all.py walk`` passes the study's machine via ``--robot-xml``.

  python -m draft.lineup.record_walk --robot giraffe \\
      --robot-xml generated/giraffe/quadruped.xml \\
      --checkpoint logs/giraffe_flat_s4/<ts>/model_<it>.pt --out rec/giraffe_s4.npz
"""

from __future__ import annotations

import os
import sys

# Headless rendering: default to EGL on Linux only (it does not exist on macOS).
if "MUJOCO_GL" not in os.environ and sys.platform.startswith("linux"):
    os.environ["MUJOCO_GL"] = "egl"

import argparse
import json
import sys
from pathlib import Path


import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
from mjlab.utils.torch import configure_torch_backends

from draft.tasks import get_task
from draft.tasks.quadruped.base_env import make_base_env_cfg

_WALK_MS = 0.40      # the flat battery's common operating point
_SECONDS = 12.0


def _build_cfg(robot_xml: str, n_envs: int, env_seed: int, walk_ms: float = _WALK_MS):
    from .evaluate import _flat_terrain, _pin_command

    # Episode well past the clip, so no timeout reset teleports a replica mid-way.
    cfg, _ = make_base_env_cfg(robot_xml=robot_xml, n_envs=n_envs,
                               episode_length_s=4 * _SECONDS, play=False,
                               clamp_command_to_vmax=False)
    cfg.curriculum.clear()
    _flat_terrain(cfg)
    _pin_command(cfg, vx=walk_ms)
    cfg.events.pop("push_robot", None)
    pr = cfg.events["reset_base"].params["pose_range"]
    pr["x"], pr["y"], pr["yaw"] = (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)
    cfg.seed = env_seed
    return cfg


def _geom_param(env, robot, attr: str, names_attr: str, id_attr: str):
    """Per-replica copy of a model field for the robot's geoms/bodies, or None."""
    try:
        arr = getattr(env.sim.model, attr)
        ids = getattr(robot.indexing, id_attr)
        val = arr[:, ids] if arr.dim() >= 3 or arr.shape[0] == env.num_envs else arr[ids]
        return val.detach().cpu().numpy(), list(getattr(robot, names_attr))
    except Exception as exc:  # noqa: BLE001 -- provenance, not a metric
        print(f"  [warn] could not read {attr}: {exc}", flush=True)
        return None, []


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", required=True, choices=("cheetah", "bear", "giraffe"))
    ap.add_argument("--robot-xml", default=None,
                    help="default generated/<robot>/quadruped.xml, the paper's machine")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-envs", type=int, default=64)
    ap.add_argument("--env-seed", type=int, default=1234)
    ap.add_argument("--walk-ms", type=float, default=_WALK_MS,
                    help="commanded forward speed (m/s); the default is the flat "
                         "battery's common operating point, which is what every "
                         "recorded clip in the paper used")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    from mjlab.rl.runner import MjlabOnPolicyRunner

    configure_torch_backends()
    device = args.device if torch.cuda.is_available() else "cpu"
    robot_xml = args.robot_xml or f"generated/{args.robot}/quadruped.xml"
    env = ManagerBasedRlEnv(_build_cfg(robot_xml, args.n_envs, args.env_seed, args.walk_ms),
                            device=device)
    vec_env = RslRlVecEnvWrapper(env)

    runner_cfg = get_task("quadruped").make_runner_cfg()
    runner_cfg.device = device
    train_cfg = runner_cfg.build_rsl_rl_cfg()
    train_cfg["experiment_name"] = "lineup_record"
    runner = MjlabOnPolicyRunner(vec_env, train_cfg, device=device)
    runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)

    robot = env.scene["robot"]
    feet = env.scene.sensors["feet_ground_contact"]
    origins = env.scene.env_origins.clone()
    dt = float(env.step_dt)
    n_steps = int(_SECONDS / dt)
    term_names = list(getattr(env.termination_manager, "active_terms", []))

    keys = ("root_pos", "root_quat", "lin_vel_b", "ang_vel_b", "proj_grav",
            "joint_pos", "joint_vel", "torque", "action", "contact", "done", "timeout")
    rec: dict[str, list] = {k: [] for k in keys}
    terms: dict[str, list] = {n: [] for n in term_names}

    def cpu(x):
        return x.detach().float().cpu().numpy()

    with torch.inference_mode():
        obs = vec_env.get_observations()
        for _ in range(n_steps):
            act = policy(obs)
            obs, _, dones, _ = vec_env.step(act)
            d = robot.data
            rec["root_pos"].append(cpu(d.root_link_pos_w - origins))
            rec["root_quat"].append(cpu(d.root_link_quat_w))
            rec["lin_vel_b"].append(cpu(d.root_link_lin_vel_b))
            rec["ang_vel_b"].append(cpu(d.root_link_ang_vel_b))
            rec["proj_grav"].append(cpu(d.projected_gravity_b))
            rec["joint_pos"].append(cpu(d.joint_pos))
            rec["joint_vel"].append(cpu(d.joint_vel))
            rec["torque"].append(cpu(d.qfrc_actuator))
            rec["action"].append(cpu(act))
            rec["contact"].append(cpu(feet.data.found) > 0)
            rec["done"].append(cpu(dones) > 0)
            rec["timeout"].append(cpu(env.termination_manager.time_outs) > 0)
            for n in term_names:
                try:
                    terms[n].append(cpu(env.termination_manager.get_term(n)) > 0)
                except Exception:  # noqa: BLE001
                    pass

    out = {k: np.stack(v) for k, v in rec.items()}          # [T, B, ...]
    out.update({f"term_{n}": np.stack(v) for n, v in terms.items() if v})
    fric, geom_names = _geom_param(env, robot, "geom_friction", "geom_names", "geom_ids")
    if fric is not None:
        out["geom_friction"] = fric
    com, body_names = _geom_param(env, robot, "body_ipos", "body_names", "body_ids")
    if com is not None:
        out["body_ipos"] = com
    meta = {
        "robot": args.robot, "robot_xml": robot_xml, "checkpoint": args.checkpoint,
        "dt": dt, "walk_ms": args.walk_ms, "env_seed": args.env_seed, "n_envs": args.n_envs,
        "joint_names": list(robot.joint_names),
        "feet": [n.split("_")[0] for n in feet.primary_names],
        "geom_names": geom_names, "body_names": body_names, "terminations": term_names,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, meta=json.dumps(meta), **out)
    fell = (out["done"] & ~out["timeout"]).any(0)
    print(f"[record] {args.robot} {Path(args.checkpoint).parent.parent.name}: "
          f"{n_steps} steps x {args.n_envs} envs, fell {fell.mean():.1%} -> {args.out}")
    env.close()


if __name__ == "__main__":
    main()
