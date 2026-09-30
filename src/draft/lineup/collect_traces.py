"""Record per-joint torque/speed and foot-contact time series at one scenario.

Keeps the full trajectory (``evaluate.py`` keeps only scalars), written as an npz
(default ``logs/lineup_traces/<robot>_<scenario>.npz``).

  python -m draft.lineup.collect_traces --robot bear --scenario sprint \\
      --checkpoint logs/<run>/<ts>/model_<it>.pt
"""
from __future__ import annotations

import os
import sys

# Headless rendering: default to EGL on Linux only (it does not exist on macOS).
if "MUJOCO_GL" not in os.environ and sys.platform.startswith("linux"):
    os.environ["MUJOCO_GL"] = "egl"

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper  # noqa: E402,F401
from draft.tasks.quadruped.base_env import make_base_env_cfg  # noqa: E402

from .scenarios import SCENARIOS

#: Parallel replicas, so per-episode gait events are not dominated by one stumble.
_N_ENVS = 32
_SECONDS = 12.0
_SETTLE_S = 2.0


def reward_terms(env) -> list[str]:
    """Reward-term names, or [] if this mjlab version does not expose them."""
    rm = getattr(env, "reward_manager", None)
    if rm is None:
        return []
    for attr in ("active_terms", "_term_names", "term_names"):
        names = getattr(rm, attr, None)
        if names:
            return [str(n) for n in names]
    return []


def sample_reward_terms(env, names: list[str], env_idx: int = 0) -> list[float]:
    """This step's value for each named term, or NaN where it cannot be read."""
    import math

    rm = getattr(env, "reward_manager", None)
    if rm is None or not names:
        return [math.nan] * len(names)
    # The per-step buffer, under the names various mjlab versions use.
    for attr in ("_step_reward", "step_reward", "_term_values"):
        buf = getattr(rm, attr, None)
        if buf is None:
            continue
        try:
            row = buf[env_idx]
            return [float(row[i]) for i in range(len(names))]
        except Exception:                       # shape or type is not what we assumed
            break
    return [math.nan] * len(names)


def collect(robot: str, scenario: str, checkpoint: str, out: Path,
            device: str = "cuda:0", *, with_rewards: bool = False) -> Path:
    from .evaluate import (_flat_terrain, _load_policy, _pin_command,
                          _set_push, _stairs_at)

    scen = SCENARIOS[scenario]
    cfg, ctx = make_base_env_cfg(
        robot_xml=f"generated/{robot}/quadruped.xml", n_envs=_N_ENVS,
        episode_length_s=_SECONDS, play=False, clamp_command_to_vmax=False,
    )
    cfg.curriculum.clear()
    if scen["terrain"] is None:
        _flat_terrain(cfg)
    else:
        _stairs_at(cfg, step_h=scen["terrain"], only=scen.get("only"))
    _pin_command(cfg, vx=scen["cmd"])
    if scen["push"] is None:
        cfg.events.pop("push_robot", None)
    else:
        _set_push(cfg, dv=scen["push"])

    env, vec_env, policy = _load_policy(
        f"generated/{robot}/quadruped.xml", cfg, checkpoint, device)
    robot_ent = env.scene["robot"]
    feet = env.scene.sensors.get("feet_ground_contact")
    dt = float(env.step_dt)
    n_steps, n_settle = int(_SECONDS / dt), int(_SETTLE_S / dt)

    tau, vel, con, rz, vx, fell = [], [], [], [], [], None
    rew_names = reward_terms(env) if with_rewards else []
    rew: list[list[float]] = []
    with torch.inference_mode():
        obs = vec_env.get_observations()
        fell = torch.zeros(_N_ENVS, dtype=torch.bool, device=env.device)
        for t in range(n_steps):
            obs, _, dones, _ = vec_env.step(policy(obs))
            fell |= dones.bool() & ~env.termination_manager.time_outs.bool()
            if t < n_settle:
                continue
            d = robot_ent.data
            tau.append(d.qfrc_actuator.detach().cpu().numpy().copy())
            vel.append(d.joint_vel.detach().cpu().numpy().copy())
            rz.append(d.root_link_pos_w[:, 2].detach().cpu().numpy().copy())
            vx.append(d.root_link_lin_vel_b[:, 0].detach().cpu().numpy().copy())
            if feet is not None and feet.data.found is not None:
                con.append((feet.data.found > 0).detach().cpu().numpy().copy())
            if rew_names:
                rew.append(sample_reward_terms(env, rew_names))

    # Per-joint envelope, read via MotorLogger so samples and limits share one source.
    from draft.tasks.motor_logger import MotorLogger
    lg = MotorLogger(vec_env)
    sat, v_nl, fl = lg._read_limits()

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        tau=np.asarray(tau, dtype=np.float32),          # [T, B, J]
        vel=np.asarray(vel, dtype=np.float32),
        contact=np.asarray(con, dtype=bool) if con else np.zeros((0, 0, 0), bool),
        root_z=np.asarray(rz, dtype=np.float32),
        vel_x=np.asarray(vx, dtype=np.float32),
        fell=fell.detach().cpu().numpy(),
        joint_names=np.asarray(list(robot_ent.joint_names)),
        foot_names=np.asarray([n.split("_")[0] for n in feet.primary_names]
                              if feet is not None else []),
        stall_torque=np.asarray(sat, dtype=np.float32),
        no_load_speed=np.asarray(v_nl, dtype=np.float32),
        effort_limit=np.asarray(fl, dtype=np.float32),
        reward_terms=np.asarray(rew_names),
        reward=np.asarray(rew, dtype=np.float32) if rew
        else np.zeros((0, 0), np.float32),        # [T, n_terms], env 0 only
        dt=dt, robot=robot, scenario=scenario, command_ms=scen["cmd"],
        leg_length_m=ctx.leg_length, checkpoint=checkpoint,
    )
    env.close()
    print(f"wrote {out}  T={len(tau)} B={_N_ENVS} J={len(robot_ent.joint_names)} "
          f"falls={int(fell.sum())}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", required=True)
    ap.add_argument("--scenario", required=True, choices=tuple(SCENARIOS))
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cmd", type=float, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--with-rewards", action="store_true",
                    help="also record each reward term per step, for "
                         "scripts/analyze.py's reward panel")
    args = ap.parse_args()
    if args.cmd is not None:
        SCENARIOS[args.scenario] = dict(SCENARIOS[args.scenario], cmd=args.cmd)
    out = Path(args.out or f"logs/lineup_traces/{args.robot}_{args.scenario}.npz")
    collect(args.robot, args.scenario, args.checkpoint, out, args.device,
            with_rewards=args.with_rewards)


if __name__ == "__main__":
    main()
