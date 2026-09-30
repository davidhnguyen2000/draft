"""Absolute evaluation battery for the cheetah / bear / giraffe lineup.

Training is scaled to each robot's leg length; comparison is not. Every battery
here uses the same absolute units (m/s, m of step, m/s of kick) for all designs,
re-pinning the task around the trained policy.

  flat     gait quality at one common walk speed
  speed    forward-speed ladder -> top sustained speed
  terrain  step-height ladder -> tallest step cleared
  push     velocity-kick ladder -> recovery rate

Writes per-level JSON rows to ``--out``; ``scripts/train_all.py freeze`` reduces them.

  python -m draft.lineup.evaluate --robot giraffe --battery speed \\
      --checkpoint logs/<run>/<ts>/model_<it>.pt
"""

from __future__ import annotations

import os
import sys

# Must precede any mujoco import: default to EGL on Linux only (absent on macOS).
if "MUJOCO_GL" not in os.environ and sys.platform.startswith("linux"):
    os.environ["MUJOCO_GL"] = "egl"

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path


import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
from mjlab.terrains.config import ROUGH_TERRAINS_CFG
from mjlab.utils.torch import configure_torch_backends

from draft.tasks import get_task
from draft.tasks.quadruped.base_env import make_base_env_cfg
from draft.tasks.gait_metrics import (DEFAULT_FEET, contact_gait_metrics,
                                      joint_symmetry_metrics)

# ── The battery, in absolute units. Identical for every robot. ───────────────
# Each ladder runs past every design's failure point.
SPEED_GRID_MS = (0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0,
                 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0,
                 10.0, 12.0, 14.0, 17.0, 20.0)
# 0.05-0.50 m every 2.5 cm.
STEP_HEIGHT_GRID_M = tuple(round(0.05 + 0.025 * i, 3) for i in range(19))
# 1-12 m/s every 0.5.
PUSH_DV_GRID_MS = tuple(round(1.0 + 0.5 * i, 2) for i in range(23))

# A speed every design can hold; the flat battery's common operating point.
COMMON_WALK_MS = 0.40

_N_ENVS = 256          # parallel replicas per level -> the rate metrics' sample
_EPISODE_S = 12.0      # long enough for a push to be recovered from or not
_SETTLE_S = 2.0        # discarded head of each episode (drop-in transient)
#: A surviving replica whose net displacement is below this fraction of what the
#: command would cover counts as stuck rather than progressing.
_STUCK_FRAC = 0.25


# ─────────────────────────────────────────────────────────────────────────────
def _pin_command(env_cfg, vx: float, vy: float = 0.0, wz: float = 0.0) -> None:
    """Force a constant absolute twist command (each range collapsed to a point)."""
    twist = env_cfg.commands["twist"]
    twist.ranges.lin_vel_x = (vx, vx)
    twist.ranges.lin_vel_y = (vy, vy)
    twist.ranges.ang_vel_z = (wz, wz)
    # Never resample mid-episode.
    if hasattr(twist, "resampling_time_range"):
        twist.resampling_time_range = (1e9, 1e9)


def _flat_terrain(env_cfg) -> None:
    """Replace the generator with flat ground (speed / push batteries)."""
    assert env_cfg.scene.terrain is not None
    env_cfg.scene.terrain.terrain_type = "plane"
    env_cfg.scene.terrain.terrain_generator = None
    env_cfg.terminations.pop("out_of_terrain_bounds", None)


def _stairs_at(env_cfg, step_h: float, only: str | None = None) -> None:
    """Pin the terrain to one absolute step height, no curriculum.

    Stair sub-terrains only; ``only`` keeps a single stair type. Tread depth stays
    at mjlab's default (0.3 m) for every design, so relative to leg length the
    same staircase differs between designs.
    """
    assert env_cfg.scene.terrain is not None
    gen = replace(ROUGH_TERRAINS_CFG)
    gen.curriculum = False
    subs = {}
    for name, st in gen.sub_terrains.items():
        if "stairs" not in name:
            continue
        if only is not None and name != only:
            continue
        ov = {"proportion": 1.0}
        if hasattr(st, "step_height_range"):
            ov["step_height_range"] = (step_h, step_h)
        subs[name] = replace(st, **ov)
    if not subs:
        raise RuntimeError("no stair sub-terrains in ROUGH_TERRAINS_CFG")
    gen.sub_terrains = subs
    env_cfg.scene.terrain.terrain_type = "generator"
    env_cfg.scene.terrain.terrain_generator = gen
    env_cfg.scene.terrain.max_init_terrain_level = 0
    env_cfg.sim.nconmax = None


def _set_push(env_cfg, dv: float) -> None:
    """One absolute linear velocity kick of magnitude dv, on a fixed cadence."""
    ev = env_cfg.events.get("push_robot")
    if ev is None:
        raise RuntimeError("base env has no push_robot event")
    ev.params["velocity_range"] = {
        "x": (-dv, dv), "y": (-dv, dv), "z": (0.0, 0.0),
        "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
    }
    ev.interval_range_s = (3.0, 3.0)


# ─────────────────────────────────────────────────────────────────────────────
def _load_policy(robot_xml: str, env_cfg, checkpoint: str, device: str):
    """Build the env and load the trained actor onto it."""
    from mjlab.rl.runner import MjlabOnPolicyRunner

    env = ManagerBasedRlEnv(env_cfg, device=device)
    vec_env = RslRlVecEnvWrapper(env)

    task = get_task("quadruped")
    runner_cfg = task.make_runner_cfg()
    runner_cfg.device = device
    train_cfg = runner_cfg.build_rsl_rl_cfg()
    train_cfg["experiment_name"] = "lineup_eval"

    runner = MjlabOnPolicyRunner(vec_env, train_cfg, device=device)
    runner.load(checkpoint, load_cfg={"actor": True}, strict=True, map_location=device)
    return env, vec_env, runner.get_inference_policy(device=device)


def _total_mass(robot_xml: str) -> float:
    """Total robot mass from the compiled MuJoCo model."""
    import mujoco
    return float(mujoco.mj_getTotalmass(mujoco.MjModel.from_xml_path(robot_xml)))


def _sim_total_mass(env, device) -> "torch.Tensor":
    """Each replica's total mass from the live model, shape (n_envs,).

    Under the inertia-mismatch arm every replica is a different machine.
    """
    robot = env.scene["robot"]
    ids = robot.indexing.body_ids
    m = env.sim.model.body_mass
    # (n_envs, nbody) once any per-env randomisation has run; (nbody,) if the
    # model is shared, in which case every replica has the design's own mass.
    if m.dim() == 1:
        return torch.full((env.num_envs,), float(m[ids].sum()), device=device)
    return m[:, ids].sum(dim=1).to(device)


def _rollout(env, vec_env, policy, seconds: float, settle: float,
             mass, step_h_ref: float | None = None) -> dict:
    """Roll out and reduce to scalar metrics.

    Rates are over replicas; continuous metrics average post-settle steps of
    replicas still alive. `mass` is a float or an (n_envs,) tensor.
    """
    dt = float(env.step_dt) if hasattr(env, "step_dt") else float(env.cfg.sim.dt)
    n_steps, n_settle = int(seconds / dt), int(settle / dt)
    n_envs = vec_env.num_envs
    dev = env.device

    robot = env.scene["robot"]
    fell = torch.zeros(n_envs, dtype=torch.bool, device=dev)
    v_sum = torch.zeros(n_envs, device=dev)
    v_err_sum = torch.zeros(n_envs, device=dev)
    tilt_sq_sum = torch.zeros(n_envs, device=dev)
    power_sum = torch.zeros(n_envs, device=dev)
    # Joule loss, which |tau.qdot| misses: P_joule = tau^2 / k_m^2 with
    # k_m^2 = tau_stall / omega_NL, both the joint's rated values.
    joule_sum = torch.zeros(n_envs, device=dev)
    live_steps = torch.zeros(n_envs, device=dev)
    # Path-integrated body-frame forward distance (resets teleport the root, so
    # x_end - x_start is meaningless).
    dist_sum = torch.zeros(n_envs, device=dev)
    # Net planar displacement: per-step deltas with reset steps masked out. Unlike
    # dist_sum, walking in circles does not count.
    netx_sum = torch.zeros(n_envs, device=dev)
    nety_sum = torch.zeros(n_envs, device=dev)
    # Net height change, same masking; on stairs |dz| > one step means a step taken.
    netz_sum = torch.zeros(n_envs, device=dev)
    xy_prev: "torch.Tensor | None" = None
    z_prev: "torch.Tensor | None" = None
    power_ok = True
    # ── Gait shape: joint excursion, body bounce, step frequency ────────────
    qpos_hist: list[torch.Tensor] = []
    rootz_hist: list[torch.Tensor] = []
    # ── Footfall schedule, reduced by draft.tasks.gait_metrics ──────────────
    contact_hist: list[torch.Tensor] = []
    feet_sensor = env.scene.sensors.get("feet_ground_contact")
    # Foot order from the sensor's resolved names; a wrong order turns a trot into a pace.
    feet_order = tuple(n.split("_")[0] for n in feet_sensor.primary_names) \
        if feet_sensor is not None else ()

    # Same envelope reader as collect_traces and the motor figures.
    from draft.tasks.motor_logger import MotorLogger
    _sat, _vnl, _ = MotorLogger(vec_env)._read_limits()
    km2 = (torch.as_tensor(_sat, device=dev, dtype=torch.float32)
           / torch.as_tensor(_vnl, device=dev, dtype=torch.float32).clamp(min=1e-6))

    obs = vec_env.get_observations()   # TensorDict, not a (obs, extras) tuple
    with torch.inference_mode():
        for t in range(n_steps):
            obs, _, dones, extras = vec_env.step(policy(obs))

            # A fall is a non-timeout termination; latched, though mjlab resets the replica.
            time_outs = env.termination_manager.time_outs
            newly_fell = dones.bool() & ~time_outs.bool()
            fell |= newly_fell

            if t < n_settle:
                # Displacement reference is taken at the end of settle.
                xy_prev = robot.data.root_link_pos_w[:, :2].clone()
                z_prev = robot.data.root_link_pos_w[:, 2].clone()
                continue

            alive = (~fell).float()
            v_x = robot.data.root_link_lin_vel_b[:, 0]
            cmd_x = env.command_manager.get_command("twist")[:, 0]
            v_sum += v_x * alive
            v_err_sum += (v_x - cmd_x).abs() * alive

            # Tilt from the projected-gravity observation: 0 upright, grows with lean.
            pg = robot.data.projected_gravity_b
            tilt = torch.linalg.norm(pg[:, :2], dim=-1)
            tilt_sq_sum += tilt.pow(2) * alive

            # Mechanical power sum |tau . qdot| (qfrc_actuator, joint space). Optional:
            # on failure CoT becomes NaN and the other metrics survive.
            if power_ok:
                try:
                    tau = robot.data.qfrc_actuator
                    power_sum += (tau * robot.data.joint_vel).abs().sum(-1) * alive
                    joule_sum += (tau.pow(2) / km2).sum(-1) * alive
                except Exception as exc:  # noqa: BLE001 - metric is optional
                    print(f"  [warn] power unavailable ({exc}); CoT -> NaN",
                          flush=True)
                    power_ok = False

            dist_sum += v_x * dt * alive
            xy_now = robot.data.root_link_pos_w[:, :2].clone()
            if xy_prev is not None:
                d_xy = xy_now - xy_prev
                # A reset teleports the root; that step's delta is not travel.
                d_xy = torch.where(dones.bool().unsqueeze(-1),
                                   torch.zeros_like(d_xy), d_xy)
                netx_sum += d_xy[:, 0] * alive
                nety_sum += d_xy[:, 1] * alive
            xy_prev = xy_now
            z_now = robot.data.root_link_pos_w[:, 2].clone()
            if z_prev is not None:
                d_z = z_now - z_prev
                d_z = torch.where(dones.bool(), torch.zeros_like(d_z), d_z)
                netz_sum += d_z * alive
            z_prev = z_now
            live_steps += alive
            qpos_hist.append(robot.data.joint_pos.clone())
            rootz_hist.append(robot.data.root_link_pos_w[:, 2].clone())
            if feet_sensor is not None and feet_sensor.data.found is not None:
                contact_hist.append(feet_sensor.data.found.clone())

    n = live_steps.clamp(min=1.0)
    v_mean = (v_sum / n)
    # Cost of transport: mechanical power / (m.g.v). Clamp v so a robot that
    # barely moved reports a large finite CoT rather than dividing by zero.
    if power_ok:
        denom = mass * 9.81 * v_mean.abs().clamp(min=1e-3)
        p_mech, p_joule = power_sum / n, joule_sum / n
        cot = p_mech / denom
        # Electrical CoT and efficiency eta = mechanical / electrical power, which
        # unlike CoT is not flattered by mass.
        cot_elec = (p_mech + p_joule) / denom
        eta = p_mech / (p_mech + p_joule).clamp(min=1e-9)
        watts = p_mech + p_joule
    else:
        cot = torch.full_like(v_mean, float("nan"))
        cot_elec = eta = watts = torch.full_like(v_mean, float("nan"))

    def _m(x):  # mean over replicas that survived, as a plain float
        keep = ~fell
        return float(x[keep].mean()) if bool(keep.any()) else float("nan")

    def _q(x, qs=(0.10, 0.25, 0.50, 0.75, 0.90)) -> list[float]:
        """Quantiles over the surviving replicas."""
        keep = ~fell
        if not bool(keep.any()):
            return [float("nan")] * len(qs)
        v = x[keep].detach().float()
        return [float(torch.quantile(v, q)) for q in qs]

    # ── Outcome per replica: fell, stuck or progressing ─────────────────────
    cmd_ref = float(env.command_manager.get_command("twist")[:, 0].abs().max())
    stuck_below = _STUCK_FRAC * cmd_ref

    # Stuck is decided on planar net displacement (a pyramid exit is radial),
    # not body-frame speed, which counts striding in place.
    _step_h_ref = step_h_ref
    live_s = (live_steps * dt).clamp(min=1e-3)
    net_disp = torch.sqrt(netx_sum.pow(2) + nety_sum.pow(2))
    reach = (cmd_ref * live_s).clamp(min=1e-3)     # what the command would cover
    progress_frac = net_disp / reach
    stuck = (~fell) & (progress_frac < _STUCK_FRAC)
    outcome = {
        "frac_fell": float(fell.float().mean()),
        "frac_stuck": float(stuck.float().mean()),
        "frac_progressing": float(((~fell) & ~stuck).float().mean()),
        "stuck_below_ms": stuck_below,
        # The same test on body-frame speed, for comparison.
        "frac_stuck_speedrule": float(((~fell) & (v_mean.abs() < stuck_below)).float().mean()),
        "net_disp_m": _m(net_disp),
        "net_disp_q_m": _q(net_disp),
        "progress_frac": _m(progress_frac),
        # Traversed: ended at least one step height from the start level. net_dz
        # is signed: negative off a mound, positive out of a pit.
        "net_dz_m": _m(netz_sum),
        "net_dz_q_m": _q(netz_sum),
        "frac_traversed": float(((~fell) & (netz_sum.abs() > _step_h_ref)).float().mean())
        if _step_h_ref else float("nan"),
    }

    gait: dict[str, float] = {}
    if qpos_hist:
        q = torch.stack(qpos_hist)                      # [T, B, J]
        z = torch.stack(rootz_hist)                     # [T, B]
        keep = ~fell
        if bool(keep.any()):
            q, z = q[:, keep], z[:, keep]
            # Excursion = 5th-95th percentile swing per joint (robust to stumbles).
            lo = torch.quantile(q, 0.05, dim=0)
            hi = torch.quantile(q, 0.95, dim=0)
            exc = (hi - lo)                              # [B, J]
            # Step frequency from mean-crossings of the most active joint.
            j = int(exc.mean(0).argmax())
            sig = q[:, :, j] - q[:, :, j].mean(0, keepdim=True)
            crossings = ((sig[:-1] * sig[1:]) < 0).float().sum(0)
            freq = crossings / (2.0 * max(len(qpos_hist) * dt, 1e-6))
            gait = {
                "joint_excursion_rad": float(exc.mean()),
                "max_joint_excursion_rad": float(exc.mean(0).max()),
                "body_bounce_m": float(z.std(dim=0).mean()),
                "step_freq_hz": float(freq.mean()),
            }
            gait.update(joint_symmetry_metrics(q, list(robot.joint_names)))
            if contact_hist:
                # [T, B, F], surviving replicas only.
                gait.update(contact_gait_metrics(
                    torch.stack(contact_hist)[:, keep].reshape(
                        len(contact_hist), int(keep.sum()), -1),
                    dt, feet_order or DEFAULT_FEET))

    return {
        "fall_rate": float(fell.float().mean()),
        "survive_rate": float((~fell).float().mean()),
        **outcome,
        "speed_mean_ms": _m(v_mean),
        # p10/p25/p50/p75/p90 over survivors.
        "speed_q_ms": _q(v_mean),
        "cot_q": _q(cot),
        "cot_elec": _m(cot_elec),
        "cot_elec_q": _q(cot_elec),
        "eta": _m(eta),
        "eta_q": _q(eta),
        "watts": _m(watts),
        "distance_q_m": _q(dist_sum),
        "speed_track_err_ms": _m(v_err_sum / n),
        "tilt_rms": _m((tilt_sq_sum / n).sqrt()),
        "distance_m": _m(dist_sum),
        "cost_of_transport": _m(cot),
        # Mean mass; `mass_kg_q` is None unless replica masses actually vary.
        "mass_kg": float(mass.mean()) if torch.is_tensor(mass) else mass,
        "mass_kg_q": (_q(mass) if torch.is_tensor(mass)
                      and float(mass.max() - mass.min()) > 1e-9 else None),
        "n_envs": int(n_envs),
        **gait,
    }


# ─────────────────────────────────────────────────────────────────────────────
def _base_cfg(robot_xml: str, clamp: bool, terrain_scan_fix: str = "all"):
    """The training env cfg, before re-pinning.

    ``terrain_scan_fix`` changes the observation, so it must match training.
    """
    cfg, ctx = make_base_env_cfg(
        robot_xml=robot_xml, n_envs=_N_ENVS, episode_length_s=_EPISODE_S,
        play=False, clamp_command_to_vmax=clamp, terrain_scan_fix=terrain_scan_fix,
    )
    # A battery states its level outright; no curriculum.
    cfg.curriculum.clear()
    return cfg, ctx


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # Any directory under generated/, so no fixed choices=.
    ap.add_argument("--robot", required=True,
                    help="name of a directory under generated/")
    ap.add_argument("--battery", required=True,
                    choices=("flat", "speed", "terrain", "push"))
    ap.add_argument("--checkpoint", required=True, help="trained .pt for this cell")
    ap.add_argument("--out", default=None, help="JSON out (default logs/lineup_eval/)")
    ap.add_argument("--device", default="cuda:0")
    # By default the terrain battery pools both stair types.
    ap.add_argument("--stairs-only", default=None,
                    choices=("pyramid_stairs", "pyramid_stairs_inv"),
                    help="restrict the terrain battery to ONE stair sub-terrain")
    # Inertia-mismatch arm: same policy, each replica a different build drawn
    # from the twins' measured inertia error.
    ap.add_argument("--dr-inertia", action="store_true",
                    help="perturb link inertias by the measured twin mismatch")
    ap.add_argument("--dr-seed", type=int, default=0,
                    help="seed for the inertia draw (one draw per replica)")
    ap.add_argument("--dr-spec", default=None,
                    help="inertia-mismatch spec (default experiments/twin_inertia_mismatch.json); "
                         "train_all.py passes the study's, see draft.lineup.study")
    ap.add_argument("--terrain-scan-fix", default="all",
                    help="the training run's terrain_scan_fix (see base_env). Must "
                         "match the run that trained the checkpoint; train_all.py "
                         "reads it from that run's own config.")
    args = ap.parse_args()

    robot_xml = f"generated/{args.robot}/quadruped.xml"
    mass = _total_mass(robot_xml)
    configure_torch_backends()
    device = args.device if torch.cuda.is_available() else "cpu"

    rows = []
    # Each level is a fresh env: terrain and push are baked into the cfg.
    if args.battery == "flat":
        levels = [("walk_ms", COMMON_WALK_MS)]
    elif args.battery == "speed":
        levels = [("cmd_ms", v) for v in SPEED_GRID_MS]
    elif args.battery == "terrain":
        levels = [("step_h_m", h) for h in STEP_HEIGHT_GRID_M]
    else:
        levels = [("push_dv_ms", dv) for dv in PUSH_DV_GRID_MS]

    for key, level in levels:
        cfg, ctx = _base_cfg(robot_xml, clamp=False,  # battery states the command
                             terrain_scan_fix=args.terrain_scan_fix)
        if args.battery in ("flat", "speed", "push"):
            _flat_terrain(cfg)
        if args.battery == "speed":
            _pin_command(cfg, vx=level)
        else:
            _pin_command(cfg, vx=COMMON_WALK_MS)
        if args.battery == "terrain":
            _stairs_at(cfg, step_h=level,
                       only=args.stairs_only)
        if args.battery == "push":
            _set_push(cfg, dv=level)
        else:
            cfg.events.pop("push_robot", None)  # isolate: no pushes off-battery

        if args.dr_inertia:
            from draft.tasks.quadruped.inertia_dr import add_inertia_mismatch_events
            from draft.twins.inertia_perturb import load_spec
            # Seeded per level, so rungs do not share one population of builds.
            dr_spec = add_inertia_mismatch_events(
                cfg, robot_xml, leg_length=ctx.leg_length,
                spec=load_spec(args.dr_spec),
                seed=args.dr_seed * 1000 + len(rows))

        env, vec_env, policy = _load_policy(robot_xml, cfg, args.checkpoint, device)
        # Per-replica mass, so CoT is right under the mismatch arm.
        m = _rollout(env, vec_env, policy, _EPISODE_S, _SETTLE_S,
                     _sim_total_mass(env, device),
                     # Only terrain has a rung height; elsewhere frac_traversed is NaN.
                     step_h_ref=(level if args.battery == "terrain" else None))
        env.close()
        if args.dr_inertia:
            m["dr_inertia"] = True
            m["dr_spec"] = args.dr_spec or "experiments/twin_inertia_mismatch.json"
            m["dr_mass_ratio"] = {g: v["mass_ratio"] for g, v in dr_spec.items()}
            m["mass_kg_stated"] = mass

        m[key] = level
        m["robot"], m["battery"] = args.robot, args.battery
        # Leg-lengths/s alongside m/s.
        m["leg_length_m"] = ctx.leg_length
        m["speed_mean_leg_s"] = m["speed_mean_ms"] / ctx.leg_length
        m["v_max_design_ms"] = ctx.v_max
        if args.battery == "push":
            # Impulse = mass x dv, using the population's mean mass.
            m["push_impulse_Ns"] = level * m["mass_kg"]
        rows.append(m)
        print(f"[{args.robot}/{args.battery}] {key}={level:g}  "
              f"survive={m['survive_rate']:.2f}  v={m['speed_mean_ms']:.2f} m/s  "
              f"tilt={m['tilt_rms']:.3f}  "
              f"exc={m.get('joint_excursion_rad', float('nan')):.3f}rad  "
              f"bounce={m.get('body_bounce_m', float('nan')):.3f}m  "
              f"step={m.get('step_freq_hz', float('nan')):.2f}Hz", flush=True)

    out = Path(args.out or f"logs/lineup_eval/{args.robot}_{args.battery}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "robot": args.robot, "battery": args.battery,
        "checkpoint": args.checkpoint,
        "dr_inertia": bool(args.dr_inertia),
        "dr_seed": args.dr_seed if args.dr_inertia else None,
        "rows": rows,
    }, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
