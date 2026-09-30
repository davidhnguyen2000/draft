"""Play a trained policy for a registered task in a live viewer.

  python scripts/play.py --task quadruped
  python scripts/play.py --task quadruped --viewer native
  python scripts/play.py --task quadruped_terrain --checkpoint logs/.../model_500.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import draft  # noqa: F401  first: loads the environment's libstdc++, see draft/__init__.py

# Friendly error when the `rl` extra (mjlab, torch) is missing.
def _need_rl_extra(exc: ModuleNotFoundError) -> "NoReturn":   # noqa: F821
    raise SystemExit(
        f"{exc.name} is not installed.\n\n"
        "Training and playback need the RL extra, which is GPU-bound:\n"
        "    pip install -e \".[rl]\" -c constraints.txt\n\n"
        "Everything else in this repository — generating a design, the editor, "
        "the twins,\nevery figure and every number the paper reports — runs "
        "without it.")


try:
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.utils.torch import configure_torch_backends
    from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
except ModuleNotFoundError as exc:
    _need_rl_extra(exc)

from draft.tasks import get_task, list_tasks
from draft.tasks.config_loader import find_latest_checkpoint, load_run_config, require_robot_xml
from draft.tasks.motor_logger import MotorLogger


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained RL policy from a registered task."
    )
    parser.add_argument(
        "--task", type=str, default=None,
        help=f"Task name. Available: {', '.join(list_tasks())}",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Per-run YAML. Defaults to <task_path>/config.yaml.",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to .pt checkpoint (auto-selects latest if omitted).",
    )
    parser.add_argument(
        "--viewer", type=str, default=None, choices=["viser", "native"],
        help="Override viewer choice from config.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # ── Resolve task + config ────────────────────────────────────────────────
    if args.config is not None:
        run_cfg = load_run_config(args.config)
        task_name = args.task or run_cfg["task"]
    elif args.task is not None:
        task = get_task(args.task)
        run_cfg = load_run_config(task.default_config_yaml())
        task_name = args.task
    else:
        raise SystemExit("Must pass --task and/or --config.")

    task = get_task(task_name)

    # ── Build runner cfg with overrides ──────────────────────────────────────
    runner_cfg = task.make_runner_cfg()
    runner_cfg.apply_overrides(run_cfg["runner_overrides"])
    runner_cfg.env_kwargs = {**runner_cfg.env_kwargs, **run_cfg["env_kwargs"]}
    require_robot_xml(runner_cfg.env_kwargs)
    runner_cfg.reward_overrides = {
        **runner_cfg.reward_overrides, **run_cfg["reward_overrides"]
    }

    # CLI flags win over config.
    if args.checkpoint is not None:
        runner_cfg.checkpoint = args.checkpoint
    if args.viewer is not None:
        runner_cfg.viewer = args.viewer

    # ── Resolve checkpoint ───────────────────────────────────────────────────
    if runner_cfg.checkpoint == "zeros":
        print("Zero-initializing policy weights.")
    elif not runner_cfg.checkpoint:
        try:
            runner_cfg.checkpoint = str(
                find_latest_checkpoint(runner_cfg.log_dir, runner_cfg.run_name)
            )
            print(f"Auto-selected checkpoint: {runner_cfg.checkpoint}")
        except FileNotFoundError as e:
            print(f"Warning: {e}. Running with a randomly initialised policy.")

    # ── Build env cfg in play mode ───────────────────────────────────────────
    env_cfg = task.make_env_cfg(
        n_envs=runner_cfg.play_n_envs,
        episode_length_s=runner_cfg.max_episode_length,
        play=True,
        **runner_cfg.env_kwargs,
    )
    if runner_cfg.reward_overrides:
        for name, weight in runner_cfg.reward_overrides.items():
            if name in env_cfg.rewards:
                env_cfg.rewards[name].weight = float(weight)

    # ── Device ──────────────────────────────────────────────────────────────
    configure_torch_backends()
    device = runner_cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    # ── Env + runner ─────────────────────────────────────────────────────────
    env = ManagerBasedRlEnv(env_cfg, device=device)
    vec_env = RslRlVecEnvWrapper(env)

    train_cfg = runner_cfg.build_rsl_rl_cfg()
    train_cfg["experiment_name"] = "eval"

    runner = MjlabOnPolicyRunner(vec_env, train_cfg, device=device)
    if runner_cfg.checkpoint == "zeros":
        with torch.no_grad():
            for p in runner.alg.get_policy().parameters():
                p.zero_()
    elif runner_cfg.checkpoint:
        runner.load(
            runner_cfg.checkpoint,
            load_cfg={"actor": True},
            strict=True,
            map_location=device,
        )
    policy = runner.get_inference_policy(device=device)

    # ── Record per-motor velocity/torque each step ───────────────────────────
    # Wrap step so either viewer's loop logs an operating point every step.
    motor_logger = MotorLogger(vec_env)
    _orig_step = vec_env.step

    def _step_and_log(actions):
        out = _orig_step(actions)
        motor_logger.record()
        return out

    vec_env.step = _step_and_log

    viewers = {"viser": ViserPlayViewer, "native": NativeMujocoViewer}
    try:
        viewers[runner_cfg.viewer](vec_env, policy).run()
    finally:
        # On viewer exit, plot beside the checkpoint (else the working dir).
        plot_path = (
            Path(runner_cfg.checkpoint).with_name("motor_curves.png")
            if runner_cfg.checkpoint and runner_cfg.checkpoint != "zeros"
            else Path("motor_curves.png")
        )
        motor_logger.plot(save_path=plot_path)

    env.close()


if __name__ == "__main__":
    main()
