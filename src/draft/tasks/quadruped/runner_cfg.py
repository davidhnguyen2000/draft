"""PPO runner configuration shared by the quadruped tasks.

Hyperparameters follow Rudin et al. (2021).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@dataclass
class QuadrupedLocomotionRunnerCfg:
    # ── Run identity ──────────────────────────────────────────────────────────
    run_name: str = "quadruped_locomotion"
    log_dir: str = "logs"
    seed: int = 0
    device: str = "cuda"

    # ── Logging ───────────────────────────────────────────────────────────────
    logger: str = "wandb"
    wandb_project: str = "quadruped_locomotion"
    upload_model: bool | None = None  # None → True iff logger == "wandb"

    # ── Environment knobs ─────────────────────────────────────────────────────
    n_envs: int = 4096
    play_n_envs: int = 1
    max_episode_length: float = 20.0

    # ── Training loop ─────────────────────────────────────────────────────────
    max_iterations: int = 1000
    num_steps_per_env: int = 24
    save_interval: int = 50

    # ── Networks ──────────────────────────────────────────────────────────────
    actor_hidden_dims: tuple[int, ...] = (512, 256, 128)
    critic_hidden_dims: tuple[int, ...] = (512, 256, 128)
    activation: str = "elu"
    actor_obs_normalization: bool = True
    critic_obs_normalization: bool = True
    init_std: float = 0.5
    std_type: str = "scalar"

    # ── PPO ───────────────────────────────────────────────────────────────────
    learning_rate: float = 1e-3
    schedule: str = "adaptive"
    gamma: float = 0.99
    lam: float = 0.95
    entropy_coef: float = 0.01
    desired_kl: float = 0.01
    clip_param: float = 0.2
    value_loss_coef: float = 1.0
    max_grad_norm: float = 1.0
    use_clipped_value_loss: bool = True
    num_learning_epochs: int = 5
    num_mini_batches: int = 4

    # ── Play / viewer ─────────────────────────────────────────────────────────
    viewer: str = "viser"
    checkpoint: str | None = None

    # ── Warmstart ─────────────────────────────────────────────────────────────
    # Flat-base .pt checkpoint to initialise actor+critic from (challenge tasks
    # share its obs/action space). scripts/train.py resets the counters to 0.
    warmstart_checkpoint: str | None = None

    # ── Forwarded to make_env_cfg ─────────────────────────────────────────────
    env_kwargs: dict = field(default_factory=dict)

    # ── Reward weight overrides applied after env_cfg is built ────────────────
    reward_overrides: dict = field(default_factory=dict)

    # ─────────────────────────────────────────────────────────────────────────
    def apply_overrides(self, overrides: dict) -> None:
        """Apply a flat dict of overrides in-place. Unknown keys raise."""
        valid = {f.name for f in dataclasses.fields(self)}
        for key, value in overrides.items():
            if key not in valid:
                raise KeyError(
                    f"Unknown runner_cfg field: '{key}'. "
                    f"Valid: {sorted(valid)}"
                )
            setattr(self, key, value)

    # ─────────────────────────────────────────────────────────────────────────
    def build_rsl_rl_cfg(self) -> dict:
        """Construct the dict consumed by MjlabOnPolicyRunner."""
        runner_cfg = RslRlOnPolicyRunnerCfg(
            experiment_name=self.run_name,
            run_name=self.run_name,
            max_iterations=self.max_iterations,
            num_steps_per_env=self.num_steps_per_env,
            save_interval=self.save_interval,
            seed=self.seed,
            logger=self.logger,
            wandb_project=self.wandb_project,
            upload_model=(
                self.logger == "wandb"
                if self.upload_model is None
                else self.upload_model
            ),
            actor=RslRlModelCfg(
                hidden_dims=tuple(self.actor_hidden_dims),
                activation=self.activation,
                obs_normalization=self.actor_obs_normalization,
                distribution_cfg={
                    "class_name": "GaussianDistribution",
                    "init_std": self.init_std,
                    "std_type": self.std_type,
                },
            ),
            critic=RslRlModelCfg(
                hidden_dims=tuple(self.critic_hidden_dims),
                activation=self.activation,
                obs_normalization=self.critic_obs_normalization,
            ),
            algorithm=RslRlPpoAlgorithmCfg(
                num_learning_epochs=self.num_learning_epochs,
                num_mini_batches=self.num_mini_batches,
                learning_rate=self.learning_rate,
                schedule=self.schedule,
                gamma=self.gamma,
                lam=self.lam,
                entropy_coef=self.entropy_coef,
                desired_kl=self.desired_kl,
                clip_param=self.clip_param,
                value_loss_coef=self.value_loss_coef,
                max_grad_norm=self.max_grad_norm,
                use_clipped_value_loss=self.use_clipped_value_loss,
            ),
            obs_groups={"actor": ("actor",), "critic": ("critic",)},
        )

        cfg_dict = dataclasses.asdict(runner_cfg)
        for key in ("actor", "critic"):
            for opt in ("cnn_cfg", "distribution_cfg"):
                if cfg_dict[key].get(opt) is None:
                    cfg_dict[key].pop(opt, None)
            if cfg_dict[key].get("rnn_type") is None:
                for opt in ("rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
                    cfg_dict[key].pop(opt, None)
        cfg_dict["algorithm"].setdefault("rnd_cfg", None)
        cfg_dict["algorithm"].setdefault("symmetry_cfg", None)
        cfg_dict["algorithm"].setdefault("share_cnn_encoders", False)
        cfg_dict["algorithm"].pop("multi_gpu_cfg", None)
        return cfg_dict


def make_runner_cfg() -> QuadrupedLocomotionRunnerCfg:
    return QuadrupedLocomotionRunnerCfg()
