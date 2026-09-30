"""Record per-motor (velocity, torque) during a rollout and plot them against
each motor's four-quadrant torque-speed band (mirrors mjlab's ``DcMotorActuator``):

    tau_max(v) = min( sat * ( 1 - v / v_nl),  effort_limit )
    tau_min(v) = max( sat * (-1 - v / v_nl), -effort_limit )

``sat`` is stall torque, ``v_nl`` no-load speed. Mirror-paired joints
(``left_/right_`` or ``fl_/fr_/rl_/rr_``) share a subplot.

    logger = MotorLogger(vec_env)   # any wrapper exposing .unwrapped
    logger.record()                 # once per step
    logger.plot(save_path="motor_curves.png")
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class MotorLogger:
    """Records joint velocity and applied actuator torque for one env each step."""

    def __init__(self, env, entity_name: str = "robot", env_idx: int = 0) -> None:
        """Args:
        env:         The (possibly wrapped) environment; must expose ``.unwrapped``
                     with a ``scene`` and ``sim``.
        entity_name: Name of the robot entity in the scene.
        env_idx:     Which parallel env to record (default 0).
        """
        base_env = env.unwrapped
        self._entity = base_env.scene[entity_name]
        self._env_idx = env_idx
        self.joint_names: list[str] = list(self._entity.joint_names)

        # Per-joint limits from the live actuators, in joint_names order.
        self.effort, self.v_nl, self.effort_limit = self._read_limits()

        self._vel: list[np.ndarray] = []
        self._tau: list[np.ndarray] = []

    # ------------------------------------------------------------------
    # Limit extraction
    # ------------------------------------------------------------------

    def _read_limits(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Read per-joint (stall torque, no-load speed, effort cap) from actuators.

        Non-DC actuators expose only ``force_limit`` and get ``v_nl = inf``.
        """
        n = len(self.joint_names)
        sat = np.full(n, np.nan)
        v_nl = np.full(n, np.inf)
        fl = np.full(n, np.nan)
        idx = {name: i for i, name in enumerate(self.joint_names)}

        def col(t, j: int) -> float:
            # Tensor is (num_envs, num_joints); take the tracked env.
            return float(t[self._env_idx, j].item())

        for act in self._entity.actuators:
            names = getattr(act, "_target_names", None)
            if not names:
                continue
            force_lim = getattr(act, "force_limit", None)
            saturation = getattr(act, "saturation_effort", None)
            vel_lim = getattr(act, "velocity_limit_motor", None)
            for j, name in enumerate(names):
                i = idx.get(name)
                if i is None:
                    continue
                if force_lim is not None:
                    fl[i] = col(force_lim, j)
                if saturation is not None:
                    sat[i] = col(saturation, j)
                if vel_lim is not None:
                    v_nl[i] = col(vel_lim, j)

        # Fall back to the continuous cap when no stall torque is exposed.
        sat = np.where(np.isnan(sat), fl, sat)
        fl = np.where(np.isnan(fl), sat, fl)
        return sat, v_nl, fl

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(self) -> None:
        """Append the current (velocity, torque) for the tracked env (1-DoF joints)."""
        data = self._entity.data
        v = data.joint_vel[self._env_idx].detach().cpu().numpy().copy()
        tau = data.qfrc_actuator[self._env_idx].detach().cpu().numpy().copy()
        self._vel.append(v)
        self._tau.append(tau)

    # ------------------------------------------------------------------
    # Joint grouping (mirror pairs share a subplot)
    # ------------------------------------------------------------------

    @staticmethod
    def _split_side(name: str) -> tuple[str, str | None]:
        """Map a joint name to (base_label, side); side is None if unpaired.

        Handles ``left_<x>``/``right_<x>`` and ``fl_/fr_/rl_/rr_<x>``.
        """
        toks = name.split("_")
        if toks and toks[0] in ("left", "right"):
            return "_".join(toks[1:]) or name, toks[0]
        prefix = name[:3]
        if prefix in ("fl_", "fr_", "rl_", "rr_"):
            row = "front" if name[0] == "f" else "rear"
            side = "left" if name[1] == "l" else "right"
            return f"{row}_{name[3:]}", side
        return name, None

    def _groups(self) -> list[tuple[str, list[tuple[int, str | None]]]]:
        """Ordered [(base_label, [(joint_idx, side), ...]), ...]."""
        order: list[str] = []
        groups: dict[str, list[tuple[int, str | None]]] = {}
        for i, name in enumerate(self.joint_names):
            base, side = self._split_side(name)
            if base not in groups:
                groups[base] = []
                order.append(base)
            groups[base].append((i, side))
        return [(b, groups[b]) for b in order]

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def _tau_bounds(self, i: int, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Signed (lower, upper) torque bounds of motor ``i`` over signed ``v``."""
        sat, v_nl, fl = self.effort[i], self.v_nl[i], self.effort_limit[i]
        upper = np.minimum(sat * (1.0 - v / v_nl), fl)
        lower = np.maximum(sat * (-1.0 - v / v_nl), -fl)
        return lower, upper

    # ------------------------------------------------------------------
    # Limit utilisation
    # ------------------------------------------------------------------

    def utilization(self, margin: float = 0.95) -> dict:
        """How hard each motor was driven, as fractions of its own envelope.

        Torque saturation is measured against the bound at the current velocity.
        ``margin`` is how close counts as saturated. Returns per-joint entries and
        a ``summary``:

          tau_sat_frac    fraction of samples riding the torque bound
          vel_sat_frac    fraction of samples above margin x no-load speed
          peak_tau_frac   largest |tau| seen, over the stall torque
          peak_vel_frac   largest |omega| seen, over the no-load speed
        """
        if not self._vel:
            return {}
        vel = np.asarray(self._vel)
        tau = np.asarray(self._tau)

        per_joint: dict[str, dict] = {}
        for i, name in enumerate(self.joint_names):
            v, tq = vel[:, i], tau[:, i]
            lower, upper = self._tau_bounds(i, v)
            # Distance to whichever bound the torque is pushing against.
            bound = np.where(tq >= 0.0, upper, lower)
            with np.errstate(divide="ignore", invalid="ignore"):
                frac_of_bound = np.abs(tq) / np.abs(bound)
            frac_of_bound = np.nan_to_num(frac_of_bound, nan=0.0, posinf=0.0)
            v_nl = self.v_nl[i]
            per_joint[name] = {
                "tau_sat_frac": float((frac_of_bound >= margin).mean()),
                "vel_sat_frac": float((np.abs(v) >= margin * v_nl).mean())
                                if np.isfinite(v_nl) else 0.0,
                "peak_tau_frac": float(np.abs(tq).max() / self.effort[i])
                                 if self.effort[i] else float("nan"),
                "peak_vel_frac": float(np.abs(v).max() / v_nl)
                                 if np.isfinite(v_nl) else float("nan"),
                "stall_torque_Nm": float(self.effort[i]),
                "no_load_speed_rad_s": float(v_nl),
            }

        agg = lambda k: float(np.mean([d[k] for d in per_joint.values()]))
        mx = lambda k: float(np.max([d[k] for d in per_joint.values()]))
        return {
            "per_joint": per_joint,
            "summary": {
                "tau_sat_frac_mean": agg("tau_sat_frac"),
                "vel_sat_frac_mean": agg("vel_sat_frac"),
                "tau_sat_frac_max": mx("tau_sat_frac"),
                "vel_sat_frac_max": mx("vel_sat_frac"),
                "peak_tau_frac_max": mx("peak_tau_frac"),
                "peak_vel_frac_max": mx("peak_vel_frac"),
                "n_samples": int(vel.shape[0]),
            },
        }

    def plot(self, save_path: str | Path | None = None, show: bool = True) -> Path | None:
        """Scatter signed operating points against each motor's band; return the saved path."""
        if not self._vel:
            print("[MotorLogger] No samples recorded; nothing to plot.")
            return None

        import matplotlib

        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        vel = np.asarray(self._vel)  # (T, n_joints), signed
        tau = np.asarray(self._tau)
        side_color = {"left": "tab:blue", "right": "tab:red", None: "tab:green"}

        groups = self._groups()
        n = len(groups)
        ncols = int(np.ceil(np.sqrt(n)))
        nrows = int(np.ceil(n / ncols))

        fig, axes = plt.subplots(
            nrows, ncols, figsize=(3.4 * ncols, 2.8 * nrows), squeeze=False
        )
        seen_sides: set[str | None] = set()
        for g, (base, members) in enumerate(groups):
            ax = axes[g // ncols][g % ncols]
            drawn_env: set[tuple] = set()
            for i, side in members:
                ax.scatter(
                    vel[:, i], tau[:, i], s=6, alpha=0.5, color=side_color[side],
                )
                seen_sides.add(side)
                # Four-quadrant torque-speed band (one pair of lines per spec).
                key = (self.effort[i], self.v_nl[i], self.effort_limit[i])
                if np.isfinite(self.v_nl[i]) and key not in drawn_env:
                    drawn_env.add(key)
                    # Span to the corner velocity where each bound meets the cap.
                    v_corner = self.v_nl[i] * (1.0 + self.effort_limit[i] / self.effort[i])
                    vv = np.linspace(-v_corner, v_corner, 400)
                    lower, upper = self._tau_bounds(i, vv)
                    ax.plot(vv, upper, "k--", lw=1.0, alpha=0.7)
                    ax.plot(vv, lower, "k--", lw=1.0, alpha=0.7)
            ax.axhline(0.0, color="0.7", lw=0.5, zorder=0)
            ax.axvline(0.0, color="0.7", lw=0.5, zorder=0)
            ax.set_title(base, fontsize=8)
            ax.tick_params(labelsize=6)
        # Hide unused axes.
        for j in range(n, nrows * ncols):
            axes[j // ncols][j % ncols].axis("off")

        # Single shared legend for the side colours actually used.
        if any(s is not None for s in seen_sides):
            from matplotlib.lines import Line2D

            handles = [
                Line2D([], [], marker="o", linestyle="", color=side_color[s], label=s)
                for s in ("left", "right")
                if s in seen_sides
            ]
            fig.legend(handles=handles, loc="upper right", fontsize=8)
        fig.supxlabel("joint velocity [rad/s]")
        fig.supylabel("applied torque [Nm]")
        fig.suptitle("Motor operating points vs. torque-speed envelope")
        fig.tight_layout()

        out: Path | None = None
        if save_path is not None:
            out = Path(save_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out, dpi=130)
            print(f"[MotorLogger] Saved motor curves: {out}")
        if show:
            try:
                plt.show()
            except Exception as exc:  # no display / headless backend
                print(f"[MotorLogger] Interactive show skipped ({exc})")
        plt.close(fig)
        return out
