"""Is a rollout limited by its actuators, its task, or its rewards?

Reads one ``.npz`` written by ``draft.lineup.collect_traces`` (no simulator
needed) and plots per-joint torque-speed operating points, task tracking, and
reward terms. Envelope: ``tau_avail(w) = min(effort_limit, tau_stall * (1 - |w|
/ w_NL))``; utilisation ``|tau| / tau_avail`` is 1.0 on the boundary. The summary
reports the worst joint, not the mean, since averaging hides a saturated knee.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Utilisation at or above which a joint counts as at its envelope.
NEAR_LIMIT = 0.9

#: Percentile taken as the operating point (the max is a single stumble).
PCTL = 95.0


@dataclass
class Trace:
    """One recorded rollout, as written by `draft.lineup.collect_traces`."""

    tau: np.ndarray            # [T, B, J] joint torque
    vel: np.ndarray            # [T, B, J] joint velocity
    contact: np.ndarray        # [T, B, F] foot contact, may be empty
    root_z: np.ndarray         # [T, B]
    vel_x: np.ndarray          # [T, B] forward speed in the base frame
    fell: np.ndarray           # [B] did this replica terminate early
    joint_names: list[str]
    foot_names: list[str]
    stall_torque: np.ndarray   # [J]
    no_load_speed: np.ndarray  # [J]
    effort_limit: np.ndarray   # [J]
    reward_terms: list[str]
    reward: np.ndarray         # [T, n_terms] for replica 0, may be empty
    dt: float
    robot: str
    scenario: str
    command_ms: float
    leg_length_m: float

    @classmethod
    def load(cls, path: Path | str) -> "Trace":
        z = np.load(Path(path), allow_pickle=False)

        def s(key: str) -> str:
            return str(z[key]) if key in z else ""

        return cls(
            tau=z["tau"], vel=z["vel"], contact=z["contact"],
            root_z=z["root_z"], vel_x=z["vel_x"], fell=z["fell"],
            joint_names=[str(n) for n in z["joint_names"]],
            foot_names=[str(n) for n in z["foot_names"]],
            stall_torque=z["stall_torque"], no_load_speed=z["no_load_speed"],
            effort_limit=z["effort_limit"],
            reward_terms=[str(n) for n in z["reward_terms"]]
            if "reward_terms" in z else [],
            reward=z["reward"] if "reward" in z else np.zeros((0, 0), np.float32),
            dt=float(z["dt"]), robot=s("robot"), scenario=s("scenario"),
            command_ms=float(z["command_ms"]), leg_length_m=float(z["leg_length_m"]),
        )


def available_torque(trace: Trace) -> np.ndarray:
    """`tau_avail(w)` at every recorded sample. Shape [T, B, J]."""
    w = np.abs(trace.vel)
    w_nl = np.where(trace.no_load_speed > 0, trace.no_load_speed, np.inf)
    linear = trace.stall_torque * np.clip(1.0 - w / w_nl, 0.0, None)
    return np.minimum(trace.effort_limit, linear)


def utilisation(trace: Trace) -> np.ndarray:
    """`|tau| / tau_avail(|w|)`, per sample. 1.0 sits on the envelope."""
    avail = available_torque(trace)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.abs(trace.tau) / avail
    return np.where(np.isfinite(u), u, 0.0)


def per_joint(trace: Trace) -> list[dict]:
    """Per-joint p95 torque %, speed % and envelope utilisation, worst first.

    A joint can be low on both torque and speed yet still on the envelope.
    """
    u = utilisation(trace)
    tau_frac = np.abs(trace.tau) / np.where(trace.effort_limit > 0,
                                            trace.effort_limit, np.inf)
    spd_frac = np.abs(trace.vel) / np.where(trace.no_load_speed > 0,
                                            trace.no_load_speed, np.inf)
    rows = []
    for j, name in enumerate(trace.joint_names):
        rows.append({
            "joint": name,
            "tau_pct": 100.0 * float(np.percentile(tau_frac[:, :, j], PCTL)),
            "speed_pct": 100.0 * float(np.percentile(spd_frac[:, :, j], PCTL)),
            "envelope": float(np.percentile(u[:, :, j], PCTL)),
            "frac_near_limit": float((u[:, :, j] > NEAR_LIMIT).mean()),
        })
    return sorted(rows, key=lambda r: -r["envelope"])


def summary(trace: Trace) -> dict:
    rows = per_joint(trace)
    worst = rows[0]
    achieved = float(np.median(trace.vel_x[trace.vel_x.shape[0] // 2:]))
    return {
        "robot": trace.robot,
        "scenario": trace.scenario,
        "commanded_ms": trace.command_ms,
        "achieved_ms": achieved,
        "tracking_error_ms": achieved - trace.command_ms,
        "fell_frac": float(trace.fell.mean()),
        "limiting_joint": worst["joint"],
        "limiting_envelope_p95": worst["envelope"],
        "actuator_limited": worst["envelope"] >= NEAR_LIMIT,
        "seconds": trace.tau.shape[0] * trace.dt,
        "replicas": trace.tau.shape[1],
    }


def verdict(s: dict) -> str:
    """One-sentence verdict: actuator limited or not."""
    if s["actuator_limited"]:
        return (f"ACTUATOR limited: {s['limiting_joint']} sits at "
                f"{s['limiting_envelope_p95']:.2f} of its torque-speed envelope. "
                f"A faster policy needs a bigger motor here, not more training.")
    return (f"NOT actuator limited: the hardest-working joint "
            f"({s['limiting_joint']}) reaches only "
            f"{s['limiting_envelope_p95']:.2f} of its envelope. The limit is the "
            f"reward, the gait or the training budget — a bigger motor buys "
            f"nothing.")


# ── plots ────────────────────────────────────────────────────────────────────

def _fig(nrows: int, ncols: int, w: float = 4.2, h: float = 3.2):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows))
    return fig, np.atleast_1d(axes).ravel()


def plot_actuators(trace: Trace, out: Path) -> Path:
    """Torque against speed, per joint, with that joint's own envelope drawn on."""
    import matplotlib.pyplot as plt

    rows = per_joint(trace)
    order = [trace.joint_names.index(r["joint"]) for r in rows]
    n = len(order)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = _fig(nrows, ncols)

    for ax, j in zip(axes, order):
        w = trace.vel[:, :, j].ravel()
        t = trace.tau[:, :, j].ravel()
        ax.scatter(w, t, s=1.5, alpha=0.15, rasterized=True)

        # Envelope, all four quadrants.
        w_nl = trace.no_load_speed[j]
        lim = trace.effort_limit[j]
        stall = trace.stall_torque[j]
        if w_nl > 0:
            ws = np.linspace(0, w_nl, 200)
            env = np.minimum(lim, stall * (1 - ws / w_nl))
            for sw, st in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                ax.plot(sw * ws, st * env, lw=1.0, color="k")
        name = trace.joint_names[j]
        r = next(r for r in rows if r["joint"] == name)
        ax.set_title(f"{name}\nenvelope p95 {r['envelope']:.2f}", fontsize=8)
        ax.set_xlabel("joint speed (rad/s)", fontsize=7)
        ax.set_ylabel("torque (N·m)", fontsize=7)
        ax.tick_params(labelsize=6)
    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(f"{trace.robot} — {trace.scenario}: operating points against "
                 f"each joint's own envelope", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_task(trace: Trace, out: Path) -> Path:
    """What the robot did: speed tracking, height, and the contact pattern."""
    import matplotlib.pyplot as plt

    t = np.arange(trace.vel_x.shape[0]) * trace.dt
    fig, axes = _fig(3, 1, w=7.0, h=2.3)

    axes[0].plot(t, np.median(trace.vel_x, axis=1), lw=1.2, label="achieved (median)")
    lo, hi = np.percentile(trace.vel_x, [5, 95], axis=1)
    axes[0].fill_between(t, lo, hi, alpha=0.2, label="5–95% of replicas")
    axes[0].axhline(trace.command_ms, ls="--", color="k", lw=1.0, label="commanded")
    axes[0].set_ylabel("forward speed (m/s)")
    axes[0].legend(fontsize=7)

    axes[1].plot(t, np.median(trace.root_z, axis=1), lw=1.2)
    lo, hi = np.percentile(trace.root_z, [5, 95], axis=1)
    axes[1].fill_between(t, lo, hi, alpha=0.2)
    axes[1].set_ylabel("base height (m)")

    if trace.contact.size:
        # Replica 0's footfalls.
        c = trace.contact[:, 0, :].astype(float)
        for i in range(c.shape[1]):
            on = c[:, i] > 0
            axes[2].fill_between(t, i, i + 0.8, where=on, step="mid")
        axes[2].set_yticks(np.arange(c.shape[1]) + 0.4)
        axes[2].set_yticklabels(trace.foot_names or
                                [f"foot {i}" for i in range(c.shape[1])], fontsize=7)
        axes[2].set_ylabel("contact")
    else:
        axes[2].text(0.5, 0.5, "no contact sensor in this recording",
                     ha="center", va="center", fontsize=8)
        axes[2].axis("off")
    axes[2].set_xlabel("time (s)")

    s = summary(trace)
    fig.suptitle(f"{trace.robot} — {trace.scenario}: commanded "
                 f"{s['commanded_ms']:.2f} m/s, achieved {s['achieved_ms']:.2f} m/s, "
                 f"{s['fell_frac']:.0%} of replicas fell", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_rewards(trace: Trace, out: Path) -> Path | None:
    """Each reward term over the episode, so a dead or dominant term is visible.

    Returns None if the trace has no reward terms (`collect_traces --with-rewards`).
    """
    import matplotlib.pyplot as plt

    if trace.reward.size == 0 or not trace.reward_terms:
        return None

    t = np.arange(trace.reward.shape[0]) * trace.dt
    mean_abs = np.nanmean(np.abs(trace.reward), axis=0)
    order = np.argsort(-mean_abs)

    fig, axes = _fig(2, 1, w=7.0, h=3.0)
    for i in order:
        axes[0].plot(t, trace.reward[:, i], lw=1.0, label=trace.reward_terms[i])
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("term value")
    axes[0].legend(fontsize=6, ncol=2)
    axes[0].set_title("each reward term over the episode (replica 0)", fontsize=9)

    names = [trace.reward_terms[i] for i in order]
    axes[1].barh(range(len(order)), mean_abs[order])
    axes[1].set_yticks(range(len(order)))
    axes[1].set_yticklabels(names, fontsize=7)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("mean |term value| — what the policy is actually optimising")

    dead = [n for n, m in zip(names, mean_abs[order]) if m == 0.0]
    if dead:
        axes[1].set_title(f"dead terms (identically zero): {', '.join(dead)}",
                          fontsize=8)

    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out
