"""Every number §V of the paper quotes, recomputed from the committed record.

    python scripts/train_all.py numbers            # print, and write figures/tab_capability.tex

Reads only `experiments/results.json` (written by `freeze_results.py`),
so it runs on a clean checkout with no GPU and no logs. `tests/test_paper_numbers.py`
pins the same values, so a change that moves one of them fails the suite.
"""
from __future__ import annotations

import argparse
import json
from statistics import median

from . import study
from ..paths import repo_root

ROOT = repo_root()
#: Everything §V quotes is read from this one file.
RECORD = study.STUDY.record
G = 9.81


def _med(cells: list[dict], key: str) -> float:
    return median(c[key] for c in cells)


def _range_frac(cells: list[dict], key: str) -> float:
    v = [c[key] for c in cells]
    return (max(v) - min(v)) / median(v)


def numbers(rec: dict | None = None) -> dict:
    rec = rec or json.loads(RECORD.read_text())
    nom, dr = rec["cells"]["nom"], rec["cells"]["dr"]
    out: dict = {"table": {}, "seed_range": {}, "stairs": {}, "inertia": {},
                 "walk": rec["walk_limits"], "sprint": rec["sprint_limits"]}
    for r in rec["robots"]:
        n, p = nom[r], dr[r]
        L = n[0]["leg_length_m"]
        froude = (G * L) ** 0.5
        v, push, step = _med(n, "top_speed_ms"), _med(n, "push_dv_ms"), _med(n, "step_m")
        out["table"][r] = {"mass_kg": n[0]["mass_kg"], "leg_m": L,
                           "speed_ms": v, "speed_hat": v / froude,
                           "step_m": step, "step_per_leg": step / L,
                           "push_ms": push, "push_hat": push / froude,
                           "eta": _med(n, "eta_at_top")}
        out["seed_range"][r] = {k: _range_frac(n, k)
                                for k in ("top_speed_ms", "push_dv_ms", "eta_at_top")}
        out["stairs"][r] = {
            "ascent_m": _med(n, "step_ascent_m"), "descent_m": _med(n, "step_descent_m"),
            "seeds_higher_up": sum(c["step_ascent_m"] > c["step_descent_m"] for c in n),
            "perturbed_ascent_m": _med(p, "step_ascent_m"),
            "perturbed_descent_m": _med(p, "step_descent_m")}
        out["inertia"][r] = {
            "mass": p[0]["mass_kg"] / n[0]["mass_kg"] - 1,
            "top_speed": _med(p, "top_speed_ms") / v - 1,
            "eta": _med(p, "eta_at_top") / _med(n, "eta_at_top") - 1,
            "push": _med(p, "push_dv_ms") / push - 1,
            "step": _med(p, "step_m") - step}
    return out


def table_tex(num: dict) -> str:
    """Table V body, bold marking the best design per column."""
    t = num["table"]
    cols = [("speed_ms", "{:.2f}"), ("speed_hat", "{:.2f}"), ("step_m", "{:.2f}"),
            ("step_per_leg", "{:.2f}"), ("push_ms", "{:.1f}"), ("push_hat", "{:.2f}"),
            ("eta", "{:.2f}")]
    best = {k: max(round(t[r][k], 2) for r in t) for k, _ in cols}
    lines = []
    for r in t:
        cells = [f"${t[r]['mass_kg']:.0f}$", f"${t[r]['leg_m']:.2f}$"]
        for k, fmt in cols:
            s = fmt.format(t[r][k])
            cells.append(f"$\\mathbf{{{s}}}$" if round(t[r][k], 2) == best[k] else f"${s}$")
        lines.append(f"{r.capitalize():7s} & " + " & ".join(cells) + " \\\\")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    record = study.STUDY.record
    print(f"reading {record.relative_to(ROOT)}\n")
    num = numbers(json.loads(record.read_text()))
    print("Table V  (capability; medians over ten seeds)")
    print(f"  {'design':8s} {'m':>5s} {'L':>5s} {'v':>5s} {'v^':>5s} {'step':>5s} "
          f"{'st/L':>5s} {'push':>5s} {'p^':>5s} {'eta':>5s}")
    for r, t in num["table"].items():
        print(f"  {r:8s} {t['mass_kg']:5.0f} {t['leg_m']:5.2f} {t['speed_ms']:5.2f} "
              f"{t['speed_hat']:5.2f} {t['step_m']:5.2f} {t['step_per_leg']:5.2f} "
              f"{t['push_ms']:5.1f} {t['push_hat']:5.2f} {t['eta']:5.2f}")
    print("\nseed range as a share of the median  (top speed / kick / efficiency)")
    for r, s in num["seed_range"].items():
        print(f"  {r:8s} {100 * s['top_speed_ms']:4.0f}% {100 * s['push_dv_ms']:4.0f}% "
              f"{100 * s['eta_at_top']:4.0f}%")
    print("\nclimbing against descending  (median m; seeds climbing higher; perturbed)")
    for r, s in num["stairs"].items():
        print(f"  {r:8s} up {s['ascent_m']:.3f} down {s['descent_m']:.3f}  "
              f"{s['seeds_higher_up']}/10  perturbed up {s['perturbed_ascent_m']:.3f} "
              f"down {s['perturbed_descent_m']:.3f}")
    print("\nsprint limits  (peak-command traces, pooled over ten seeds)")
    for r, s in num["sprint"].items():
        print(f"  {r:8s} knee near torque limit {100 * s['knee']['near_torque_limit']:.1f}%  "
              f"joint speed p95 {s['speed_p95_all']:.2f} of no-load  "
              f"past no-load {100 * s['past_no_load_all']:.1f}%")
    print("\nwalk limits at 0.4 m/s  (thirty seeds; fraction of limit at p95)")
    for r, s in num["walk"].items():
        print(f"  {r:8s} knee torque {s['knee']['tau_p95']:.2f} speed {s['knee']['speed_p95']:.2f} "
              f"near no-load {100 * s['knee']['near_no_load']:.1f}%   hip pitch torque "
              f"{s['hip_pitch']['tau_p95']:.2f}")
    print("\ninertia error  (perturbed against nominal)")
    for r, s in num["inertia"].items():
        print(f"  {r:8s} mass {100 * s['mass']:+.1f}%  top speed {100 * s['top_speed']:+.1f}%  "
              f"eta {100 * s['eta']:+.1f}%  kick {100 * s['push']:+.1f}%  "
              f"step {100 * s['step']:+.1f} cm")
    tex = ROOT / "figures" / "tab_capability.tex"
    tex.parent.mkdir(parents=True, exist_ok=True)
    tex.write_text(table_tex(num))
    print(f"\nwrote {tex.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
