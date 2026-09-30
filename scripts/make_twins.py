#!/usr/bin/env python3
"""Rebuild the four off-the-shelf platforms from the trends and score them (§IV-F).

A *twin* takes only the vendor's geometry and per-joint torque/speed limits;
every mass, inertia and armature comes from the trends.

Steps, each needing the one before:

    build     the twin
    adapt     the vendor model, renamed/re-rooted to match the twin's joints
    inertia   per-group residual -> experiments/twin_inertia_mismatch.json
              (used by §V's perturbed evaluation)
    render    the plates for the paper's twin figure

    python scripts/make_twins.py                    # all four steps, all targets
    python scripts/make_twins.py --only go2 b2      # two targets
    python scripts/make_twins.py --steps build      # just the twins
    python scripts/make_twins.py --steps render --mujoco   # no browser

No vendor model ships with Draft: run `python scripts/setup_data.py` first.
A target whose description is missing is skipped.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import date
from pathlib import Path

from draft.paths import repo_root
from draft.twins.targets import BY_KEY, resolve

REPO = repo_root()
TWINS = REPO / "generated" / "twins"

STEPS = ("build", "adapt", "inertia", "render")

#: Quadrupeds only: the humanoid twins' joint-axis misalignment would read as
#: mass error, and §V perturbs quadrupeds.
INERTIA_POPULATION = ("go2", "b2")


def _rel(p: Path) -> Path:
    """`p` relative to the checkout when it is inside it, else as given."""
    return p.relative_to(REPO) if p.is_relative_to(REPO) else p


# ── build ────────────────────────────────────────────────────────────────────

def step_build(targets) -> int:
    from draft.twins.synthesize import build_twin

    TWINS.mkdir(parents=True, exist_ok=True)
    index, missing, failed = [], [], 0

    for target in targets:
        print(f"\n=== {target.label} ({target.category}) ===")
        if not target.urdf.exists():
            print(f"  SKIP — description not found: {target.urdf}")
            missing.append(target.label)
            continue
        try:
            res = build_twin(target, TWINS, REPO)
        except Exception:               # one bad target must not take the rest down
            traceback.print_exc()
            print(f"  FAILED — {target.label}")
            failed += 1
            continue

        t, w = res.target, res.twin
        print(f"  target {t.total_mass:7.2f} kg / {t.n_dof:2d} DOF   "
              f"twin {w.total_mass:7.2f} kg / {w.n_dof:2d} DOF   "
              f"ratio {w.total_mass / t.total_mass:.3f}")
        if res.pruned_roles:
            print(f"  pruned to match the target: {', '.join(res.pruned_roles)}")
        if res.separation_floors:
            print("  actuator packing forced these links longer than measured: "
                  + ", ".join(f"{k}={v * 1e3:.0f} mm"
                              for k, v in sorted(res.separation_floors.items())))
        if res.law_relaxed:
            print(f"  trend relaxed — {len(res.warnings)} frontier violation(s)")
            for msg in res.warnings:
                print(f"    ! {msg}")

        (res.out_dir / "measurements.json").write_text(json.dumps({
            "target": t.as_dict(), "twin": w.as_dict(),
            "summary": res.summary(),
            "target_meta": {"key": target.key, "vendor": target.vendor,
                            "notes": target.notes, "citation": target.citation,
                            "urdf": str(target.urdf),
                            "mjcf": str(target.mjcf) if target.mjcf else None},
            "law_warnings": res.warnings,
        }, indent=1))
        index.append(res.summary())

    (TWINS / "index.json").write_text(json.dumps(index, indent=1))
    print(f"\nwrote {len(index)} twins to {TWINS.relative_to(REPO)}")
    if missing:
        print(f"\n{len(missing)} target(s) skipped for want of a vendor "
              f"description: {', '.join(missing)}")
        print("Draft distributes no robot models. Fetch them with:\n"
              "    python scripts/setup_data.py\n"
              "and re-run. If a vendor has removed the repository, that target "
              "stays skipped and the rest are unaffected.")
    return 1 if failed else 0


# ── adapt ────────────────────────────────────────────────────────────────────

def step_adapt(targets) -> int:
    """Rename/re-root each vendor model to match its twin joint-for-joint.

    Mass, inertia and limits are untouched; `rl_adapter_report.yaml` logs changes.
    """
    from draft.twins.render import urdf_for_mujoco
    from draft.twins.rl_adapter import adapt_quadruped
    from draft.twins.rl_adapter_humanoid import adapt_humanoid

    failures = 0
    for t in targets:
        twin_dir = TWINS / t.key
        stem = "quadruped.xml" if t.category == "quadruped" else "humanoid.xml"
        twin_mjcf = twin_dir / stem
        if not twin_mjcf.exists():
            print(f"[skip] {t.key}: no twin at "
                  f"{twin_mjcf.relative_to(REPO)} — run the build step first")
            failures += 1
            continue

        # B2 ships no MJCF and H2's is a closed-loop model, so both are built
        # from a URDF, converted the same way the renders do.
        shipped = t.rl_source or t.mjcf
        if shipped is None or Path(shipped).suffix.lower() == ".urdf":
            shipped = urdf_for_mujoco(Path(shipped or t.urdf), twin_dir / "shipped_mjcf")
        if shipped is None or not Path(shipped).exists():
            # Usually a missing mesh reader; the conversion prints the reason.
            print(f"[skip] {t.key}: could not build a comparable vendor model "
                  f"from {t.rl_source or t.mjcf or t.urdf}.\n"
                  f"        If the line above names a mesh reader, install the "
                  f"twins extra: pip install -e \".[twins]\"")
            failures += 1
            continue

        adapt = adapt_quadruped if t.category == "quadruped" else adapt_humanoid
        try:
            out, rep = adapt(t, Path(shipped), twin_mjcf, twin_dir / "shipped_rl")
        except Exception as exc:        # keep going; one bad target is not the set
            print(f"[FAIL] {t.key}: {exc}")
            traceback.print_exc()
            failures += 1
            continue

        worst = max((abs(v) for v in rep.zero_pose_error_deg.values()), default=0.0)
        print(f"[ok]  {t.label:12s} -> {out.relative_to(REPO)}")
        print(f"        mass {rep.total_mass_kg:6.2f} kg   "
              f"zero-pose agreement {worst:.2f} deg   "
              f"flipped {len(rep.flipped_joints)} axes")
        print(f"        effort {min(rep.effort_nm.values()):.0f}-"
              f"{max(rep.effort_nm.values()):.0f} N·m, speed "
              f"{min(rep.velocity_rad_s.values()):.1f}-"
              f"{max(rep.velocity_rad_s.values()):.1f} rad/s")
        for n in rep.notes:
            print(f"        note: {n}")
    return 1 if failures else 0


# ── inertia ──────────────────────────────────────────────────────────────────

def step_inertia(bodies: bool, out: Path | None) -> int:
    from draft.twins.inertia_mismatch import (pair_error, population, ranges,
                                              print_bodies, print_table)

    missing = [k for k in INERTIA_POPULATION
               if not (TWINS / k / "measurements.json").exists()]
    if missing:
        print(f"no twin built for {missing} — run:\n"
              f"    python scripts/make_twins.py --steps build adapt "
              f"--only {' '.join(missing)}")
        return 1

    print("=== the error a generated quadruped carries, per anatomical group ===")
    print_table(INERTIA_POPULATION, TWINS)
    if bodies:
        print_bodies(INERTIA_POPULATION, TWINS)

    pop = population(INERTIA_POPULATION, TWINS)
    spec = {
        "measured": date.today().isoformat(),
        "population": list(INERTIA_POPULATION),
        "convention": "every ratio is TARGET / TWIN; com_shift is in leg lengths, "
                      "in the root frame, left/right folded",
        "pairs": {k: {g: e.as_dict() for g, e in pair_error(k, TWINS)[0].items()}
                  for k in INERTIA_POPULATION},
        "ranges": ranges(pop),
    }
    dest = out or REPO / "experiments" / "twin_inertia_mismatch.json"
    if dest.exists():
        # A re-measurement that moves only the date, or the last digit of a float,
        # leaves the committed file alone.
        from draft.jsonio import same
        old = json.loads(dest.read_text())
        if same(old, json.loads(json.dumps(spec)), ignore=frozenset({"measured"})):
            print(f"\n{_rel(dest)} unchanged (measured {old['measured']})")
            return 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(spec, indent=2) + "\n")
    print(f"\nwrote {_rel(dest)}")
    return 0


# ── render ───────────────────────────────────────────────────────────────────

def step_render(targets, mujoco: bool, port: int, out: Path | None) -> int:
    dest = out or TWINS / "plots"
    if mujoco:
        from draft.twins.render import render_all
        render_all(TWINS, targets, dest)
    else:
        from draft.twins.viser_render import render_all
        render_all(TWINS, targets, dest, port=port)
    print(f"\nwrote plates to {_rel(dest)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", nargs="+", choices=STEPS, default=list(STEPS),
                    help="which steps to run (default: all four, in order)")
    ap.add_argument("--only", nargs="*", choices=sorted(BY_KEY), default=None,
                    help="restrict to these targets (default: all four)")
    ap.add_argument("--bodies", action="store_true",
                    help="inertia step: print the per-body detail table")
    ap.add_argument("--mujoco", action="store_true",
                    help="render step: use MuJoCo's offscreen renderer rather "
                         "than viser, which needs a local Chrome")
    ap.add_argument("--port", type=int, default=8099, help="render step: viser port")
    ap.add_argument("--out", default=None,
                    help="inertia step: the spec file; render step: the plate dir")
    args = ap.parse_args()

    targets = resolve(args.only)
    out = Path(args.out).resolve() if args.out else None
    rc = 0
    for step in STEPS:                       # always in dependency order
        if step not in args.steps:
            continue
        print(f"\n{'=' * 70}\n== {step}\n{'=' * 70}")
        if step == "build":
            rc |= step_build(targets)
        elif step == "adapt":
            rc |= step_adapt(targets)
        elif step == "inertia":
            rc |= step_inertia(args.bodies, out)
        elif step == "render":
            rc |= step_render(targets, args.mujoco, args.port, out)
    return rc


if __name__ == "__main__":
    sys.exit(main())
