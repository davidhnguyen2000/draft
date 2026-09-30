"""The 120 run configs expanded from `experiments/lineup.yaml`.

Pinned by a digest over what a run acts on (task, seed, iterations, warm-start
source, every `env_kwarg`; not comments or layout), the per-design command
ceilings, and the sweep's structure.
"""

import pytest

from draft.lineup import configs

#: sha256 over the 120 expanded runs: the configurations the paper trained.
DIGEST = "65825b294d229e4f60938a9d5bf75bac190f84ad18b0a91353822b1fbb708fb4"

#: `estimate_top_speed` on each machine, so no model is needed on disk.
#: `tests/test_speed_shaping.py` checks the estimator reproduces these to 2 d.p.
V_MAX = {"cheetah": 2.452074, "bear": 3.012984, "giraffe": 1.774002}


@pytest.fixture(scope="module")
def runs():
    return configs.expand(v_max=V_MAX)


def test_the_expansion_is_what_the_paper_trained(runs):
    assert configs.digest(runs) == DIGEST, (
        "the expansion no longer matches the configurations the paper was trained "
        "from. If that is deliberate, the paper's numbers no longer describe it: "
        "re-pin this digest and retrain.")


def test_it_is_three_designs_ten_seeds_four_tasks(runs):
    spec = configs.load_spec()
    assert len(runs) == 120
    assert len(spec["designs"]) == 3 and len(spec["seeds"]) == 10
    assert len(spec["tasks"]) == 4
    assert len({r.run_name for r in runs}) == 120, "run names must be unique"


def test_every_stage_two_run_warm_starts_from_its_own_seeds_flat_policy(runs):
    """Crossing seeds here would collapse the variance the sweep measures."""
    by_name = configs.by_name(runs)
    for r in runs:
        if r.stage == 1:
            assert r.warmstart_from is None
            continue
        assert r.warmstart_from in by_name, f"{r.run_name}: dangling warm-start"
        src = by_name[r.warmstart_from]
        assert src.stage == 1, f"{r.run_name} warm-starts from a stage-2 run"
        assert src.seed == r.seed, (
            f"{r.run_name} (seed {r.seed}) warm-starts from seed {src.seed}")


def test_every_run_states_terrain_scan_fix(runs):
    """The study mixes settings (terrain: "all", others: "none"), so every run
    must state it rather than inherit the default."""
    from pathlib import Path
    import ast

    from draft.paths import repo_root

    base = repo_root() / "src" / "draft" / "tasks" / "quadruped" / "base_env.py"
    known = next(
        ast.literal_eval(n.value)
        for n in ast.walk(ast.parse(Path(base).read_text()))
        if isinstance(n, ast.Assign)
        and any(getattr(t, "id", None) == "TERRAIN_SCAN_FIXES" for t in n.targets))

    for r in runs:
        fix = r.env_kwargs.get("terrain_scan_fix")
        assert fix is not None, f"{r.run_name} does not state terrain_scan_fix"
        assert fix in known, f"{r.run_name}: unknown fix {fix!r}"
    assert {r.env_kwargs["terrain_scan_fix"] for r in runs} == {"all", "none"}, \
        "the paper used exactly two of the three settings"


def test_the_clamped_axes_state_a_ceiling_and_the_others_do_not(runs):
    """A clamped axis needs a v_max; an unclamped one must not imply a fake one."""
    for r in runs:
        clamped = r.env_kwargs.get("clamp_command_to_vmax", False)
        has_vmax = "v_max_override" in r.env_kwargs
        assert clamped == has_vmax, (
            f"{r.run_name}: clamp={clamped} but v_max_override "
            f"{'present' if has_vmax else 'absent'}")


def test_terrain_commands_half_of_the_flat_ceiling(runs):
    """Terrain runs command half the flat run's speed ceiling."""
    by = configs.by_name(runs)
    for design, top in V_MAX.items():
        flat = by[f"{design}_flat_s0"].env_kwargs["v_max_override"]
        terrain = by[f"{design}_terrain_s0"].env_kwargs["v_max_override"]
        assert flat == pytest.approx(round(top, 2))
        assert terrain == pytest.approx(round(top * 0.5, 2))


def test_narrowing_to_one_design_needs_only_that_machine(runs):
    """`train_all.py --designs bear` must not require the other machines on disk."""
    only = configs.expand(v_max={"bear": V_MAX["bear"]}, designs=["bear"])
    assert [r.run_name for r in only] == [r.run_name for r in runs if r.run_name.startswith("bear_")]
