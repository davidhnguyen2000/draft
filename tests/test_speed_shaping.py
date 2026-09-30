"""Speed commands share one absolute ceiling; reward shaping scales per design.

The command range (demand) is shared so designs are comparable; the shaping
scale `v_cap` comes from each design's `estimate_top_speed`. `speed_limits`
imports no simulator. Reads the paper's machines `generated/<robot>/`
(skipped, with the build command, if absent).
"""

import json
from statistics import median

import pytest

from draft.paths import repo_root

#: The paper's measured top speed per design (Table V), checked against the
#: record by `test_measured_speeds_match_the_frozen_record`.
MEASURED_MS = {"cheetah": 3.6258, "bear": 4.3333, "giraffe": 2.1739}

#: Flat/velocity command cap per design (terrain uses half); equals
#: `estimate_top_speed` on each machine.
CONFIG_V_MAX = {"cheetah": 2.45, "bear": 3.01, "giraffe": 1.77}

#: Which of the two limits binds per design.
BINDS_ON = {"cheetah": "froude", "bear": "both", "giraffe": "actuator"}


def _xml(name):
    p = repo_root() / "generated" / name / "quadruped.xml"
    if not p.exists():
        pytest.skip(f"{name} not built — python "
                    f"scripts/generate_quadrupeds.py --only {name}")
    return p


@pytest.fixture(scope="module")
def est():
    from draft.tasks.quadruped.speed_limits import estimate_top_speed
    return estimate_top_speed


def test_measured_speeds_match_the_frozen_record():
    """`MEASURED_MS` matches `experiments/results.json`."""
    rec = json.loads(
        (repo_root() / "experiments" / "results.json").read_text())
    nom = rec["cells"]["nom"]
    for name, expected in MEASURED_MS.items():
        got = median(c["top_speed_ms"] for c in nom[name])
        assert got == pytest.approx(expected, rel=1e-3), name


@pytest.mark.parametrize("name", sorted(CONFIG_V_MAX))
def test_the_estimate_is_what_the_committed_configs_were_built_with(name, est):
    """`estimate_top_speed` on each machine reproduces its config ceiling."""
    assert est(_xml(name)) == pytest.approx(CONFIG_V_MAX[name], abs=0.005), name


@pytest.mark.parametrize("name", sorted(CONFIG_V_MAX))
def test_the_expanded_configs_state_that_ceiling(name, est):
    """The expanded configs state that ceiling (terrain: half), checked against
    the real estimator (complements `tests/test_lineup_configs.py`)."""
    from draft.lineup import configs

    _xml(name)          # skip, with the build command, if the machines are absent
    by = configs.by_name(configs.expand(designs=[name]))
    flat = by[f"{name}_flat_s0"].env_kwargs["v_max_override"]
    terrain = by[f"{name}_terrain_s0"].env_kwargs["v_max_override"]
    assert flat == pytest.approx(CONFIG_V_MAX[name])
    assert flat == pytest.approx(round(est(_xml(name)), 2))
    assert terrain == pytest.approx(CONFIG_V_MAX[name] / 2, abs=0.01), \
        "terrain trains at half the flat ceiling"


@pytest.mark.parametrize("name", sorted(MEASURED_MS))
def test_the_estimate_beats_the_heuristic_it_replaces(name, est):
    """`estimate_top_speed` is closer to measured speed than the simpler
    `measure_max_body_speed` heuristic (which still sets `ctx.v_max`)."""
    qe = pytest.importorskip("draft.tasks.quadruped.quadruped_entity",
                             reason="the OLD heuristic still lives beside mjlab")
    measure_max_body_speed = qe.measure_max_body_speed
    new = abs(est(_xml(name)) / MEASURED_MS[name] - 1.0)
    old = abs(measure_max_body_speed(_xml(name)) / MEASURED_MS[name] - 1.0)
    assert new < old, f"{name}: new {new:.2f} is no better than old {old:.2f}"


@pytest.mark.parametrize("name", sorted(MEASURED_MS))
def test_the_speed_bonus_can_fire_on_the_ramps_first_rung(name, est):
    """The bonus gate |cmd_x| > 0.5 * v_cap lies below the ramp's opening command,
    so the forward-speed reward is live from the start."""
    from draft.tasks.quadruped.speed_limits import (COMMON_V_CEILING,
                                                    SHAPE_HEADROOM)
    v_cap = SHAPE_HEADROOM * est(_xml(name))
    opening_cmd = 0.4 * COMMON_V_CEILING           # stages[0] in make_env_cfg
    assert 0.5 * v_cap < opening_cmd, (
        f"{name}: bonus gate {0.5 * v_cap:.2f} m/s is above the ramp's opening "
        f"command {opening_cmd:.2f} m/s — it cannot fire until the ramp climbs")


@pytest.mark.xfail(
    strict=True,
    reason="The paper's policies caught up with the shaping cap. v_cap is "
           "SHAPE_HEADROOM (1.5) x the estimate, and the estimate under-reads "
           "what the trained policy reaches by 18-32%, so the cap lands at "
           "3.68 m/s against the cheetah's measured 3.63 and 4.52 against the "
           "bear's 4.33 — above what they do, but nowhere near the 20% margin "
           "this asks for. The giraffe still clears it. Recorded rather than "
           "retuned: raising SHAPE_HEADROOM changes what every stage-one policy "
           "optimises, so it is a new study, not a fix. If a retrain makes this "
           "pass, the strict xfail turns red and you update the constants here.")
@pytest.mark.parametrize("name", ["cheetah", "bear"])
def test_the_bonus_has_headroom_above_what_the_design_already_does(name, est):
    """v_cap should exceed the measured speed by 20% (known to fail; see xfail)."""
    from draft.tasks.quadruped.speed_limits import SHAPE_HEADROOM
    v_cap = SHAPE_HEADROOM * est(_xml(name))
    assert v_cap > MEASURED_MS[name] * 1.2, name


def test_the_bonus_still_has_headroom_on_the_design_that_clears_it(est):
    """The giraffe meets the 20% headroom."""
    from draft.tasks.quadruped.speed_limits import SHAPE_HEADROOM
    v_cap = SHAPE_HEADROOM * est(_xml("giraffe"))
    assert v_cap > MEASURED_MS["giraffe"] * 1.2


@pytest.mark.parametrize("name", sorted(MEASURED_MS))
def test_the_bonus_is_at_least_live_at_the_speed_the_design_reaches(name, est):
    """v_cap exceeds the measured speed, so the bonus is not flat where the
    policy operates."""
    from draft.tasks.quadruped.speed_limits import SHAPE_HEADROOM
    v_cap = SHAPE_HEADROOM * est(_xml(name))
    assert v_cap > MEASURED_MS[name], (
        f"{name}: shaping cap {v_cap:.2f} m/s is BELOW the measured "
        f"{MEASURED_MS[name]:.2f} m/s — the speed bonus is flat wherever the "
        f"policy actually operates")


def test_the_command_ceiling_stays_shared_and_absolute():
    """The command ceiling is shared (15 m/s) and does not bind on any design."""
    from draft.tasks.quadruped.speed_limits import COMMON_V_CEILING
    assert COMMON_V_CEILING == 15.0

    # 2x margin over the fastest design.
    fastest = max(MEASURED_MS.values())
    assert COMMON_V_CEILING > 2 * fastest, (
        f"the shared ceiling {COMMON_V_CEILING} m/s is only "
        f"{COMMON_V_CEILING / fastest:.2f}x the fastest design ({fastest:.2f} "
        f"m/s) and is starting to bind on it")
