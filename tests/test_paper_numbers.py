"""Every §V number the paper quotes, pinned at its printed precision.

Reads the committed `experiments/results.json` via `draft.lineup.paper_numbers`
(no GPU or logs needed).
"""
from __future__ import annotations

import pytest

from draft.lineup import paper_numbers as PN


@pytest.fixture(scope="module")
def num():
    return PN.numbers()


# (mass, L, v, v^, step, step/L, push, push^, eta), Table V.
TABLE = {
    "cheetah": (33, 0.48, 3.63, 1.67, 0.12, 0.26, 7.9, 3.65, 0.20),
    "bear":    (110, 0.74, 4.33, 1.61, 0.12, 0.17, 8.6, 3.20, 0.47),
    "giraffe": (91, 1.30, 2.17, 0.61, 0.20, 0.15, 5.3, 1.48, 0.41),
}
KEYS = ("mass_kg", "leg_m", "speed_ms", "speed_hat", "step_m", "step_per_leg",
        "push_ms", "push_hat", "eta")
DIGITS = (0, 2, 2, 2, 2, 2, 1, 2, 2)


@pytest.mark.parametrize("robot", TABLE)
def test_capability_table(num, robot):
    got = num["table"][robot]
    for key, want, nd in zip(KEYS, TABLE[robot], DIGITS):
        assert round(got[key], nd) == pytest.approx(want), (robot, key, got[key])


def test_seed_spread(num):
    s = num["seed_range"]
    assert [round(100 * s[r]["top_speed_ms"]) for r in TABLE] == [9, 14, 14]
    assert min(round(100 * s[r]["push_dv_ms"]) for r in TABLE) == 14
    assert max(round(100 * s[r]["push_dv_ms"]) for r in TABLE) == 26
    assert min(round(100 * s[r]["eta_at_top"]) for r in TABLE) == 4
    assert max(round(100 * s[r]["eta_at_top"]) for r in TABLE) == 6


def test_climbing_against_descending(num):
    s = num["stairs"]
    want = {"cheetah": (0.125, 0.125, 4), "bear": (0.175, 0.100, 9), "giraffe": (0.275, 0.1875, 7)}
    for r, (up, down, higher) in want.items():
        assert s[r]["ascent_m"] == pytest.approx(up, abs=1e-6), r
        assert s[r]["descent_m"] == pytest.approx(down, abs=1e-6), r
        assert s[r]["seeds_higher_up"] == higher, r
    # the longer the leg, the further descent lags climbing
    lag = [s[r]["ascent_m"] - s[r]["descent_m"] for r in TABLE]
    assert lag[0] < lag[1] < lag[2]
    # perturbed: cheetah's gap stays zero, giraffe's shrinks to 3.75 cm
    assert s["cheetah"]["perturbed_ascent_m"] == pytest.approx(s["cheetah"]["perturbed_descent_m"])
    assert s["giraffe"]["perturbed_ascent_m"] - s["giraffe"]["perturbed_descent_m"] == pytest.approx(0.0375, abs=1e-6)


def test_inertia_error(num):
    s = num["inertia"]
    assert [round(100 * s[r]["mass"]) for r in TABLE] == [11, 12, 9]
    speeds = [round(100 * s[r]["top_speed"], 1) for r in TABLE]
    assert (max(speeds), min(speeds)) == (5.2, -0.4)
    etas = [round(-100 * s[r]["eta"]) for r in TABLE]
    assert (min(etas), max(etas)) == (1, 4)
    assert round(-100 * s["cheetah"]["push"]) == 6
    assert round(-100 * s["giraffe"]["push"]) == 7
    assert all(s[r]["step"] >= 0 for r in TABLE)     # the tallest step falls on no design
