"""Gait metrics recover known answers from synthetic footfall patterns (numpy only)."""

import numpy as np
import pytest

from draft.tasks.gait_metrics import contact_gait_metrics, joint_symmetry_metrics

DT, T, B = 0.02, 600, 8
TIME = np.arange(T) * DT

#: Foot order throughout: front-left, front-right, rear-left, rear-right.
TROT = [0.0, 0.5, 0.5, 0.0]
BOUND = [0.0, 0.0, 0.5, 0.5]
WALK = [0.0, 0.5, 0.25, 0.75]


def contacts(phases, period=0.5, duty=0.5, jitter=0.0, duty_per_foot=None,
             seed=0):
    """`contact[T, B, 4]` from a phase-offset square wave per foot."""
    rng = np.random.default_rng(seed)
    c = np.zeros((T, B, 4), bool)
    for b in range(B):
        for f, ph in enumerate(phases):
            d = duty if duty_per_foot is None else duty_per_foot[f]
            j = rng.normal(0, jitter, T) if jitter else 0.0
            c[:, b, f] = ((TIME / period) - ph + j) % 1.0 < d
    return c


def metrics(**kw):
    return contact_gait_metrics(contacts(**kw), DT)


# ── the three canonical gaits are named correctly ────────────────────────────

@pytest.mark.parametrize("name, phases, duty", [
    ("trot", TROT, 0.5),
    ("bound", BOUND, 0.5),
    ("walk", WALK, 0.75),
])
def test_a_clean_gait_is_classified_as_itself(name, phases, duty):
    m = metrics(phases=phases, duty=duty)
    assert m["gait_class"] == name
    assert m["gait_class_share"] == pytest.approx(1.0)
    assert m["gait_defined_frac"] == pytest.approx(1.0)
    assert m["gait_class_dist"] < 0.05, "a clean gait sits on its own template"
    assert m["duty_factor"] == pytest.approx(duty, abs=0.02)


@pytest.mark.parametrize("phases", [TROT, BOUND, WALK])
def test_a_clean_gait_is_perfectly_phase_locked_and_periodic(phases):
    m = metrics(phases=phases, duty=0.75 if phases is WALK else 0.5)
    assert m["gait_phase_lock"] == pytest.approx(1.0, abs=1e-6)
    assert m["stride_period_cv"] == pytest.approx(0.0, abs=1e-6)
    assert m["contacts_per_stride"] == pytest.approx(1.0, abs=0.05)


def test_a_bound_is_left_right_symmetric():
    """Both feet of a pair land together, so there is no asymmetry to find."""
    assert metrics(phases=BOUND)["gait_lr_phase_asymmetry"] == pytest.approx(
        0.0, abs=0.01)


# ── robustness: contact jitter and a foot that never lifts ───────────────────

def test_jitter_degrades_the_lock_but_does_not_rename_the_gait():
    """0.06-cycle contact jitter lowers phase lock and regularity but a trot
    stays a trot (contacts are debounced)."""
    clean = metrics(phases=TROT)
    noisy = metrics(phases=TROT, jitter=0.06)

    assert noisy["gait_class"] == "trot", (
        "jitter renamed the gait — the contact signal is not being debounced")
    assert noisy["gait_phase_lock"] < clean["gait_phase_lock"] - 0.05
    assert noisy["stride_period_cv"] > clean["stride_period_cv"] + 0.05


def test_a_foot_that_never_lifts_reports_a_number_not_a_nan():
    """At duty 0.97 the gait is undefined, but `duty_factor` is still a number (~1)."""
    m = metrics(phases=TROT, duty=0.97)

    assert not np.isnan(m["duty_factor"]), "duty_factor went back to NaN"
    assert m["duty_factor"] == pytest.approx(1.0, abs=0.05)
    assert m["gait_defined_frac"] == pytest.approx(0.0), (
        "with no swing phase there is no gait to classify, and it should say so")


# ── the two asymmetries, which are different things ──────────────────────────

def test_an_uneven_stance_shows_in_duty_spread_and_not_in_timing():
    """A limp: right feet hold stance twice as long, but land on time."""
    m = metrics(phases=TROT, duty_per_foot=[0.4, 0.8, 0.4, 0.8])

    assert m["duty_spread"] == pytest.approx(0.4, abs=0.05), "the limp itself"
    assert m["gait_lr_phase_asymmetry"] < 0.1, (
        "timing is unaffected, so the phase asymmetry must not fire")
    assert m["gait_class"] == "trot"


def test_a_mistimed_rear_pair_shows_in_phase_asymmetry():
    """The other asymmetry: front pair offset by .5, rear by .75."""
    clean = metrics(phases=TROT)
    broken = metrics(phases=[0.0, 0.5, 0.5, 0.25])

    assert broken["gait_lr_phase_asymmetry"] > clean["gait_lr_phase_asymmetry"] + 0.1
    assert broken["gait_class_dist"] > clean["gait_class_dist"], (
        "a mistimed pair should also sit further from every gait template")


# ── joint excursion symmetry, which is amplitude and not timing ──────────────

def _swing(scale_right: float, seed: int = 0) -> tuple[np.ndarray, list[str]]:
    names = [f"{leg}_{j}" for leg in ("fl", "fr", "rl", "rr")
             for j in ("hip", "knee")]
    rng = np.random.default_rng(seed)
    q = rng.normal(0, 1e-3, (T, B, 8))
    wave = 0.30 * np.sin(2 * np.pi * TIME / 0.5)[:, None]
    for i, n in enumerate(names):
        q[:, :, i] += wave * (scale_right if n.startswith(("fr", "rr")) else 1.0)
    return q, names


def test_equal_excursions_are_symmetric():
    q, names = _swing(scale_right=1.0)
    idx = joint_symmetry_metrics(q, names)["gait_lr_excursion_symmetry_index"]
    assert idx == pytest.approx(0.0, abs=0.01)


def test_one_side_swinging_twice_as_far_reads_two_thirds():
    """|2a - a| / ((2a + a)/2) = 2/3, and the index must be that ratio."""
    q, names = _swing(scale_right=2.0)
    idx = joint_symmetry_metrics(q, names)["gait_lr_excursion_symmetry_index"]
    assert idx == pytest.approx(2 / 3, abs=0.01)
