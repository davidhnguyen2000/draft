"""`draft.analysis` actuator-vs-policy verdict on synthetic traces (no GPU)."""

import numpy as np
import pytest

from draft import analysis


def trace(tau, vel, *, stall=10.0, w_nl=20.0, limit=10.0, names=None,
          rewards=None, term_names=()) -> analysis.Trace:
    """A trace from explicit per-joint (torque, speed) series."""
    tau = np.asarray(tau, dtype=np.float32)          # [T, J]
    vel = np.asarray(vel, dtype=np.float32)
    T, J = tau.shape
    n = J if names is None else len(names)
    assert n == J
    return analysis.Trace(
        tau=tau.reshape(T, 1, J), vel=vel.reshape(T, 1, J),
        contact=np.zeros((0, 0, 0), bool),
        root_z=np.full((T, 1), 0.4, np.float32),
        vel_x=np.full((T, 1), 1.0, np.float32),
        fell=np.zeros(1, bool),
        joint_names=list(names or [f"j{i}" for i in range(J)]),
        foot_names=[],
        stall_torque=np.full(J, stall, np.float32),
        no_load_speed=np.full(J, w_nl, np.float32),
        effort_limit=np.full(J, limit, np.float32),
        reward_terms=list(term_names),
        reward=np.asarray(rewards if rewards is not None
                          else np.zeros((0, 0)), dtype=np.float32),
        dt=0.02, robot="test", scenario="synthetic",
        command_ms=1.0, leg_length_m=0.5,
    )


def test_a_joint_at_rest_under_full_torque_sits_on_the_envelope():
    """At zero speed the whole stall torque is available, so |tau|/tau_avail = 1."""
    t = trace(tau=[[10.0]] * 10, vel=[[0.0]] * 10)
    assert analysis.utilisation(t).max() == pytest.approx(1.0)
    assert analysis.summary(t)["actuator_limited"] is True


def test_the_envelope_falls_with_speed():
    """Half the no-load speed leaves half the stall torque, so 5 N·m is the edge."""
    t = trace(tau=[[5.0]] * 10, vel=[[10.0]] * 10, stall=10.0, w_nl=20.0, limit=10.0)
    assert analysis.utilisation(t).max() == pytest.approx(1.0)

    # The same torque at rest is only half of what is available there.
    slow = trace(tau=[[5.0]] * 10, vel=[[0.0]] * 10)
    assert analysis.utilisation(slow).max() == pytest.approx(0.5)


def test_the_effort_limit_caps_the_envelope_at_low_speed():
    """A joint whose effort limit bites before stall torque does."""
    t = trace(tau=[[4.0]] * 10, vel=[[0.0]] * 10, stall=10.0, limit=4.0)
    assert analysis.utilisation(t).max() == pytest.approx(1.0)


def test_one_saturated_joint_is_not_averaged_away_by_three_idle_ones():
    """One saturated knee among three idle hips must still read actuator-limited."""
    tau = [[10.0, 0.4, 0.4, 0.4]] * 20
    vel = [[0.0, 0.0, 0.0, 0.0]] * 20
    t = trace(tau, vel, names=["knee", "hip_a", "hip_b", "hip_c"])

    u = analysis.utilisation(t)
    assert u.mean() < 0.3, "the mean really is this misleading"

    s = analysis.summary(t)
    assert s["limiting_joint"] == "knee"
    assert s["limiting_envelope_p95"] == pytest.approx(1.0)
    assert s["actuator_limited"] is True
    assert "ACTUATOR limited" in analysis.verdict(s)


def test_a_policy_limited_design_says_so():
    """No joint near its envelope -> NOT actuator limited."""
    t = trace(tau=[[2.0, 1.0]] * 20, vel=[[0.0, 0.0]] * 20, stall=10.0, limit=10.0,
              names=["knee", "hip"])
    s = analysis.summary(t)
    assert s["actuator_limited"] is False
    v = analysis.verdict(s)
    assert "NOT actuator limited" in v and "buys" in v


def test_per_joint_is_ordered_worst_first_and_separates_the_two_axes():
    """Worst joint first; at 90% no-load speed, 10% torque is already saturated."""
    tau = [[1.0, 5.0]] * 20
    vel = [[18.0, 0.0]] * 20        # 18 = 0.9 * w_nl
    t = trace(tau, vel, stall=10.0, w_nl=20.0, limit=10.0, names=["fast", "slow"])

    rows = analysis.per_joint(t)
    assert [r["joint"] for r in rows] == ["fast", "slow"], "worst first"
    fast = rows[0]
    assert fast["tau_pct"] == pytest.approx(10.0, abs=0.5), "only 10% of its torque"
    assert fast["speed_pct"] == pytest.approx(90.0, abs=0.5)
    assert fast["envelope"] == pytest.approx(1.0, abs=0.01), "...but on the envelope"


def test_a_joint_past_its_no_load_speed_is_reported_not_crashed():
    """Back-driving pushes a joint past its velocity envelope; tau_avail hits 0."""
    t = trace(tau=[[1.0]] * 5, vel=[[30.0]] * 5, w_nl=20.0)
    u = analysis.utilisation(t)
    assert np.isfinite(u).all(), "a zero envelope must not produce inf or nan"
    assert u.max() == 0.0, "division by a zero envelope is recorded as 0, not inf"


def test_the_reward_panel_is_skipped_when_nothing_was_recorded(tmp_path):
    """Recording reward terms is opt-in, so absence is normal, not an error."""
    t = trace(tau=[[1.0]] * 5, vel=[[0.0]] * 5)
    assert analysis.plot_rewards(t, tmp_path / "r.png") is None


def test_the_reward_panel_draws_when_terms_are_present(tmp_path):
    pytest.importorskip("matplotlib")
    rewards = np.column_stack([np.linspace(0, 1, 20), np.zeros(20)])
    t = trace(tau=[[1.0]] * 20, vel=[[0.0]] * 20,
              rewards=rewards, term_names=("tracking", "dead_term"))
    out = analysis.plot_rewards(t, tmp_path / "r.png")
    assert out is not None and out.exists()


def test_the_plots_write_files(tmp_path):
    pytest.importorskip("matplotlib")
    t = trace(tau=[[1.0, 2.0]] * 20, vel=[[1.0, 2.0]] * 20, names=["knee", "hip"])
    assert analysis.plot_actuators(t, tmp_path / "a.png").exists()
    assert analysis.plot_task(t, tmp_path / "t.png").exists()
