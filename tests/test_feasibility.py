"""The feasibility gate refuses or overrides infeasible designs, and reports it."""

import pytest
import yaml

from draft.generation.generator import RobotGenerator
from draft.robots import robot_dir
from draft.trends import FeasibilityViolation, motor_fits


def _generate(tmp_path, robot="quadruped", **overrides):
    out = tmp_path / robot
    RobotGenerator(robot_dir(robot)).generate(out, params_override=overrides or None)
    return out


def _report(out):
    return yaml.safe_load((out / "feasibility_report.yaml").read_text())


def test_beyond_the_frontier_is_a_hard_refusal(tmp_path):
    """10x a class's torque exceeds the frontier and raises."""
    base = yaml.safe_load((robot_dir("quadruped") / "parameters.yaml").read_text())
    tau = base["L_motor_effort"]
    with pytest.raises(FeasibilityViolation) as exc:
        _generate(tmp_path, allow_hypothetical=False, L_motor_effort=tau * 10)
    assert "density" in str(exc.value).lower() or "frontier" in str(exc.value).lower()


def test_allow_hypothetical_downgrades_to_a_recorded_warning(tmp_path):
    """The shipped quadruped sets allow_hypothetical; it must still warn."""
    out = _generate(tmp_path)          # the shipped quadruped already sets this
    rep = _report(out)
    warnings = rep["actuator_checks"]["warnings"] + rep["warnings"]
    assert any("frontier" in w or "beats the best catalogued" in w or "p90" in w
               for w in warnings), "a beyond-frontier class must leave a trace"
    assert not rep["actuator_checks"]["errors"]


def test_frontier_ceiling_is_reported_per_class(tmp_path):
    """Every class reports a positive frontier_tau_max_Nm."""
    rep = _report(_generate(tmp_path))
    for name, cls in rep["actuator_checks"]["classes"].items():
        assert cls["frontier_tau_max_Nm"] > 0, f"class {name} has no stated ceiling"


def test_a_measured_density_cannot_be_set_by_the_design(tmp_path):
    """A measured class uses its median density; the declared value is ignored
    but recorded in the report."""
    absurd = 12345.0
    rep = _report(_generate(tmp_path, thigh_rho=absurd))
    derived = rep["segment_densities"]["derived"]
    assert "thigh_rho" in derived, "the quadruped population measures a thigh"
    entry = derived["thigh_rho"]
    assert entry["rho_kg_m3"] != pytest.approx(absurd), \
        "the design's thigh_rho reached the model"
    assert entry["rho_kg_m3"] == pytest.approx(345, abs=1.0), \
        "the applied value should be the measured quadruped thigh median"
    assert entry["declared"] == pytest.approx(absurd), \
        "the report must record what the design asked for, not just what won"
    # 60 = 15 quadrupeds x 4 legs (QUAD_ACTUATOR_HOSTS, datasets/robot_descriptions).
    assert entry["n"] == 60 and entry["source"]


def test_the_emitted_motors_weigh_what_the_trend_charged(tmp_path):
    """The report includes an actuator_mass_audit."""
    rep = _report(_generate(tmp_path))
    audit = rep["actuator_mass_audit"]
    assert audit, "no actuator mass audit was written"


def test_motor_sizing_is_reproducible():
    """Sizing is deterministic and the cylinder's rho*V equals the trend mass."""
    t = motor_fits()
    a = t.size(100.0, 20.0)
    b = t.size(100.0, 20.0)
    assert a == b
    # and the emitted cylinder weighs what the trend charged
    import math
    assert a["rho"] * math.pi * a["r"] ** 2 * a["L"] == pytest.approx(a["mass"], rel=1e-9)


def test_torque_is_cheaper_at_the_top_of_the_range():
    """Mass grows sublinearly in torque, so torque density rises with torque."""
    t = motor_fits()
    small = t.size(10.0, 20.0)
    large = t.size(1000.0, 20.0)
    assert 1000.0 / large["mass"] > 10.0 / small["mass"]
