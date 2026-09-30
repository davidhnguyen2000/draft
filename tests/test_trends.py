"""The shipped trend coefficients are the paper's Tables I and II, and the
catalogue is the paper's population (114 modules, 103 geared, 88 with rotor
inertia).
"""

import json

import pytest

from draft.trends import ACTUATOR_TRENDS_JSON, CATALOG_JSON, LINK_TRENDS_JSON, motor_fits


def test_trend_files_ship_inside_the_package():
    """The trend JSONs live inside `draft.trends` (paths resolved for symlinks)."""
    import pathlib

    import draft.trends
    pkg = pathlib.Path(draft.trends.__file__).resolve().parent
    for p in (ACTUATOR_TRENDS_JSON, LINK_TRENDS_JSON, CATALOG_JSON):
        assert p.exists(), f"{p.name} is missing"
        assert pkg in p.resolve().parents, f"{p} is outside the package"


def test_catalogue_is_the_population_the_paper_describes():
    rows = json.loads(CATALOG_JSON.read_text())
    assert len(rows) == 114, "the paper surveys 114 integrated actuators"
    by_cat = {}
    for r in rows:
        by_cat[r["category"]] = by_cat.get(r["category"], 0) + 1
    assert by_cat == {"QDD": 33, "MidGear": 33, "Harmonic": 48}
    assert sum(1 for r in rows if r.get("gear")) == 103, \
        "103 of them publish a reduction; the geared trends are fitted on those"
    assert sum(1 for r in rows if r.get("rotor_inertia_kgm2")) >= 88


def test_mass_trend_matches_the_paper():
    """m = 0.0653 tau^0.666, R^2 0.90 over n=103 (Table I); gear exponent is
    exactly zero (torque-only fit)."""
    t = json.loads(ACTUATOR_TRENDS_JSON.read_text())["mass_trend"]
    assert t["coef"] == pytest.approx(0.0653, abs=5e-4)
    assert t["tau_exp"] == pytest.approx(0.666, abs=1e-3)
    assert t["gear_exp"] == 0.0
    assert t["r2"] == pytest.approx(0.90, abs=0.01)
    assert t["n"] == 103


def test_the_dropped_gear_term_is_kept_as_evidence():
    """The two-variable fit ships alongside, showing the gear term is within one
    standard error of zero."""
    d = json.loads(ACTUATOR_TRENDS_JSON.read_text())["mass_trend"]["gear_term_dropped"]
    assert d["gear_exp"] == pytest.approx(-0.021, abs=2e-3)
    assert abs(d["gear_exp"]) < d["gear_exp_stderr"]


def test_airgap_shear_matches_the_paper():
    """sigma_eff = 11974 tau^0.300 N^-0.833 (Table I); r^2 L is imposed by physics."""
    g = json.loads(ACTUATOR_TRENDS_JSON.read_text())["geometry_trend"]
    assert g["sigma0_Pa"] == pytest.approx(11974, rel=2e-3)
    assert g["s_torque_exp"] == pytest.approx(0.300, abs=2e-3)
    assert g["k_gear_exp"] == pytest.approx(0.833, abs=2e-3)
    # R^2 of the sigma_eff regression itself.
    assert g["r2"] == pytest.approx(0.84, abs=0.01)
    # R^2 on volume, as reported in Table I and Fig. 4(c).
    assert g["r2_volume"] == pytest.approx(0.87, abs=0.01)
    assert g["n"] == 103
    # Inverted, the equation the generator solves:
    #   pi r^2 L = tau^0.700 * N^-0.167 / (2 sigma0)
    assert g["volume_tau_exp"] == pytest.approx(0.700, abs=2e-3)
    assert g["volume_gear_exp"] == pytest.approx(-0.167, abs=2e-3)


def test_slenderness_is_the_one_free_shape():
    """L/D = 0.401 tau^-0.098 N^+0.316, R^2 0.64: the default aspect when a
    design does not state one (the airgap relation fixes only r^2 L)."""
    g = json.loads(ACTUATOR_TRENDS_JSON.read_text())["geometry_trend"]
    assert g["q_coef"] == pytest.approx(0.401, abs=2e-3)
    assert g["q_tau_exp"] == pytest.approx(-0.098, abs=2e-3)
    assert g["q_gear_exp"] == pytest.approx(+0.316, abs=2e-3)
    assert g["q_r2"] == pytest.approx(0.64, abs=0.01)
    # Catalogue aspect range; outside it a class warns rather than refuses.
    a = json.loads(ACTUATOR_TRENDS_JSON.read_text())["aspect_trend"]
    assert (a["min"], a["max"]) == pytest.approx((0.282, 1.685), abs=1e-3)


def test_rotor_inertia_is_the_loosest_link():
    """J = 27.2 r^4 over 88 modules, R^2 0.87 (Table I); only the constant is fitted."""
    i = json.loads(ACTUATOR_TRENDS_JSON.read_text())["inertia_trend"]
    assert i["k_areal_kg_m2"] == pytest.approx(27.2, abs=0.1)
    assert i["r2"] == pytest.approx(0.87, abs=0.01)
    assert i["n_published"] == 88
    # r^4 is what rigid-body physics requires; the catalogue measures 4.02.
    assert i["radius_exp_physics"] == 4.0
    assert i["radius_exp_measured"] == pytest.approx(4.02, abs=0.02)
    # Reported catalogue statistic only; not used by size().
    assert i["active_fraction_mean"] == pytest.approx(0.0961, abs=1e-4)


def test_the_dropped_length_term_is_kept_as_evidence():
    """The discarded length-term fit ships alongside: its residual correlates with
    gear ratio (package length is mostly gearbox), the adopted form's does not."""
    d = json.loads(ACTUATOR_TRENDS_JSON.read_text())["inertia_trend"]["length_term_dropped"]
    # A solid cylinder requires 1.0; the catalogue says the package is not one.
    assert d["length_exp_free"] == pytest.approx(0.41, abs=0.02)
    assert abs(d["length_exp_free"] - 1.0) > 3 * d["length_exp_free_stderr"]
    assert d["resid_corr_log_gear"] < -0.5
    assert abs(d["resid_corr_log_gear_areal"]) < 0.3


def test_structural_densities_are_the_measured_medians():
    """Table II: each measured class uses its population median density."""
    trends = json.loads(LINK_TRENDS_JSON.read_text())
    hum = trends["effective_density"]
    quad = trends["quadruped"]["effective_density"]
    expected_hum = {"torso": 716, "head": 301, "upper_arm": 448, "forearm": 784,
                    "hand": 571, "thigh": 782, "shank": 533, "foot": 301}
    for seg, median in expected_hum.items():
        assert seg in hum, f"humanoid population has no {seg} density"
        assert hum[seg]["rho_eff_kg_m3"]["median"] == pytest.approx(median, abs=1.0), seg
    for seg, median in {"pelvis": 654, "thigh": 345, "shank": 644}.items():
        assert quad[seg]["rho_eff_kg_m3"]["median"] == pytest.approx(median, abs=1.0), seg


def test_there_is_exactly_one_actuator_population():
    """There is one actuator catalogue, and its fits are built once and shared."""
    import draft.trends.feasibility as F
    assert not hasattr(F, "ACTUATOR_POPULATIONS")
    assert not hasattr(F, "actuator_population")
    fits = motor_fits()
    assert motor_fits() is fits, "the trends should be built once and shared"


def test_frontier_is_a_ceiling_not_a_band():
    """Torque density rises with torque (m ~ tau^0.666), so the frontier caps
    torque; checks the frontier statistics."""
    f = json.loads(ACTUATOR_TRENDS_JSON.read_text())["frontier"]
    assert f["ALL"]["torque_density_Nm_per_kg"]["max"] == pytest.approx(159, abs=1.0)
    assert f["ALL"]["power_density_W_per_kg"]["max"] == pytest.approx(871, abs=1.0)
    assert f["ALL"]["torque_density_Nm_per_kg"]["p90"] == pytest.approx(121, abs=1.0)
    assert f["ALL"]["power_density_W_per_kg"]["p90"] == pytest.approx(419, abs=1.0)
    assert f["ALL"]["n"] == 114
    for cat in ("QDD", "MidGear", "Harmonic"):
        assert cat in f, f"{cat} must be scored against its own family"


def test_the_fitting_stage_reproduces_the_committed_trends(tmp_path):
    """Re-running `04_fit_trends.py` reproduces the committed coefficients."""
    import json
    import subprocess
    import sys

    from draft.paths import repo_root

    committed = json.loads(ACTUATOR_TRENDS_JSON.read_text())
    backup = ACTUATOR_TRENDS_JSON.read_bytes()
    stage = repo_root() / "datasets" / "actuators"
    try:
        r = subprocess.run([sys.executable, "scripts/04_fit_trends.py"],
                           cwd=stage, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-800:]
        refit = json.loads(ACTUATOR_TRENDS_JSON.read_text())
    finally:
        # never leave package data changed as a side effect of a test run
        ACTUATOR_TRENDS_JSON.write_bytes(backup)

    for block, keys in (("mass_trend", ("coef", "tau_exp", "gear_exp", "r2", "n")),
                        ("geometry_trend", ("sigma0_Pa", "s_torque_exp",
                                            "k_gear_exp", "q_coef")),
                        ("inertia_trend", ("active_fraction_mean",))):
        for key in keys:
            assert refit[block][key] == pytest.approx(committed[block][key], rel=1e-6), (
                f"{block}.{key}: committed {committed[block][key]}, "
                f"stage 04 now produces {refit[block][key]} — re-run the stage "
                f"and commit, or revert the stage.")
