"""Every shipped robot compiles, and its generated mass is pinned.

Masses are outputs of the trends, not chosen targets.
"""

from pathlib import Path

import pytest
import yaml

from draft.generation.generator import RobotGenerator
from draft.paths import repo_root
from draft.robots import available, robot_dir

#: Total mass (kg) as generated; 0.1% tolerance. Changing one means a trend,
#: density or packing rule changed.
EXPECTED_MASS_KG = {
    "humanoid": 53.7210,
    "quadruped": 31.9348,
}

#: The paper's §V designs (overrides over the quadruped).
VARIANT_MASS_KG = {
    "cheetah": 32.6381,
    "bear": 110.0500,
    "giraffe": 91.2642,
}


@pytest.mark.parametrize("robot,classes", [
    ("humanoid", {"head", "shoulder_link", "upper_arm", "forearm", "wrist_link",
                  "hand", "waist_link", "torso", "hip_link", "thigh", "shank",
                  "foot"}),
    ("quadruped", {"hip_link", "thigh", "shank"}),
])
def test_every_member_weighs_its_structural_trend(tmp_path, robot, classes):
    """Each member weighs its class trend's mass, minus fixed mass in the segment
    (floored at 5%), and is never wider than its radius cap."""
    import yaml
    out = _generate(tmp_path, robot)
    rep = yaml.safe_load((out / "feasibility_report.yaml").read_text())
    links = rep["structural_links"]
    assert {e["class"] for e in links.values()} == classes
    assert not any("declare no link_mass_class" in w for w in rep["warnings"])
    for name, e in links.items():
        want = max(e["target_kg"] - e["fixed_in_segment_kg"], 0.05 * e["target_kg"])
        assert e["emitted_member_kg"] == pytest.approx(want, rel=2e-3), name
        if e["r_cap_m"] is not None:
            assert e["r_m"] <= e["r_cap_m"] + 1e-9, name


def _generate(tmp_path, robot, overrides=None):
    out = tmp_path / robot
    RobotGenerator(robot_dir(robot)).generate(out, params_override=overrides)
    return out


def _total_mass(out: Path) -> float:
    import mujoco
    xml = next(p for p in out.glob("*.xml") if p.name != "scene.xml")
    model = mujoco.MjModel.from_xml_path(str(xml))
    return float(sum(model.body_mass))


def test_every_shipped_robot_is_discoverable():
    """Robots are discovered from their directories."""
    assert set(available()) == {"humanoid", "quadruped"}


@pytest.mark.parametrize("robot", sorted(EXPECTED_MASS_KG))
def test_robot_generates_and_compiles(tmp_path, robot):
    out = _generate(tmp_path, robot)
    xmls = {p.name for p in out.glob("*.xml")}
    assert "scene.xml" in xmls, "a generated model ships the scene it loads in"
    assert (out / "feasibility_report.yaml").exists()
    assert (out / "parameters_resolved.yaml").exists()
    # Reload the file from disk to prove it compiles as written.
    _total_mass(out)


@pytest.mark.parametrize("robot,expected", sorted(EXPECTED_MASS_KG.items()))
def test_mass_is_unchanged(tmp_path, robot, expected):
    got = _total_mass(_generate(tmp_path, robot))
    assert got == pytest.approx(expected, rel=1e-3), (
        f"{robot} weighs {got:.4f} kg, recorded {expected:.4f} kg. A trend, a "
        f"density or a drawn length changed.")


@pytest.mark.parametrize("variant,expected", sorted(VARIANT_MASS_KG.items()))
def test_paper_variants_are_unchanged(tmp_path, variant, expected):
    """Cheetah / bear / giraffe masses (§V) are pinned."""
    cfg = repo_root() / "experiments" / "quadruped_variants" / f"{variant}.yaml"
    overrides = yaml.safe_load(cfg.read_text())
    got = _total_mass(_generate(tmp_path, "quadruped", overrides))
    assert got == pytest.approx(expected, rel=1e-3)


def test_report_records_what_the_trends_took_over(tmp_path):
    """The report records what each actuator class was derived from and what the
    trends overrode."""
    out = _generate(tmp_path, "quadruped")
    report = yaml.safe_load((out / "feasibility_report.yaml").read_text())

    classes = report["actuator_checks"]["classes"]
    assert classes, "no actuator class was sized"
    for name, cls in classes.items():
        assert cls["mass_kg"] > 0, f"class {name} was charged no mass"
        assert cls["radius_m"] > 0 and cls["length_m"] > 0, (
            f"class {name} has no envelope; the airgap relation did not run")
        assert cls["armature_kgm2"] > 0, f"class {name} reflects no rotor inertia"
        assert cls["designed_from"], f"class {name} does not say what it was solved from"
        assert "overridden_by_law" in cls

    assert report["segment_densities"], "no measured segment density was applied"

    assert report["total_mass_kg"] > 0


def test_mass_by_role_sums_to_the_total(tmp_path):
    """Per-role masses sum to the total."""
    out = _generate(tmp_path, "quadruped")
    report = yaml.safe_load((out / "feasibility_report.yaml").read_text())
    by_role = report["mass_kg"]
    assert by_role, "no per-role mass breakdown"
    # Tolerance covers rounding to 4 decimals in the report.
    assert sum(by_role.values()) == pytest.approx(report["total_mass_kg"], abs=5e-4)
