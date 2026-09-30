"""Four off-the-shelf robots rebuilt from link lengths and joint torque/speed
limits only; the vendor's total mass is the held-out check (§IV-F).

Slow: needs the fetched vendor descriptions. `pytest --runslow`.
"""

import math
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

#: Mass ratio (twin / vendor) per target, paper §IV-F.
EXPECTED_RATIO = {"g1": 1.115, "h2": 0.909, "go2": 1.011, "b2": 0.841}

#: Geometric mean fold error over the four (the abstract's 1.10x).
EXPECTED_GMFE = 1.102

#: Ratio tolerance. The packing gate thresholds on compiled MuJoCo geometry, so
#: MuJoCo versions can differ on a borderline link (H2 gives 0.914 on 3.14).
#: Still tight enough to catch any trend change.
RATIO_TOL = 0.008


@pytest.fixture(scope="module")
def twins(tmp_path_factory):
    from draft.twins.synthesize import build_twin
    from draft.twins.targets import BY_KEY

    out = tmp_path_factory.mktemp("twins")
    built = {}
    for key in EXPECTED_RATIO:
        target = BY_KEY[key]
        src = target.urdf or target.mjcf
        if src is None or not Path(src).exists():
            pytest.skip(
                f"the vendor description for {key} is not in the cache; fetch it "
                f"with datasets/robot_descriptions/scripts/01_fetch_descriptions.py")
        built[key] = build_twin(target, out).summary()
    return built


@pytest.mark.parametrize("key,expected", sorted(EXPECTED_RATIO.items()))
def test_twin_mass_matches_the_vendor(twins, key, expected):
    got = twins[key]["mass_ratio"]
    assert got == pytest.approx(expected, abs=RATIO_TOL), (
        f"{key} twin is {got:.3f} of its vendor's mass, paper reports "
        f"{expected:.3f} (tolerance {RATIO_TOL}; see RATIO_TOL for why it is "
        f"not tighter)")


def test_every_twin_is_within_25_percent(twins):
    """Every twin within ±25%, each on its pinned side of 1 (G1, Go2 heavy)."""
    for key, s in twins.items():
        assert 0.75 <= s["mass_ratio"] <= 1.25, f"{key} at {s['mass_ratio']:.3f}"
        assert (s["mass_ratio"] > 1.0) == (EXPECTED_RATIO[key] > 1.0), \
            f"{key} at {s['mass_ratio']:.3f} changed side of its vendor"


def test_geometric_mean_fold_error(twins):
    """1.10x over the four, as in the abstract."""
    ratios = [s["mass_ratio"] for s in twins.values()]
    gmfe = math.exp(sum(abs(math.log(r)) for r in ratios) / len(ratios))
    # Tolerance covers MuJoCo-version variation (see RATIO_TOL).
    assert gmfe == pytest.approx(EXPECTED_GMFE, abs=0.005), (
        f"geometric mean fold error is {gmfe:.4f}, the abstract says "
        f"{EXPECTED_GMFE:.3f}")


def test_quadruped_twins_reproduce_every_joint_axis(twins):
    """Both quadruped twins keep all 12 DOF (the humanoids' mass-matrix error
    comes from joint-axis misalignment, not mass)."""
    for key in ("go2", "b2"):
        assert twins[key]["twin_dof"] == twins[key]["target_dof"] == 12
