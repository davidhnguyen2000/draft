"""The inertia perturbation (§V) applies the error the twin study measured.

1. `inertia_perturb` groups bodies as `draft.twins.measure` does.
2. `alpha` (mass), `d` (shape) and `t` (COM) are separable.
3. A pair's own correction applied to its twin reproduces the vendor machine.

Skipped where the twins are not built (`scripts/make_twins.py`).
"""

import json
import math

import numpy as np
import pytest

from draft.paths import repo_root
from draft.twins.inertia_perturb import (GroupRange, _solve_d, apply_numpy,
                                         body_groups, group_inertia, limb_sets,
                                         perturb_body, resolve)

mujoco = pytest.importorskip("mujoco")

QUADS = ("go2", "b2")
GROUPS = ("trunk", "hip_link", "thigh", "shank")
TWINS = repo_root() / "generated" / "twins"


def _built(key):
    if not (TWINS / key / "measurements.json").exists():
        pytest.skip(f"{key} twin not built — scripts/make_twins.py --only {key}")
    return json.loads((TWINS / key / "measurements.json").read_text())


def _model(key):
    return mujoco.MjModel.from_xml_path(str(TWINS / key / "quadruped.xml"))


def _leg(meas):
    f = meas["target"]["features_m"]
    return float(f["thigh"] + f["shank"])


# ── 1. the grouping ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("key", QUADS)
def test_group_masses_match_the_twin_study(key):
    """`body_groups` matches `draft.twins.measure`, compared via group mass."""
    meas = _built(key)
    m = _model(key)
    got = {g: sum(float(m.body_mass[b]) for b in ids)
           for g, ids in body_groups(m).items()}
    want = meas["twin"]["group_mass_kg"]
    assert set(got) == set(want)
    for g in want:
        assert got[g] == pytest.approx(want[g], rel=1e-6), g


@pytest.mark.parametrize("key", QUADS)
def test_every_massive_body_is_in_exactly_one_group(key):
    m = _model(_built(key) and key)
    ids = [b for ids in body_groups(m).values() for b in ids]
    assert len(ids) == len(set(ids)), "a body landed in two groups"
    massive = {i for i in range(1, m.nbody) if m.body_mass[i] > 0}
    assert set(ids) == massive, "a body with mass was charged to nothing"


@pytest.mark.parametrize("key", QUADS)
def test_the_legs_are_split_four_ways(key):
    m = _model(_built(key) and key)
    g = body_groups(m)
    for name in ("hip_link", "thigh", "shank"):
        assert len(limb_sets(m, name, g[name])) == 4, name
    assert len(limb_sets(m, "trunk", g["trunk"])) == 1


# ── 2. the three parameters are separable ────────────────────────────────────

_M, _C, _I = 2.0, np.array([0.1, -0.02, 0.03]), np.diag([0.010, 0.020, 0.025])


def _k(mass, I):
    return math.sqrt(np.trace(I) / (2.0 * mass))


def test_alpha_moves_mass_and_nothing_else():
    a = 0.2
    m1, c1, I1 = perturb_body(_M, _C, _I, a, 0.0, np.zeros(3))
    assert m1 == pytest.approx(_M * math.exp(2 * a))
    assert np.allclose(c1, _C)
    assert _k(m1, I1) == pytest.approx(_k(_M, _I))
    assert np.allclose(I1, _I * math.exp(2 * a))


def test_d_moves_the_shape_and_not_the_mass():
    d = 0.3
    m1, c1, I1 = perturb_body(_M, _C, _I, 0.0, d, np.zeros(3))
    assert m1 == pytest.approx(_M)
    assert _k(m1, I1) == pytest.approx(_k(_M, _I) * math.exp(d))
    assert np.allclose(c1, _C * math.exp(d))


def test_t_moves_the_com_and_scales_with_alpha():
    """`c' = e^d c + e^-a t`."""
    a, t = -0.28, np.array([0.01, 0.02, -0.03])
    m1, c1, I1 = perturb_body(_M, _C, _I, a, 0.0, t)
    assert np.allclose(c1, _C + math.exp(-a) * t)
    assert _k(m1, I1) == pytest.approx(_k(_M, _I))


def test_the_perturbed_body_is_always_a_body_a_solid_could_have():
    """Positive mass, positive moments, and the triangle inequality."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        a, d = rng.uniform(-0.8, 0.8, 2)
        t = rng.uniform(-0.2, 0.2, 3)
        m1, _, I1 = perturb_body(_M, _C, _I, a, d, t)
        w = np.sort(np.linalg.eigvalsh(I1))
        assert m1 > 0 and w[0] > 0
        assert w[0] + w[1] >= w[2] * (1 - 1e-9)


# ── 3. the correction reproduces the machine it predicts ─────────────────────

def _self_correct(key):
    """Perturb `key`'s twin by `key`'s own measured correction."""
    import copy
    meas = _built(key)
    spec = json.loads((repo_root() / "experiments"
                       / "twin_inertia_mismatch.json").read_text())["pairs"][key]
    leg = _leg(meas)
    m = _model(key)
    groups = body_groups(m)
    before = {g: group_inertia(m, limb_sets(m, g, ids)[0])
              for g, ids in groups.items()}
    rngs, draw = [], {}
    for g, e in spec.items():
        a = 0.5 * math.log(e["mass_ratio"])
        d = _solve_d(m, g, groups[g], math.log(e["k_gyr_ratio"]))
        sh = np.asarray(e["com_shift_per_leg"]) * leg
        rngs.append(GroupRange(g, tuple(groups[g]), (a, a), (d, d), sh, sh))
        draw[g] = (a, d, sh)
    m2 = copy.deepcopy(m)
    apply_numpy(m2, rngs, draw)
    after = {g: group_inertia(m2, limb_sets(m, g, ids)[0])
             for g, ids in groups.items()}
    return spec, before, after, m2, meas


@pytest.mark.parametrize("key", QUADS)
def test_self_correction_lands_on_the_target_group_masses(key):
    spec, before, after, _, _ = _self_correct(key)
    for g in GROUPS:
        got = after[g][0] / before[g][0]
        assert got == pytest.approx(spec[g]["mass_ratio"], rel=1e-6), g


@pytest.mark.parametrize("key", QUADS)
def test_self_correction_lands_on_the_target_radii_of_gyration(key):
    """Within 0.5%: the outboard COM shift slightly spreads the multi-body trunk
    beyond the solved dilation (~0.2% on B2)."""
    spec, before, after, _, _ = _self_correct(key)
    for g in GROUPS:
        got = after[g][2] / before[g][2]
        assert got == pytest.approx(spec[g]["k_gyr_ratio"], rel=5e-3), g


@pytest.mark.parametrize("key", QUADS)
def test_self_correction_lands_on_the_target_total_mass(key):
    """Corrected twin total mass equals the vendor's."""
    _, _, _, m2, meas = _self_correct(key)
    assert mujoco.mj_getTotalmass(m2) == pytest.approx(
        meas["target"]["total_mass_kg"], rel=1e-4)


@pytest.mark.parametrize("key", QUADS)
def test_self_correction_lands_on_the_target_com_positions(key):
    """Group COMs land within 0.5% of leg length of the target."""
    spec, before, after, _, meas = _self_correct(key)
    leg = _leg(meas)
    for g in GROUPS:
        want = before[g][1] + np.asarray(spec[g]["com_shift_per_leg"]) * leg
        assert np.linalg.norm(after[g][1] - want) < 5e-3 * leg, g


# ── the spec the lineup actually runs ────────────────────────────────────────

def test_the_spec_keeps_the_measured_bias():
    """The perturbation ranges keep the measured bias (not centred on 1.0):
    trunk under-charged on B2, hip link within 15%, shank over-charged."""
    spec = json.loads((repo_root() / "experiments"
                       / "twin_inertia_mismatch.json").read_text())
    r = spec["ranges"]
    assert r["trunk"]["mass_ratio"][1] > 1.3, "B2's frame is under-charged"
    assert all(abs(v - 1.0) < 0.15 for v in r["hip_link"]["mass_ratio"]), \
        "the hip links are within 15%"
    assert r["shank"]["mass_ratio"][1] < 0.9, "the shank is over-charged, x0.70-0.83"
    assert r["thigh"]["k_gyr_ratio"][0] > 1.1, "a real thigh is spread further"
    assert spec["population"] == ["go2", "b2"], "quadrupeds only — see the module"


@pytest.mark.parametrize("name", ("cheetah", "bear", "giraffe"))
def test_the_spec_resolves_onto_every_lineup_design(name):
    xml = repo_root() / "generated" / name / "quadruped.xml"
    if not xml.exists():
        pytest.skip(f"{name} not generated — scripts/generate_quadrupeds.py")
    m = mujoco.MjModel.from_xml_path(str(xml))
    knee = m.body_pos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "fl_lower_leg_link")]
    foot = m.site_pos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "fl_foot")]
    leg = float(np.linalg.norm(knee) + np.linalg.norm(foot))
    rs = resolve(m, leg)
    assert {r.group for r in rs} == set(GROUPS)
    for r in rs:
        # the solved dilation must be finite
        assert all(np.isfinite(r.d)), r.group
        assert len(r.body_ids) in (4, 5, 8), (r.group, len(r.body_ids))
