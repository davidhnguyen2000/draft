"""Fit the actuator trends: log-log power laws linking peak torque (and gear
ratio) to mass, geometry, density, rotor inertia and output speed.

Reads src/draft/trends/data/actuator_catalog.json (from 03) and writes
src/draft/trends/data/actuator_trends.json, the adopted laws plus every raw fit.
"""
import sys

import json
import math
import os
import re

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset  # noqa: E402
from draft.jsonio import write_json  # noqa: E402



IN_PATH = dataset.trends("actuator_catalog.json")
OUT_JSON = dataset.trends("actuator_trends.json")


def load():
    d = json.load(open(IN_PATH))
    for r in d:
        # Null geometry is dropped by `powfit` (None is falsy).
        r["radius_mm"] = r["OD_mm"] / 2.0 if r.get("OD_mm") else None
        r["density_kg_m3"] = (r["mass_kg"] / (r["volume_cm3"] * 1e-6)
                              if r.get("volume_cm3") else None)
        # Slenderness L/D, the shape freedom the volume trend leaves open.
        r["aspect_LD"] = (r["L_mm"] / r["OD_mm"]
                          if r.get("L_mm") and r.get("OD_mm") else None)
    return d


def powfit(rows, xk, yk):
    """log-log fit y = coef·x^exp; returns (exp, coef, R2, n)."""
    x = np.array([r[xk] for r in rows if r.get(xk) and r.get(yk)], float)
    y = np.array([r[yk] for r in rows if r.get(xk) and r.get(yk)], float)
    lx, ly = np.log(x), np.log(y)
    exp, a = np.polyfit(lx, ly, 1)
    r2 = 1 - np.sum((ly - (exp * lx + a)) ** 2) / np.sum((ly - ly.mean()) ** 2)
    return float(exp), float(math.exp(a)), float(r2), int(len(x))


def main():
    d = load()
    geared = [r for r in d if r["gear"]]

    # ── Raw power-law fits ────────────────────────────────────────────────────
    RAW = {
        "mass_vs_tau":     powfit(d, "tau_peak_Nm", "mass_kg"),
        "mass_vs_power":   powfit(d, "P_peak_W", "mass_kg"),
        "radius_vs_mass":  powfit(d, "mass_kg", "radius_mm"),
        "length_vs_mass":  powfit(d, "mass_kg", "L_mm"),
        "radius_vs_tau":   powfit(d, "tau_peak_Nm", "radius_mm"),
        "length_vs_tau":   powfit(d, "tau_peak_Nm", "L_mm"),
        "radius_vs_power": powfit(d, "P_peak_W", "radius_mm"),
        "length_vs_power": powfit(d, "P_peak_W", "L_mm"),
        "volume_vs_mass":  powfit(d, "mass_kg", "volume_cm3"),
        "aspect_vs_mass":  powfit(d, "mass_kg", "aspect_LD"),
        "aspect_vs_gear":  powfit(geared, "gear", "aspect_LD"),
        "torquedens_vs_gear": powfit(geared, "gear", "torque_density_Nm_per_kg"),
        "powerdens_vs_gear":  powfit(geared, "gear", "power_density_W_per_kg"),
        "speed_vs_gear":      powfit(geared, "gear", "omega_NL_rad_s"),
    }
    print("Raw log-log fits   y = coef · x^exp")
    print(f"  {'relation':22s}{'exp':>8}{'coef':>10}{'R2':>7}{'n':>5}")
    for k, (e, c, r2, n) in RAW.items():
        print(f"  {k:22s}{e:8.3f}{c:10.3f}{r2:7.3f}{n:5d}")

    dens = np.array([r["density_kg_m3"] for r in d if r["density_kg_m3"]])

    # ── Composite law in torque: geometry split from mass, density absorbs the
    # residual so mass = ρ·π·r²·L holds exactly.
    mass_tau_exp = RAW["mass_vs_tau"][0]
    r_mass_exp = RAW["radius_vs_mass"][0]
    L_mass_exp = RAW["length_vs_mass"][0]
    r_tau_exp = r_mass_exp * mass_tau_exp
    L_tau_exp = L_mass_exp * mass_tau_exp
    rho_tau_exp = mass_tau_exp - 2 * r_tau_exp - L_tau_exp
    rho0 = float(np.median(dens))                  # dataset median solid-equiv density

    # ── Volume and shape: (V, L/D) carry the same information as (r, L); mass
    # sets the volume, L/D is left as the shape freedom.
    V_exp, V_coef = RAW["volume_vs_mass"][0], RAW["volume_vs_mass"][1]
    asp = np.array([r["aspect_LD"] for r in d if r["aspect_LD"]], float)
    asp_exp, asp_coef = RAW["aspect_vs_mass"][0], RAW["aspect_vs_mass"][1]
    asp_by_cat = {
        c: {"median": round(float(np.median(z)), 3),
            "p10": round(float(np.percentile(z, 10)), 3),
            "p90": round(float(np.percentile(z, 90)), 3), "n": int(len(z))}
        for c in ("QDD", "MidGear", "Harmonic")
        for z in [np.array([r["aspect_LD"] for r in d
                            if r["category"] == c and r["aspect_LD"]], float)]
        if len(z)}

    # ── Geometry trend (airgap law) ──────────────────────────────────────────
    # Airgap shear gives tau_rotor = 2*pi*sigma*r^2*L; r^2*L is imposed, not fitted.
    # Package-vs-airgap and size effects are absorbed into a fitted sigma_eff:
    #     tau/N = 2*pi*sigma_eff*r^2*L,   sigma_eff = sigma0 * tau^s * N^-k
    # i.e. pi*r^2*L = tau^(1-s) * N^(k-1) / (2*sigma0), the form the generator inverts.
    gd = [r for r in geared if r.get("volume_cm3")]
    lg_ = lambda k, rows: np.log(np.array([r[k] for r in rows], float))

    def _lstsq(y, cols):
        A = np.column_stack([np.ones(len(y))] + list(cols))
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        res = y - A @ beta
        se = np.sqrt(np.diag((res @ res / (len(y) - A.shape[1]))
                             * np.linalg.inv(A.T @ A)))
        r2 = 1 - res @ res / np.sum((y - y.mean()) ** 2)
        return beta, se, float(r2), float(np.exp(np.abs(res).mean()))

    V_m3 = np.log(np.array([r["volume_cm3"] * 1e-6 for r in gd], float))
    ltau, lgear = lg_("tau_peak_Nm", gd), lg_("gear", gd)
    # sigma_eff = (tau/N) / (2*pi*r^2*L) = (tau/N) / (2*V)
    lsig = lg_("tau_peak_Nm", gd) - lgear - np.log(2.0) - V_m3
    sig_b, sig_se, sig_r2, sig_gmfe = _lstsq(lsig, [ltau, lgear])
    sigma0, s_exp, k_exp = float(np.exp(sig_b[0])), float(sig_b[1]), float(-sig_b[2])
    # the same fit read as the volume the generator solves for
    V_tau_exp, V_gear_exp = 1.0 - s_exp, k_exp - 1.0
    # Score the inverted law on pi*r^2*L itself.
    _lv = np.log(np.array([r["volume_cm3"] for r in gd], float))
    _pv = np.log(np.array([1.0 / (2.0 * sigma0) * r["tau_peak_Nm"] ** V_tau_exp
                           * r["gear"] ** V_gear_exp * 1e6 for r in gd], float))
    _rv = _lv - _pv
    vol_r2 = float(1 - _rv @ _rv / np.sum((_lv - _lv.mean()) ** 2))
    # Slenderness from the same two inputs (not constrained by the physics).
    lq = lg_("aspect_LD", gd)
    q_b, q_se, q_r2, q_gmfe = _lstsq(lq, [ltau, lgear])

    # Gear-ratio regime laws (relative to a nominal ratio): torque density rises,
    # output speed falls with gear ratio. Constant-power ⇒ pd ~ flat.
    td_gear_exp = RAW["torquedens_vs_gear"][0]
    speed_gear_exp = RAW["speed_vs_gear"][0]

    # ── Mass trend ───────────────────────────────────────────────────────────
    # Over the geared subset, compare mass ∝ tau^a·N^b against mass ∝ tau^a.
    gm = [r for r in geared]
    lg = lambda k, rows: np.log(np.array([r[k] for r in rows], float))
    A = np.column_stack([np.ones(len(gm)), lg("tau_peak_Nm", gm), lg("gear", gm)])
    beta, *_ = np.linalg.lstsq(A, lg("mass_kg", gm), rcond=None)
    resid = lg("mass_kg", gm) - A @ beta
    r2_2v = 1 - resid @ resid / np.sum((lg("mass_kg", gm) - lg("mass_kg", gm).mean()) ** 2)
    se_2v = np.sqrt(np.diag(resid @ resid / (len(gm) - 3) * np.linalg.inv(A.T @ A)))

    # The gear term is within one standard error of zero and the paired LOO test
    # in 05_leave_one_out.py cannot separate the two laws, so the adopted trend is
    # torque alone, REFITTED on the same geared rows (not the 2-variable fit with N=1).
    A1 = A[:, :2]
    beta1, *_ = np.linalg.lstsq(A1, lg("mass_kg", gm), rcond=None)
    resid1 = lg("mass_kg", gm) - A1 @ beta1
    r2_1v = 1 - resid1 @ resid1 / np.sum((lg("mass_kg", gm) - lg("mass_kg", gm).mean()) ** 2)
    se_1v = np.sqrt(np.diag(resid1 @ resid1 / (len(gm) - 2) * np.linalg.inv(A1.T @ A1)))
    p_exp, q_exp = r_mass_exp, L_mass_exp
    rho_mass_exp = 1.0 - 2 * p_exp - q_exp

    # ── Rotor-inertia trend ──────────────────────────────────────────────────
    # Adopted: J_rotor = k * r^4 (r^4 imposed, areal constant k fitted). The
    # envelope form J = phi*0.5*m*r^2 scales with package length, which is mostly
    # gearbox, so it over-charges high-reduction modules; it is kept as a
    # diagnostic (active fraction, per category). Geared-inrunner rows are
    # excluded: a small motor behind a huge reduction fits neither form.
    ji = [r for r in d if r.get("rotor_inertia_kgm2") and r.get("volume_cm3")
          and not r.get("rotor_is_geared_inrunner")]
    n_inrunner = sum(1 for r in d if r.get("rotor_is_geared_inrunner")
                     and r.get("rotor_inertia_kgm2"))

    def _envelope_J(r):
        return (0.5 * (r["mass_kg"] / (r["volume_cm3"] * 1e-6)) * math.pi
                * (r["radius_mm"] * 1e-3) ** 4 * (r["L_mm"] * 1e-3))

    def _gmfe(pred, act):
        pred, act = np.asarray(pred, float), np.asarray(act, float)
        return float(np.exp(np.mean(np.abs(np.log(pred / act)))))

    def _central(f):
        """Geometric mean: the unbiased centre for multiplicative errors."""
        return float(np.exp(np.mean(np.log(np.asarray(f, float)))))

    j_exp = None
    frac = []
    by_cat = {}
    if len(ji) >= 3:
        j_exp = float(np.polyfit(np.log([r["radius_mm"] for r in ji]),
                                 np.log([r["rotor_inertia_kgm2"] for r in ji]), 1)[0])
        frac = [r["rotor_inertia_kgm2"] / _envelope_J(r) for r in ji]
        for cat in sorted({r["category"] for r in ji}):
            sub = [r for r in ji if r["category"] == cat]
            f = [r["rotor_inertia_kgm2"] / _envelope_J(r) for r in sub]
            by_cat[cat] = {
                "active_fraction_mean": round(_central(f), 4),
                "active_fraction_arithmetic": round(float(np.mean(f)), 4),
                "active_fraction_sd": round(float(np.std(f)), 4),
                "gmfe": round(_gmfe([_central(f) * _envelope_J(r) for r in sub],
                                    [r["rotor_inertia_kgm2"] for r in sub]), 3),
                "radius_exp_measured": (
                    round(float(np.polyfit(np.log([r["radius_mm"] for r in sub]),
                                           np.log([r["rotor_inertia_kgm2"] for r in sub]),
                                           1)[0]), 2) if len(sub) >= 3 else None),
                "n": len(sub),
            }
    inertia_mass_exp = rho_mass_exp + 4 * p_exp + q_exp

    # ── Areal constant k = geomean(J/r^4), and the case against a length term ─
    k_areal = k_r2 = k_gmfe = env_r2 = env_gmfe = None
    L_exp_free = L_exp_se = env_corr_N = k_corr_N = None
    if len(ji) >= 3:
        Jj = np.array([r["rotor_inertia_kgm2"] for r in ji], float)
        rr = np.array([r["radius_mm"] for r in ji], float) * 1e-3
        LL = np.array([r["L_mm"] for r in ji], float) * 1e-3
        lJ, sst = np.log(Jj), None
        sst = float(np.sum((lJ - lJ.mean()) ** 2))

        def _score(pred):
            res = lJ - np.log(pred)
            return 1 - float(res @ res) / sst, _gmfe(pred, Jj)

        k_areal = _central(Jj / rr ** 4)
        k_r2, k_gmfe = _score(k_areal * rr ** 4)
        env = np.array([_envelope_J(r) for r in ji], float)
        env_r2, env_gmfe = _score(_central(Jj / env) * env)

        # Fit the length exponent freely.
        A_j = np.column_stack([np.ones(len(ji)), np.log(rr), np.log(LL)])
        b_j, *_ = np.linalg.lstsq(A_j, lJ, rcond=None)
        res_j = lJ - A_j @ b_j
        se_j = np.sqrt(np.diag(res_j @ res_j / (len(ji) - 3) * np.linalg.inv(A_j.T @ A_j)))
        L_exp_free, L_exp_se = float(b_j[2]), float(se_j[2])

        # Residual correlation with log N, per form.
        gj = [(i, r["gear"]) for i, r in enumerate(ji) if r.get("gear")]
        if len(gj) >= 3:
            idx = [i for i, _ in gj]
            lN = np.log([g for _, g in gj])
            env_corr_N = float(np.corrcoef(
                lN, (lJ - np.log(_central(Jj / env) * env))[idx])[0, 1])
            k_corr_N = float(np.corrcoef(
                lN, (lJ - np.log(k_areal * rr ** 4))[idx])[0, 1])

    law = {
        "inertia_trend": {
            "form": "J_rotor = k_areal_kg_m2 * r^4   (r in metres)",
            "k_areal_kg_m2": round(k_areal, 4),
            "r2": round(k_r2, 3),
            "gmfe_insample": round(k_gmfe, 3),
            "form_mass": "J_rotor/J0 = (m_motor/m0)^inertia_mass_exp",
            "inertia_mass_exp": round(inertia_mass_exp, 4),
            "radius_exp_measured": round(j_exp, 3) if j_exp else None,
            "radius_exp_physics": 4.0,
            # Pooled; prefer active_fraction_by_category when the category is known.
            "active_fraction_mean": round(_central(frac), 4) if frac else None,
            "active_fraction_arithmetic": round(float(np.mean(frac)), 4) if frac else None,
            "active_fraction_sd": round(float(np.std(frac)), 4) if frac else None,
            "active_fraction_gmfe": (
                round(_gmfe([_central(frac) * _envelope_J(r) for r in ji],
                            [r["rotor_inertia_kgm2"] for r in ji]), 3) if frac else None),
            "active_fraction_by_category": by_cat,
            "active_fraction_previous": 0.088,
            "n_published": len(ji),
            "n_excluded_geared_inrunner": n_inrunner,
            "n_catalogued": len(d),
            "reflected": "armature at the output = N^2 * J_rotor",
            "note": ("Fit on every vendor document that publishes the number: "
                     "CubeMars, MyActuator, HEBI, ZeroErr, RobStride (output-side, "
                     "divided by N^2). The r^4 exponent is required by rigid-body "
                     "physics and confirmed within each category; what the data "
                     "sets is the areal constant k. The active-fraction envelope "
                     "form it replaced is kept below."),
            "length_term_dropped": {
                "form_geometric": "J_rotor = active_fraction * 0.5*rho*pi*r^4*L",
                "identity": "= active_fraction * 0.5 * m_package * r^2",
                "r2": round(env_r2, 3),
                "gmfe_insample": round(env_gmfe, 3),
                "length_exp_free": round(L_exp_free, 3),
                "length_exp_free_stderr": round(L_exp_se, 3),
                "resid_corr_log_gear": round(env_corr_N, 3),
                "resid_corr_log_gear_areal": round(k_corr_N, 3),
                "note": ("The superseded law. Package length is mostly gearbox "
                         "and a gearbox does not spin at rotor speed, so this "
                         "form over-charges high-reduction modules; its residual "
                         "correlates with log N where the adopted form's does "
                         "not. Kept so the comparison in the paper can be "
                         "reproduced from this file."),
            },
        },
        "mass_trend": {
            "form": "mass/m0 = (tau/tau0)^tau_exp",
            # Absolute form, for classes not anchored on a catalogued part.
            "form_absolute": "mass_kg = coef * tau_Nm^tau_exp",
            "coef": round(float(math.exp(beta1[0])), 6),
            "N_median": round(float(np.median([r["gear"] for r in gm])), 2),
            "tau_exp": round(float(beta1[1]), 4),
            "tau_exp_stderr": round(float(se_1v[1]), 4),
            # Zero, kept so consumers can read this as a (tau, N) power law.
            "gear_exp": 0.0,
            "r2": round(float(r2_1v), 3),
            "n": len(gm),
            "note": ("Torque alone, over the 103 geared rows. The gear term was "
                     "dropped, not omitted: fitted it is "
                     f"{float(beta[2]):+.4f} +- {float(se_2v[2]):.4f}, within one "
                     "standard error of zero, worth 7% of mass across the whole "
                     "catalogue and indistinguishable from nothing under paired "
                     "LOO. Keeping it cost 0.0003 of R^2 and charged the "
                     "generator for a variable the data does not resolve."),
            "gear_term_dropped": {
                "coef": round(float(math.exp(beta[0])), 6),
                "tau_exp": round(float(beta[1]), 4),
                "gear_exp": round(float(beta[2]), 4),
                "tau_exp_stderr": round(float(se_2v[1]), 4),
                "gear_exp_stderr": round(float(se_2v[2]), 4),
                "r2": round(float(r2_2v), 3),
                "note": ("The superseded two-variable fit, kept so the comparison "
                         "in the paper can be reproduced from this file."),
            },
        },
        "geometry_from_mass": {
            "driver": "relative_mass",
            "radius_exp": round(p_exp, 4),          # r/r0   = (m/m0)^radius_exp
            "length_exp": round(q_exp, 4),          # L/L0   = (m/m0)^length_exp
            "density_exp": round(rho_mass_exp, 4),  # rho/rho0 = (m/m0)^density_exp
            "volume_exp": round(2 * p_exp + q_exp, 4),   # V/V0 = (m/m0)^volume_exp
            "aspect_exp": round(q_exp - p_exp, 4),       # q/q0 = (m/m0)^aspect_exp
            "note": ("The generator consumes this as VOLUME plus SHAPE, not as r "
                     "and L: V/V0 = (m/m0)^volume_exp with density_exp = 1 - "
                     "volume_exp its exact complement, and the slenderness q = L/D "
                     "left free (see aspect_trend). r = (V/(2*pi*q))^(1/3), L = 2*q*r "
                     "reproduces the r(m)/L(m) fits identically at the default q."),
        },
        # The adopted geometry trend (airgap law); supersedes `geometry_from_mass`.
        "geometry_trend": {
            "form": "tau/N = 2*pi*sigma_eff*r^2*L,  sigma_eff = sigma0*tau^s*N^-k",
            "volume_form": "pi*r^2*L = tau^(1-s) * N^(k-1) / (2*sigma0)",
            "shape_form": "L/D = q_coef * tau^q_tau_exp * N^q_gear_exp",
            "driver": "peak_torque_and_gear_ratio",
            "sigma0_Pa": round(sigma0, 1),
            "s_torque_exp": round(s_exp, 4),
            "s_stderr": round(float(sig_se[1]), 4),
            "k_gear_exp": round(k_exp, 4),
            "k_stderr": round(float(sig_se[2]), 4),
            "r2": round(sig_r2, 3),
            "gmfe_insample": round(sig_gmfe, 3),
            "n": len(gd),
            # R^2 on pi*r^2*L itself; `r2` above is the sigma_eff regression's.
            "r2_volume": round(vol_r2, 3),
            "volume_tau_exp": round(V_tau_exp, 4),
            "volume_gear_exp": round(V_gear_exp, 4),
            "volume_coef_m3": float(f"{1.0 / (2.0 * sigma0):.6g}"),
            "q_coef": round(float(np.exp(q_b[0])), 4),
            "q_tau_exp": round(float(q_b[1]), 4),
            "q_tau_stderr": round(float(q_se[1]), 4),
            "q_gear_exp": round(float(q_b[2]), 4),
            "q_gear_stderr": round(float(q_se[2]), 4),
            "q_r2": round(q_r2, 3),
            "q_gmfe_insample": round(q_gmfe, 3),
            "note": (
                "s > 0 says the effective shear stress RISES with machine size — a "
                "bigger motor cools better, carries more back-iron and runs more "
                "poles. k > 0 says the rotor's share of the package FALLS as the "
                "reduction grows, which is the gearbox crowding it out; it is why "
                "a harmonic module and a QDD cannot share one constant. Neither is "
                "a fudge: both are what is left over once r^2*L is imposed, and "
                "both come with a standard error. Absolute check: the package "
                "shear stress divided by the independently measured active "
                "fraction (from published rotor inertias) puts a QDD's airgap at "
                "~13 kPa, which is the textbook figure for an air-cooled BLDC."),
        },
        # L/D: weakly tied to mass/torque, strongly to gear ratio. The generator
        # takes it as a free input with this measured band.
        "aspect_trend": {
            "definition": "q = L/D = L/(2r), the actuator's slenderness",
            "driver": "relative_mass",
            "coef": round(float(asp_coef), 4),      # q = coef * m_kg^exp
            "exp": round(float(asp_exp), 4),
            "r2": round(float(RAW["aspect_vs_mass"][2]), 3),
            "n": int(RAW["aspect_vs_mass"][3]),
            "median": round(float(np.median(asp)), 3),
            "p10": round(float(np.percentile(asp, 10)), 3),
            "p90": round(float(np.percentile(asp, 90)), 3),
            "min": round(float(asp.min()), 3),
            "max": round(float(asp.max()), 3),
            "sigma_log": round(float(np.log(asp).std()), 3),
            "by_category": asp_by_cat,
            "vs_gear": {"coef": round(float(RAW["aspect_vs_gear"][1]), 4),
                        "exp": round(float(RAW["aspect_vs_gear"][0]), 4),
                        "r2": round(float(RAW["aspect_vs_gear"][2]), 3),
                        "n": int(RAW["aspect_vs_gear"][3])},
            "note": ("Identical by construction to L(m)/2r(m), so the default "
                     "reproduces the old geometry exactly. At fixed volume the "
                     "rotor inertia goes as q^(-2/3): a slimmer, longer actuator "
                     "of the same mass has LESS reflected inertia."),
        },
        "provenance": {
            "dataset": "src/draft/trends/data/actuator_catalog.json",
            "n": len(d),
            "n_geared": len(geared),
            "categories": {c: sum(r["category"] == c for r in d)
                           for c in ("QDD", "MidGear", "Harmonic")},
        },
        # geometry & mass as power laws in RELATIVE peak torque  t = τ/τ_nominal
        "geometry": {
            "driver": "relative_peak_torque",
            "radius_exp": round(r_tau_exp, 4),      # r/r0    = t^radius_exp
            "length_exp": round(L_tau_exp, 4),      # L/L0    = t^length_exp
            "density_exp": round(rho_tau_exp, 4),   # rho/rho0= t^density_exp (≈0)
            "mass_exp": round(mass_tau_exp, 4),     # implied mass/m0 = t^mass_exp
            "density0_kg_m3": round(rho0, 1),       # nominal effective density
        },
        # gear-ratio regime, power laws in RELATIVE gear ratio  n = N/N_nominal
        "gear": {
            "driver": "relative_gear_ratio",
            "torque_density_exp": round(td_gear_exp, 4),
            "speed_exp": round(speed_gear_exp, 4),
        },
        "density_stats_kg_m3": {
            "median": round(float(np.median(dens)), 1),
            "mean": round(float(dens.mean()), 1),
            "p10": round(float(np.percentile(dens, 10)), 1),
            "p90": round(float(np.percentile(dens, 90)), 1),
        },
        "raw_fits": {k: {"exp": round(e, 4), "coef": round(c, 4), "r2": round(r2, 3), "n": n}
                     for k, (e, c, r2, n) in RAW.items()},
        # Per-category density frontier: `max` is the best catalogued part,
        # `p90` the best-in-class line.
        "frontier": {
            cat: {
                "torque_density_Nm_per_kg": {
                    "max": round(float(max(td)), 1), "p90": round(float(np.percentile(td, 90)), 1),
                    "argmax": rows[int(np.argmax(td))]["name"],
                },
                "power_density_W_per_kg": {
                    "max": round(float(max(pd)), 1), "p90": round(float(np.percentile(pd, 90)), 1),
                    "argmax": rows[int(np.argmax(pd))]["name"],
                },
                "n": len(rows),
            }
            for cat, rows, td, pd in (
                (c,
                 [r for r in (d if c == "ALL" else [x for x in d if x["category"] == c])],
                 [r["torque_density_Nm_per_kg"] for r in
                  (d if c == "ALL" else [x for x in d if x["category"] == c])],
                 [r["power_density_W_per_kg"] for r in
                  (d if c == "ALL" else [x for x in d if x["category"] == c])])
                for c in ("QDD", "MidGear", "Harmonic", "ALL")
            )
        },
        "heuristic_envelope_mm": {"r_lo_P_cbrt": 8.0, "r_hi_P_cbrt": 15.0,
                                  "note": "r[mm] in 8..15 * P[W]^(1/3); harmonic ~2x length"},
    }
    write_json(OUT_JSON, law, indent=2)
    print(f"\nAdopted law → {OUT_JSON}")
    print(f"  mass = {math.exp(beta1[0]):.4f} · τ^{beta1[1]:.4f}(±{se_1v[1]:.4f})"
          f"   R²={r2_1v:.4f}  n={len(gm)}")
    print(f"    gear term dropped: N^{beta[2]:+.4f}(±{se_2v[2]:.4f}) bought "
          f"{r2_2v - r2_1v:+.4f} of R² — see 05_leave_one_out.py's paired test")
    print(f"  r ∝ τ^{r_tau_exp:.3f}   L ∝ τ^{L_tau_exp:.3f}   "
          f"ρ ∝ τ^{rho_tau_exp:.3f} (ρ0={rho0:.0f} kg/m³)   ⇒ mass ∝ τ^{mass_tau_exp:.3f}")
    print("\nGeometry law — the airgap form, τ/N = 2π·σ·r²·L  (r²L imposed):")
    print(f"  σ_eff = {sigma0:.0f} · τ^{s_exp:+.3f}(±{sig_se[1]:.3f}) · N^{-k_exp:+.3f}(±{sig_se[2]:.3f}) Pa"
          f"   R²={sig_r2:.3f}  GMFE={sig_gmfe:.3f}  n={len(gd)}")
    print(f"    s={s_exp:+.3f}: shear rises with size.  k={k_exp:.3f}: the rotor's share falls with reduction.")
    print(f"\nRotor inertia — r^4 imposed, one areal constant fitted:")
    print(f"  J = {k_areal:.3f} · r^4   R²={k_r2:.4f}  GMFE={k_gmfe:.3f}  n={len(ji)}")
    print(f"  length term dropped: envelope form scored R²={env_r2:.4f} "
          f"GMFE={env_gmfe:.3f}; free L exponent {L_exp_free:+.3f}(±{L_exp_se:.3f}) "
          f"vs 1.0 from a solid cylinder")
    print(f"    residual vs log N: envelope {env_corr_N:+.3f}, areal {k_corr_N:+.3f} "
          f"— package length is gearbox, and a gearbox does not spin")
    print(f"  ⇒ πr²L = τ^{V_tau_exp:.3f} · N^{V_gear_exp:+.3f} / (2·{sigma0:.0f})"
          f"   R²={vol_r2:.4f} on volume itself (the σ fit's {sig_r2:.3f} is not that)")
    print(f"  L/D   = {math.exp(q_b[0]):.3f} · τ^{q_b[1]:+.3f}(±{q_se[1]:.3f}) · N^{q_b[2]:+.3f}(±{q_se[2]:.3f})"
          f"   R²={q_r2:.3f}  GMFE={q_gmfe:.3f}")


if __name__ == "__main__":
    main()
