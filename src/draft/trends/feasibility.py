"""Fitted trends and the feasibility gate (paper §IV).

``MotorTrends`` and ``LinkTrends`` are regressions over real hardware, read from
``data/actuator_trends.json`` and ``data/link_trends.json``. The generator uses
them to size a design, then checks the result; violations raise
``FeasibilityViolation`` and everything is written to ``feasibility_report.yaml``.

  input    two of (peak torque, no-load speed, gear ratio N, peak power),
           link lengths
  derived  the rest of that quadruple, motor radius, length, density,
           armature, and every segment density the population measures
  checked  torque/power density, radius envelope, speed vs the speed trend,
           link density band, packing, actuator and trunk mass fractions,
           mass vs size

A class that names a catalogued part is sized relative to that part, so the
trend only prices the deviation from it. Stdlib only.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Optional

#: Package data; written only by the stages under ``datasets/``.
_DATA = Path(__file__).resolve().parent / "data"
ACTUATOR_TRENDS_JSON = _DATA / "actuator_trends.json"
LINK_TRENDS_JSON = _DATA / "link_trends.json"
CATALOG_JSON = _DATA / "actuator_catalog.json"


class FeasibilityViolation(ValueError):
    """Raised when a design is infeasible under the fitted trends; carries every violation."""

    def __init__(self, messages: list[str]):
        self.messages = messages
        super().__init__("\n".join(["design violates the fitted physical laws:"]
                                   + [f"  - {m}" for m in messages]))


def _load(path: Path) -> dict:
    with Path(path).open() as f:
        return json.load(f)


#: Parameters set by a dataset calibration stage rather than chosen freely,
#: mapped to a description. Shown as calibrated in the editor. Currently empty.
CALIBRATED_PARAMS: dict[str, str] = {
}


# ── Motor laws ────────────────────────────────────────────────────────────────

def parse_gear_ratio(text: str) -> Optional[float]:
    """Reduction ratio from ``"9:1"`` text or a CubeMars-style name (``AK80-64``), else None.

    Fallback only: prefer a row's ``gear`` field, since ``notes`` may mention a
    sibling part's ratio.
    """
    if not text:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)\s*:\s*1", text)
    if m:
        return float(m.group(1))
    m = re.search(r"[A-Za-z]+\d+-(\d+(?:\.\d+)?)", text)
    return float(m.group(1)) if m else None


class MotorTrends:
    """Sizes and vets an actuator class from (peak torque, gear ratio).

    ``size()`` returns mass, radius, length, density and reflected inertia, with
    density solved so ``ρ·π·r²·L`` equals the trend's mass exactly.
    """

    def __init__(self,
                 laws_path: Path | None = None,
                 catalog_path: Path | None = None):
        laws_path = laws_path or ACTUATOR_TRENDS_JSON
        catalog_path = catalog_path or CATALOG_JSON
        laws = _load(laws_path)
        self._laws = laws
        ml, gm, il = laws["mass_trend"], laws["geometry_from_mass"], laws["inertia_trend"]
        raw = laws["raw_fits"]

        # mass_kg = coef · τ^tau_exp. No reduction term (gear_exp is 0.0 if
        # absent); N costs speed and reflected inertia, not mass.
        self.mass_coef = ml["coef"]
        self.tau_exp = ml["tau_exp"]
        self.gear_exp = ml.get("gear_exp", 0.0)
        self.gear_median = ml["N_median"]

        # r(m), L(m) fits: used only by the fallback below when the file has no
        # geometry_trend.
        self.radius_exp = gm["radius_exp"]
        self.length_exp = gm["length_exp"]
        self.radius_coef_mm = raw["radius_vs_mass"]["coef"]
        self.length_coef_mm = raw["length_vs_mass"]["coef"]
        # Geometry from the airgap relation tau/N = 2*pi*sigma*r^2*L (imposed by
        # physics); the catalogue fits sigma's size dependence (exponents s, k):
        #     pi*r^2*L = tau^(1-s) * N^(k-1) / (2*sigma0)
        gl = laws.get("geometry_trend")
        if gl:
            self.sigma0_Pa = gl["sigma0_Pa"]
            self.shear_tau_exp = gl["s_torque_exp"]
            self.shear_gear_exp = gl["k_gear_exp"]
            # Derived from (sigma0, s, k) so the airgap identity holds exactly.
            self.volume_coef_m3 = 1.0 / (2.0 * self.sigma0_Pa)
            self.volume_tau_exp = 1.0 - self.shear_tau_exp
            self.volume_gear_exp = self.shear_gear_exp - 1.0
            self.aspect_coef = gl["q_coef"]
            self.aspect_tau_exp = gl["q_tau_exp"]
            self.aspect_gear_exp = gl["q_gear_exp"]
        else:
            # No geometry_trend: split geometry off the mass trend instead.
            v_exp = 2 * self.radius_exp + self.length_exp
            q_exp = self.length_exp - self.radius_exp
            m_c, m_t, m_g = ml["coef"], ml["tau_exp"], self.gear_exp
            self.sigma0_Pa = None
            self.shear_tau_exp = self.shear_gear_exp = None
            self.volume_coef_m3 = (math.pi * self.radius_coef_mm ** 2
                                   * self.length_coef_mm * 1e-9) * m_c ** v_exp
            self.volume_tau_exp, self.volume_gear_exp = m_t * v_exp, m_g * v_exp
            self.aspect_coef = (self.length_coef_mm / (2.0 * self.radius_coef_mm)
                                * m_c ** q_exp)
            self.aspect_tau_exp, self.aspect_gear_exp = m_t * q_exp, m_g * q_exp

        # Catalogued L/D band, reported by check().
        al = laws.get("aspect_trend") or {}
        self.aspect_band = (al.get("p10"), al.get("p90"))
        self.aspect_range = (al.get("min"), al.get("max"))

        # J_rotor = k · r⁴: r⁴ from rigid-body physics, k (areal density) fitted.
        # active_fraction is diagnostic only.
        self.inertia_coef = il["k_areal_kg_m2"]
        self.active_fraction = il["active_fraction_mean"]

        # ω_NL = coef · N^exp: checks a declared speed, or solves one (motor_solve).
        self.speed_coef = raw["speed_vs_gear"]["coef"]
        self.speed_exp = raw["speed_vs_gear"]["exp"]

        env = laws["heuristic_envelope_mm"]
        self.r_lo_k, self.r_hi_k = env["r_lo_P_cbrt"], env["r_hi_P_cbrt"]
        self.frontier = laws["frontier"]

        self._catalog_path = Path(catalog_path)
        self._catalog: Optional[list[dict]] = None

    # ── catalogue ─────────────────────────────────────────────────────────────

    @property
    def catalog(self) -> list[dict]:
        if self._catalog is None:
            self._catalog = (_load(self._catalog_path)
                             if self._catalog_path.exists() else [])
        return self._catalog

    @staticmethod
    def _norm(name: str) -> str:
        return name.lower().replace(" ", "_").replace("-", "_")

    def find_part(self, name: str) -> Optional[dict]:
        """Catalogue entry by case/dash/space-insensitive name, or None."""
        want = self._norm(name)
        return next((e for e in self.catalog if self._norm(e["name"]) == want), None)

    def anchor(self, name: str) -> Optional[dict]:
        """A catalogued part's measured envelope, the reference for relative sizing.

        None if the part has no published dimensions (the absolute fit is used).
        """
        e = self.find_part(name)
        if e is None or not e.get("has_dims"):
            return None
        r = e["OD_mm"] / 2000.0
        L = e["L_mm"] / 1000.0
        gear = (e.get("gear")
                or parse_gear_ratio(e.get("notes", ""))
                or parse_gear_ratio(e["name"]))
        return {"name": e["name"], "category": e["category"], "tau": e["tau_peak_Nm"],
                "mass": e["mass_kg"], "r": r, "L": L, "gear": gear,
                "omega": e["omega_NL_rad_s"]}

    # ── sizing ────────────────────────────────────────────────────────────────

    def gear_from_speed(self, omega_rad_s: float) -> float:
        """Reduction implied by a no-load speed, inverting ``ω = coef·N^exp``.

        Used when a class declares neither a reduction nor a catalogue part.
        """
        return (omega_rad_s / self.speed_coef) ** (1.0 / self.speed_exp)

    def mass(self, tau_Nm: float, gear: float, anchor: Optional[dict] = None) -> float:
        """Actuator mass (kg) for a peak torque and reduction."""
        if anchor and anchor.get("gear"):
            return (anchor["mass"]
                    * (tau_Nm / anchor["tau"]) ** self.tau_exp
                    * (gear / anchor["gear"]) ** self.gear_exp)
        return self.mass_coef * tau_Nm ** self.tau_exp * gear ** self.gear_exp

    def shear_stress(self, tau_Nm: float, gear: float) -> Optional[float]:
        """Effective shear stress (Pa) over the whole package, ``(tau/N) / (2*pi*r^2*L)``.

        Well below a true airgap value, since most of the package is not airgap.
        """
        if self.sigma0_Pa is None:
            return None
        return (self.sigma0_Pa * tau_Nm ** self.shear_tau_exp
                * gear ** -self.shear_gear_exp)

    def _law_ratio(self, coef: float, t_exp: float, g_exp: float,
                   tau_Nm: float, gear: float, anchor: Optional[dict]) -> float:
        """A (tau, N) power law, relative to an anchor part when there is one."""
        val = coef * tau_Nm ** t_exp * gear ** g_exp
        if anchor and anchor.get("gear"):
            ref = coef * anchor["tau"] ** t_exp * anchor["gear"] ** g_exp
            return val / ref
        return val

    def aspect_default(self, tau_Nm: float, gear: float,
                       anchor: Optional[dict] = None) -> float:
        """Slenderness ``q = L/D`` the catalogue fit expects at this (tau, N).

        The airgap relation fixes only ``r^2*L``; this fit supplies the split.
        """
        rel = self._law_ratio(self.aspect_coef, self.aspect_tau_exp,
                              self.aspect_gear_exp, tau_Nm, gear, anchor)
        return rel * (anchor["L"] / (2.0 * anchor["r"])) if anchor else rel

    def volume(self, tau_Nm: float, gear: float,
               anchor: Optional[dict] = None) -> float:
        """Package volume (m^3), inverted from the airgap relation."""
        rel = self._law_ratio(self.volume_coef_m3, self.volume_tau_exp,
                              self.volume_gear_exp, tau_Nm, gear, anchor)
        return rel * (math.pi * anchor["r"] ** 2 * anchor["L"]) if anchor else rel

    def size(self, tau_Nm: float, gear: float,
             anchor: Optional[dict] = None,
             aspect: Optional[float] = None) -> dict:
        """Full envelope for one actuator class.

        Mass from the mass trend, volume from the airgap relation, ``rho = m/V``,
        rotor and reflected inertia (MuJoCo ``armature``), and the speed-trend
        no-load speed. ``aspect`` (L/D) defaults to the catalogue fit; changing it
        moves r (armature goes as ``q^(-2/3)``) but not mass.
        """
        m = self.mass(tau_Nm, gear, anchor)
        V = self.volume(tau_Nm, gear, anchor)
        q_pop = self.aspect_default(tau_Nm, gear, anchor)
        q = float(aspect) if aspect else q_pop
        if q <= 0:
            raise ValueError(f"aspect ratio L/D must be positive, got {aspect!r}")
        # V = pi r^2 L with L = 2 q r  =>  V = 2 pi q r^3
        r = (V / (2.0 * math.pi * q)) ** (1.0 / 3.0)
        L = 2.0 * q * r
        # Not fitted: rho makes the cylinder weigh exactly the trend's mass.
        rho = m / (math.pi * r * r * L)
        j_rotor = self.inertia_coef * r ** 4
        j_source = "areal(k*r^4)"
        return {
            "mass": m, "r": r, "L": L, "rho": rho,
            "volume": V,
            "shear_Pa": self.shear_stress(tau_Nm, gear),
            "aspect": q,
            "aspect_population": q_pop,
            "aspect_source": "declared" if aspect else "population_fit",
            "J_rotor": j_rotor,
            "J_rotor_source": j_source,
            "armature": gear ** 2 * j_rotor,   # reflected to the output
            "omega_law": self.speed_coef * gear ** self.speed_exp,
            "anchor": anchor["name"] if anchor else None,
            "category": (anchor or {}).get("category"),
        }

    # ── validation ────────────────────────────────────────────────────────────

    def check(self, label: str, tau_Nm: float, omega_rad_s: float,
              gear: float, sized: dict) -> tuple[list[str], list[str]]:
        """Return (errors, warnings) for a sized class.

        Errors: beyond the best catalogued actuator of its category. Warnings:
        best-in-class (above p90) or off a fitted trend.
        """
        errors: list[str] = []
        warns: list[str] = []
        cat = sized.get("category") or "ALL"
        front = self.frontier.get(cat, self.frontier["ALL"])
        mass = sized["mass"]

        td = tau_Nm / mass
        td_max = front["torque_density_Nm_per_kg"]["max"]
        td_p90 = front["torque_density_Nm_per_kg"]["p90"]
        if td > td_max:
            errors.append(
                f"{label}: torque density {td:.0f} Nm/kg beats the best catalogued "
                f"{cat} actuator ({td_max:.0f} Nm/kg, "
                f"{front['torque_density_Nm_per_kg']['argmax']}). "
                f"τ={tau_Nm:.0f} Nm at N={gear:g} implies {mass*1e3:.0f} g.")
        elif td > td_p90:
            warns.append(
                f"{label}: torque density {td:.0f} Nm/kg is above the {cat} p90 "
                f"({td_p90:.0f} Nm/kg) — best-in-class territory.")

        p_peak = tau_Nm * omega_rad_s / 4.0     # the catalogue's envelope convention
        pd = p_peak / mass
        pd_max = front["power_density_W_per_kg"]["max"]
        pd_p90 = front["power_density_W_per_kg"]["p90"]
        if pd > pd_max:
            errors.append(
                f"{label}: power density {pd:.0f} W/kg beats the best catalogued "
                f"{cat} actuator ({pd_max:.0f} W/kg, "
                f"{front['power_density_W_per_kg']['argmax']}). "
                f"P_peak={p_peak:.0f} W in {mass*1e3:.0f} g.")
        elif pd > pd_p90:
            warns.append(
                f"{label}: power density {pd:.0f} W/kg is above the {cat} p90 "
                f"({pd_p90:.0f} W/kg) — best-in-class territory.")

        # Speed vs the speed trend; wide 2.5x gate since the trend is scattered.
        ratio = omega_rad_s / sized["omega_law"]
        if ratio > 2.5 or ratio < 1 / 2.5:
            warns.append(
                f"{label}: no-load speed {omega_rad_s:.1f} rad/s is {ratio:.1f}x the "
                f"speed trend's {sized['omega_law']:.1f} rad/s for N={gear:g} "
                f"(ω = {self.speed_coef:.0f}·N^{self.speed_exp:.2f}).")

        # Slenderness: warn only outside the catalogued min-max, not p10-p90.
        lo, hi = self.aspect_range
        q = sized.get("aspect")
        if q and lo and hi and (q < lo or q > hi):
            p10, p90 = self.aspect_band
            warns.append(
                f"{label}: aspect ratio L/D = {q:.2f} is outside every catalogued "
                f"actuator in this population ({lo:.2f}-{hi:.2f}, p10-p90 "
                f"{p10:.2f}-{p90:.2f}). Volume is still the trend's, so mass is "
                f"unchanged; what moves is r, and armature goes as (L/D)^(-2/3).")

        # Radius envelope; skipped for harmonic units, which run long and narrow.
        if cat != "Harmonic" and p_peak > 0:
            r_mm = sized["r"] * 1000.0
            lo, hi = self.r_lo_k * p_peak ** (1 / 3), self.r_hi_k * p_peak ** (1 / 3)
            if r_mm < lo * 0.5 or r_mm > hi * 2.0:
                warns.append(
                    f"{label}: radius {r_mm:.0f} mm is outside the empirical envelope "
                    f"{lo:.0f}-{hi:.0f} mm for P_peak={p_peak:.0f} W.")
        return errors, warns


def rotor_inertia(r_m: float) -> float:
    """Motor-side rotor inertia ``k·r⁴`` (kg·m²) from the package radius alone.

    Same law as ``MotorTrends.size``; prefer that when (tau, N) are known.
    """
    return motor_fits().inertia_coef * r_m ** 4


def armature_for(r_m: float, gear_ratio: float) -> float:
    """MuJoCo ``armature`` (output-side reflected inertia) = N²·J_rotor."""
    return gear_ratio ** 2 * rotor_inertia(r_m)


# ── Structural trends ─────────────────────────────────────────────────────────


class LinkTrends:
    """Structural densities and whole-robot mass-composition audit for one population.

    Humanoids and quadrupeds are fitted separately and do not transfer.
    ``population`` selects the block: ``humanoid`` is the top level of
    link_trends.json, ``quadruped`` is ``laws["quadruped"]``.
    """

    #: Legal values of `mass_composition_reference`.
    POPULATIONS = ("humanoid", "quadruped")

    def __init__(self, laws_path: Path = LINK_TRENDS_JSON,
                 population: str = "humanoid"):
        if population not in self.POPULATIONS:
            raise ValueError(
                f"unknown reference population {population!r}; datasets/robot_descriptions fits "
                f"{', '.join(self.POPULATIONS)}. A robot with no comparable "
                f"measured population (a hand, a payload) should omit "
                f"`mass_composition_reference` rather than borrow another's.")
        all_laws = _load(laws_path)
        self.population = population
        self._all = all_laws
        self._laws = all_laws if population == "humanoid" else all_laws[population]

        wr = self._laws["whole_robot"]
        self.actuator_fraction = wr["actuator_mass_fraction"]
        self.trunk_fraction = self._laws["segment_mass_fraction"]["trunk_pelvis_torso_head"]
        # mass = coef · S^exp; S is standing height (humanoid) or thigh + shank
        # (quadruped).
        self.mass_vs_size = wr["mass_vs_height"]
        self.size_metric = wr.get("adopted_size_metric", "z_span_m")
        self.size_range = tuple(wr.get("size_range_m") or (0.0, float("inf")))
        # Alias of mass_vs_size.
        self.mass_vs_height = self.mass_vs_size
        # Limb classes for the link-density band (hands/feet excluded).
        eff = self._laws["effective_density"]
        self._limb_rho = {k: eff[k]["rho_eff_kg_m3"]
                          for k in ("thigh", "shank", "forearm") if k in eff}

    def distal_taper(self, chain: str = "leg") -> dict | None:
        """Per-step ratio of structural λ (kg/m) along a limb chain, or None.

        Fit within robots (log-demeaned), so it describes shape, not level. The
        adopted span is the proximal pair (thigh→shank).
        """
        blk = (self._laws.get("distal_taper") or {}).get("chains", {}).get(chain)
        if not blk:
            return None
        adopted = (self._laws.get("distal_taper") or {}).get("adopted") or {}
        span = adopted.get("span", "proximal_pair")
        fit = (blk.get(span) or {}).get("lambda_struct")
        if not fit or not fit.get("per_step"):
            return None
        return {**fit, "chain": chain, "span": span,
                "classes": blk["classes"][:2] if span == "proximal_pair"
                           else blk["classes"],
                "population": self.population,
                # Density share of the taper (not significant for quadrupeds);
                # see `taper_mode` in parameters.yaml.
                "rho_per_step": ((blk.get(span) or {}).get("rho_struct") or {})
                                .get("per_step")}

    def check_distal_taper(self, achieved: float, chain: str = "leg") -> list[str]:
        """Warn if a limb's λ taper is outside the population's CI95 (never an error).

        Catches mass distributed wrongly along a limb, which total-mass checks miss.
        """
        law = self.distal_taper(chain)
        if not law or not achieved or achieved <= 0:
            return []
        lo, hi = law["per_step_ci95"]
        if lo <= achieved <= hi:
            return []
        which = "flatter" if achieved > hi else "steeper"
        return [f"{chain} distal taper {achieved:.2f}x per chain step is {which} "
                f"than the measured {self.population} band "
                f"{lo:.2f}-{hi:.2f} (median {law['per_step']:.2f}, n={law['n']} "
                f"segments over {law['n_robots']} robots): the limb carries its "
                f"mass spread differently along itself than the population does, "
                f"which mass checks cannot see."]

    def trunk_shell_density(self) -> float:
        """Median structural density of the measured pelvis class, actuators removed."""
        return self._laws["effective_density"]["pelvis"]["rho_eff_kg_m3"]["median"]

    #: Classes ``segment_density`` skips: the trunk is owned by ``trunk_density``.
    _DENSITY_LAW_EXCLUDES = ("torso", "pelvis")

    #: Measured class used as the trunk, per population (a quadruped body is `pelvis`).
    TRUNK_CLASS = {"humanoid": "torso", "quadruped": "pelvis"}

    def trunk_density(self) -> Optional[dict]:
        """Measured trunk density for ``torso_rho`` (battery and compute included).

        The quadruped band is wider, since body payloads vary across that population.
        """
        cls = self.TRUNK_CLASS.get(self.population)
        if cls is None:
            return None
        stats = (self._laws["effective_density"].get(cls) or {}).get("rho_eff_kg_m3")
        if not stats or stats.get("median") is None:
            return None
        return {"segment": cls, "rho": float(stats["median"]),
                "p10": stats["p10"], "p90": stats["p90"], "n": stats["n"],
                "population": self.population}

    def segment_density(self, segment: str) -> Optional[dict]:
        """Median measured structural density of one segment class, or None.

        Excludes ``_DENSITY_LAW_EXCLUDES``.
        """
        if segment in self._DENSITY_LAW_EXCLUDES:
            return None
        entry = self._laws["effective_density"].get(segment)
        stats = (entry or {}).get("rho_eff_kg_m3")
        if not stats or stats.get("median") is None:
            return None
        return {"segment": segment, "rho": float(stats["median"]),
                "p10": stats["p10"], "p90": stats["p90"], "n": stats["n"],
                "population": self.population}

    def member_density(self, segment: str) -> Optional[dict]:
        """Like ``segment_density`` but without the trunk exclusion, for generated members."""
        entry = self._laws["effective_density"].get(segment)
        stats = (entry or {}).get("rho_eff_kg_m3")
        if not stats or stats.get("median") is None:
            return None
        return {"segment": segment, "rho": float(stats["median"]),
                "p10": stats["p10"], "p90": stats["p90"], "n": stats["n"],
                "population": self.population}

    def structural_mass(self, segment: str, length_m: float) -> Optional[dict]:
        """Structural mass of a member of this class and joint-to-joint length.

        ``m = c L^e`` inside the fitted length range; outside it, mass per metre is
        held at its edge value. Mass per metre is clamped to the class p10-p90.
        None if the class is not fitted.
        """
        block = (self._laws.get("structural_linear_density") or {}).get(segment)
        if not block or length_m <= 0:
            return None
        lam = block.get("lambda_kg_per_m") or {}
        if lam.get("median") is None:
            return None
        fit = block.get("m_struct_vs_L")
        lo, hi = block.get("length_range_m") or (0.0, float("inf"))
        L = float(length_m)
        if fit:
            edge = min(max(L, lo), hi)
            per_m = fit["coef"] * edge ** (fit["exp"] - 1.0)
            basis = f"m = {fit['coef']:.4g} L^{fit['exp']:.3g} (n={fit['n']})"
            if edge != L:
                basis += (f", mass per metre held at its {edge:.3f} m value: "
                          f"L={L:.3f} m is outside the fitted {lo}-{hi} m")
        else:
            per_m = float(lam["median"])
            basis = f"median {per_m:.3g} kg/m x L (no power fit)"
        clamped = min(max(per_m, lam["p10"]), lam["p90"])
        if clamped != per_m:
            basis += f", clamped to the class p10-p90 ({clamped:.3g} kg/m)"
        return {"segment": segment, "population": self.population,
                "length_m": L, "lambda_kg_per_m": clamped,
                "mass_kg": clamped * L, "basis": basis}

    def segment_density_params(self, params: dict) -> dict[str, dict]:
        """``<seg>_rho`` parameters this population can derive, matched by segment name."""
        out = {}
        for key in params:
            k = str(key)
            if not k.endswith("_rho"):
                continue
            # `torso_rho` maps via TRUNK_CLASS; other keys match by name.
            law = self.trunk_density() if k == "torso_rho" else self.segment_density(k[:-4])
            if law is not None:
                out[k] = law
        return out

    def check_trunk_density(self, rho: float) -> tuple[list[str], list[str]]:
        """(errors, warnings) for trunk density vs measured pelvis segments (actuators included).

        Error above the measured max, warning above p90.
        """
        stats = self._laws["effective_density"]["pelvis"]["rho_total_kg_m3"]
        if rho > stats["max"]:
            return ([f"trunk density {rho:.0f} kg/m³ exceeds the densest measured "
                     f"robot trunk segment ({stats['max']:.0f} kg/m³): the battery "
                     f"pack this design's installed power requires does not fit in "
                     f"the torso volume it declares."], [])
        if rho > stats["p90"]:
            return ([], [f"trunk density {rho:.0f} kg/m³ is above the p90 of measured "
                         f"trunk segments ({stats['p90']:.0f} kg/m³) — the pack only "
                         f"just fits the declared torso volume."])
        return ([], [])

    def link_density_band(self) -> tuple[float, float]:
        """(min p10, max p90) of structural density over the limb classes: a loose sanity band."""
        return (min(v["p10"] for v in self._limb_rho.values()),
                max(v["p90"] for v in self._limb_rho.values()))

    def check_link_density(self, rho: float) -> list[str]:
        lo, hi = self.link_density_band()
        if lo <= rho <= hi:
            return []
        meds = ", ".join(f"{k} {v['median']:.0f}" for k, v in self._limb_rho.items())
        return [f"link_rho {rho:.0f} kg/m³ is outside the measured structural band "
                f"{lo:.0f}-{hi:.0f} kg/m³ (datasets/robot_descriptions effective density, actuator "
                f"mass separated out; segment medians: {meds})."]

    # Body-name substrings summed as trunk mass, to match the reference band
    # ``trunk_pelvis_torso_head`` (which includes mounted actuators).
    TRUNK_BODY_PATTERNS = ("torso", "core", "pelvis", "head", "waist", "trunk")

    @classmethod
    def trunk_mass(cls, mass_by_body: dict, patterns=None) -> float:
        pats = tuple(patterns or cls.TRUNK_BODY_PATTERNS)
        return sum(m for b, m in mass_by_body.items()
                   if any(p in b for p in pats))

    def audit(self, mass_by_role: dict, size_m: float | None = None,
              mass_by_body: dict | None = None, trunk_patterns=None) -> dict:
        """Compare a generated robot's mass composition to this population's bands.

        ``size_m`` is measured per ``self.size_metric``. Skipped by the generator
        when a design declares no ``mass_composition_reference``.
        """
        total = sum(mass_by_role.values())
        out: dict = {"reference_population": self.population,
                     "size_metric": self.size_metric,
                     "total_mass_kg": total, "by_role_kg": dict(mass_by_role),
                     "checks": []}
        if total <= 0:
            return out

        def band(name, value, stats, unit=""):
            if not stats:            # law family missing this class entirely
                out["checks"].append({"quantity": name, "value": round(value, 4),
                                      "status": "no_reference_data"})
                return "no_reference_data"
            status = ("ok" if stats["p10"] <= value <= stats["p90"]
                      else ("outside_p10_p90" if stats["min"] <= value <= stats["max"]
                            else "outside_measured_range"))
            out["checks"].append({
                "quantity": name, "value": round(value, 4), "status": status,
                "median": stats["median"], "p10": stats["p10"], "p90": stats["p90"],
                "min": stats["min"], "max": stats["max"], "n": stats["n"],
                "unit": unit})
            return status

        band("actuator_mass_fraction",
             mass_by_role.get("motor", 0.0) / total, self.actuator_fraction)

        if mass_by_body:
            trunk = self.trunk_mass(mass_by_body, trunk_patterns) / total
            out["trunk_bodies"] = sorted(
                b for b in mass_by_body
                if any(p in b for p in (trunk_patterns or self.TRUNK_BODY_PATTERNS)))
            band("trunk_mass_fraction", trunk, self.trunk_fraction)

        if size_m:
            law = self.mass_vs_size
            lo_fit, hi_fit = self.size_range
            inside = lo_fit <= size_m <= hi_fit
            pred = law["coef"] * size_m ** law["exp"]
            lo = pred * math.exp(-2 * law["sigma_log"])
            hi = pred * math.exp(+2 * law["sigma_log"])
            # Outside the fitted size range: still report, flagged as extrapolated.
            check = {
                "quantity": "total_mass_vs_size_kg", "value": round(total, 3),
                "status": ("ok" if lo <= total <= hi else "outside_2sigma")
                          if inside else "size_outside_fit_range",
                "predicted": round(pred, 3), "lo_2sigma": round(lo, 3),
                "hi_2sigma": round(hi, 3), "size_m": round(size_m, 4),
                "size_metric": self.size_metric, "extrapolated": not inside}
            if not inside:
                check["fit_range_m"] = [lo_fit, hi_fit]
                # How far past the range, and the resulting mass leverage.
                reach = size_m / hi_fit if size_m > hi_fit else lo_fit / size_m
                check["reach_past_fit"] = round(reach, 3)
                check["mass_leverage"] = round(reach ** law["exp"], 2)
            out["checks"].append(check)
        return out

    def warnings(self, audit: dict) -> list[str]:
        pop = {"humanoid": "humanoids", "quadruped": "quadrupeds"}.get(
            audit.get("reference_population"), "robots")
        msgs = []
        for c in audit.get("checks", []):
            if c["status"] in ("ok", "no_reference_data"):
                continue
            if "median" in c:
                msgs.append(
                    f"{c['quantity']} = {c['value']:.3f} is {c['status'].replace('_', ' ')} "
                    f"(measured median {c['median']:.3f}, p10-p90 "
                    f"{c['p10']:.3f}-{c['p90']:.3f}, n={c['n']} {pop}).")
            elif c["quantity"].endswith("_distal_taper"):
                msgs.extend(self.check_distal_taper(
                    c["value"], c["quantity"][:-len("_distal_taper")]))
            elif c["status"] == "size_outside_fit_range":
                lo, hi = c["fit_range_m"]
                inband = c["lo_2sigma"] <= c["value"] <= c["hi_2sigma"]
                msgs.append(
                    f"{c['size_metric']} = {c['size_m']:.2f} m is outside the "
                    f"{lo:.2f}-{hi:.2f} m range of the measured {pop}, so "
                    f"mass-vs-size is EXTRAPOLATED: the law predicts "
                    f"{c['predicted']:.1f} kg and the design is "
                    f"{c['value']:.1f} kg, "
                    f"{'inside' if inband else 'outside'} 2σ. At "
                    f"{c['reach_past_fit']:.2f}x past the fitted range a "
                    f"^{self.mass_vs_size['exp']:.2f} law carries "
                    f"{c['mass_leverage']:.1f}x of leverage on mass, so treat "
                    f"this as a sanity check and not as evidence.")
            else:
                msgs.append(
                    f"{c['quantity']} = {c['value']:.1f} kg is outside 2σ of the "
                    f"{pop[:-1]} mass-vs-size law (predicts {c['predicted']:.1f} kg "
                    f"for {c['size_metric']} = {c['size_m']:.2f} m).")
        return msgs


# ── Convenience: shared singletons ────────────────────────────────────────────

_MOTOR_FITS: list[MotorTrends] = []
_LINK_FITS: dict[str, LinkTrends] = {}


def motor_fits() -> MotorTrends:
    """The fitted actuator trends (one catalogue population), built once and shared."""
    if not _MOTOR_FITS:
        _MOTOR_FITS.append(MotorTrends())
    return _MOTOR_FITS[0]


def link_fits(population: str = "humanoid") -> LinkTrends:
    """Link/whole-robot trends for one population, cached.

    Audits should pass the design's own ``mass_composition_reference``.
    """
    if population not in _LINK_FITS:
        _LINK_FITS[population] = LinkTrends(population=population)
    return _LINK_FITS[population]
