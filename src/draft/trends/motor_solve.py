"""Design a motor class from any two of (torque, speed, gear ratio, power).

The fitted speed trend ties no-load speed to reduction, so ``<cls>_motor_mode``
picks which two are stated and the rest is solved; mass, radius, length, density
and armature then follow from (tau, N) via ``MotorTrends.size``. The generator
and the editor share this solve.

Modes: ``torque_speed`` (default; N declared, else the anchor's, else inferred
from the speed trend), ``torque_gear`` (omega from the speed trend),
``power_gear`` (omega from the trend, tau = 4P/omega), ``power_torque``
(omega = 4P/tau). P = tau*omega/4, the peak of the linear torque-speed envelope.

``<cls>_motor_aspect`` (L/D) is free in every mode: the airgap relation fixes
only r^2*L. Because torque density rises with torque under the mass trend, the
catalogued density frontier is a ceiling on tau; ``frontier_ceilings`` returns it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from .feasibility import MotorTrends, motor_fits

#: Modes in the order a picker should offer them.
MODES: tuple[str, ...] = ("torque_speed", "torque_gear", "power_gear", "power_torque")

#: Display label per mode.
MODE_LABELS: dict[str, str] = {
    "torque_speed": "torque + speed  (mass ← laws)",
    "torque_gear":  "torque + gear  (speed ← law)",
    "power_gear":   "power + gear  (torque, speed ← laws)",
    "power_torque": "power + torque  (speed = 4P/τ)",
}

#: The class inputs each mode reads; the rest are solved.
MODE_INPUTS: dict[str, tuple[str, ...]] = {
    "torque_speed": ("effort", "velocity", "gear"),
    "torque_gear":  ("effort", "gear"),
    "power_gear":   ("power", "gear"),
    "power_torque": ("power", "effort", "gear"),
}

#: Fields the fitted trends own outright (see ``feasibility.py``).
DERIVED_FIELDS: tuple[str, ...] = ("r", "L", "rho", "armature", "mass")

#: Inputs every mode accepts. ``aspect`` (L/D) defaults to the population fit.
FREE_INPUTS: tuple[str, ...] = ("aspect",)


def normalize_mode(mode: object) -> str:
    """Mode name, defaulting to ``torque_speed``."""
    m = str(mode) if mode is not None else ""
    return m if m in MODE_INPUTS else "torque_speed"


def peak_power_W(tau_Nm: float, omega_rad_s: float) -> float:
    """The catalogue's peak mechanical power convention, tau*omega/4."""
    return tau_Nm * omega_rad_s / 4.0


# ── Operating point ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class OperatingPoint:
    """One actuator class's (tau, omega, N), and where each number came from."""

    tau: float
    omega: float
    gear: float
    mode: str
    gear_source: str
    notes: tuple[str, ...] = ()

    @property
    def power(self) -> float:
        return peak_power_W(self.tau, self.omega)


def _resolve_gear(gear: Optional[float], omega: Optional[float],
                  anchor: Optional[dict], laws: MotorTrends) -> tuple[float, str, list[str]]:
    """(N, source, notes). Precedence: declared, catalogue part, speed trend."""
    if gear is not None:
        return float(gear), "declared", []
    if anchor and anchor.get("gear"):
        return float(anchor["gear"]), "catalogue_part", []
    if omega is None:
        raise ValueError("cannot infer a gear ratio without a declared no-load speed")
    n = laws.gear_from_speed(float(omega))
    return n, "inferred_from_speed_trend", [
        f"no gear ratio declared and none implied by a catalogue part — inferred "
        f"N={n:.1f} from the declared {omega:.1f} rad/s no-load speed. Mass and "
        f"armature both depend on it; declare it to pin it down."]


def solve_point(mode: str, *, effort: Optional[float] = None,
                velocity: Optional[float] = None, gear: Optional[float] = None,
                power: Optional[float] = None, anchor: Optional[dict] = None,
                laws: Optional[MotorTrends] = None) -> OperatingPoint:
    """Solve (tau, omega, N) from the two quantities ``mode`` says are stated."""
    laws = laws or motor_fits()
    mode = normalize_mode(mode)
    notes: list[str] = []

    def _need(value: Optional[float], name: str) -> float:
        if value is None:
            raise ValueError(f"mode {mode!r} needs {name}, which is not declared")
        return float(value)

    if mode == "torque_gear":
        tau = _need(effort, "peak torque")
        n, src, ns = _resolve_gear(gear, velocity, anchor, laws)
        omega = laws.speed_coef * n ** laws.speed_exp
        notes += ns
    elif mode == "power_gear":
        p = _need(power, "peak power")
        n, src, ns = _resolve_gear(gear, velocity, anchor, laws)
        omega = laws.speed_coef * n ** laws.speed_exp
        tau = 4.0 * p / omega
        notes += ns
    elif mode == "power_torque":
        p = _need(power, "peak power")
        tau = _need(effort, "peak torque")
        omega = 4.0 * p / tau
        n, src, ns = _resolve_gear(gear, omega, anchor, laws)
        notes += ns
    else:  # torque_speed
        tau = _need(effort, "peak torque")
        omega = _need(velocity, "no-load speed")
        n, src, ns = _resolve_gear(gear, omega, anchor, laws)
        notes += ns

    return OperatingPoint(tau=tau, omega=omega, gear=n, mode=mode,
                          gear_source=src, notes=tuple(notes))


# ── Mass response and the frontier ceilings it implies ────────────────────────

def mass_coefficient(gear: float, anchor: Optional[dict],
                     laws: Optional[MotorTrends] = None) -> tuple[float, float]:
    """(C, e) with actuator mass = C * tau^e at this reduction.

    This form makes the frontier ceiling closed-form in tau.
    """
    laws = laws or motor_fits()
    e = laws.tau_exp
    if anchor and anchor.get("gear"):
        c = (anchor["mass"] * anchor["tau"] ** -e
             * (gear / anchor["gear"]) ** laws.gear_exp)
    else:
        c = laws.mass_coef * gear ** laws.gear_exp
    return c, e


def frontier_of(category: Optional[str], laws: Optional[MotorTrends] = None) -> dict:
    """The catalogued density frontier for an actuator family (or ALL)."""
    laws = laws or motor_fits()
    return laws.frontier.get(category or "ALL", laws.frontier["ALL"])


def frontier_ceilings(gear: float, omega: float, anchor: Optional[dict] = None,
                      category: Optional[str] = None,
                      laws: Optional[MotorTrends] = None) -> dict:
    """Largest peak torque the torque- and power-density frontiers allow at (N, omega)."""
    laws = laws or motor_fits()
    c, e = mass_coefficient(gear, anchor, laws)
    front = frontier_of(category, laws)
    inv = 1.0 / (1.0 - e)
    out = {"by_torque_density": (front["torque_density_Nm_per_kg"]["max"] * c) ** inv}
    if omega > 0:
        out["by_power_density"] = (
            4.0 * front["power_density_W_per_kg"]["max"] * c / omega) ** inv
    else:
        out["by_power_density"] = float("inf")
    out["tau_max"] = min(out["by_torque_density"], out["by_power_density"])
    out["binding"] = ("torque_density"
                      if out["by_torque_density"] <= out["by_power_density"]
                      else "power_density")
    return out


def _density_status(value: float, stats: dict) -> str:
    if value > stats["max"]:
        return "beyond_frontier"
    return "above_p90" if value > stats["p90"] else "ok"


# ── A designed class ──────────────────────────────────────────────────────────

@dataclass
class ClassDesign:
    """One motor class solved end to end: stated inputs → operating point → scale."""

    cls: str
    point: OperatingPoint
    sized: dict
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    anchor: Optional[dict] = None
    #: The trends this class was solved against, reused for its frontier verdict.
    laws: Optional[MotorTrends] = None

    @property
    def category(self) -> str:
        return self.sized.get("category") or "ALL"

    @property
    def mass(self) -> float:
        return float(self.sized["mass"])

    @property
    def torque_density(self) -> float:
        return self.point.tau / self.mass

    @property
    def power_density(self) -> float:
        return self.point.power / self.mass

    def densities(self, laws: Optional[MotorTrends] = None) -> dict:
        """Both density metrics against the frontier of this class's family."""
        front = frontier_of(self.sized.get("category"), laws or self.laws)
        td, pd = front["torque_density_Nm_per_kg"], front["power_density_W_per_kg"]
        return {
            "category": self.category,
            "torque_density": {"value": self.torque_density, "unit": "Nm/kg",
                               "p90": td["p90"], "max": td["max"],
                               "argmax": td["argmax"],
                               "status": _density_status(self.torque_density, td)},
            "power_density": {"value": self.power_density, "unit": "W/kg",
                              "p90": pd["p90"], "max": pd["max"],
                              "argmax": pd["argmax"],
                              "status": _density_status(self.power_density, pd)},
        }

    def ceilings(self, laws: Optional[MotorTrends] = None) -> dict:
        return frontier_ceilings(self.point.gear, self.point.omega, self.anchor,
                                 self.sized.get("category"), laws or self.laws)

    def as_params(self) -> dict:
        """The ``<cls>_motor_*`` keys this design implies, inputs and derived alike."""
        c = self.cls
        out = {f"{c}_motor_effort": self.point.tau,
               f"{c}_motor_velocity": self.point.omega,
               f"{c}_motor_gear": self.point.gear,
               f"{c}_motor_power": self.point.power,
               f"{c}_motor_mode": self.point.mode,
               f"{c}_motor_aspect": self.sized["aspect"]}
        for f in DERIVED_FIELDS:
            out[f"{c}_motor_{f}"] = self.sized[f]
        return out


def catalog_name_for(cls: str, params: dict) -> Optional[str]:
    """Catalogue part a class is anchored on, from either spelling."""
    name = params.get(f"_{cls}_motor_catalog")
    if not isinstance(name, str):
        name = params.get(f"{cls}_motor")
    return name if isinstance(name, str) else None


def design_class(cls: str, params: dict,
                 laws: Optional[MotorTrends] = None) -> ClassDesign:
    """Solve one motor class from a flat parameter dict per its ``<cls>_motor_mode``.

    Returns the trend-derived envelope, the feasibility errors/warnings and the anchor.
    """
    laws = laws or motor_fits()
    mode = normalize_mode(params.get(f"{cls}_motor_mode"))
    warnings: list[str] = []

    cat_name = catalog_name_for(cls, params)
    anchor = laws.anchor(cat_name) if cat_name else None
    if cat_name and anchor is None:
        warnings.append(
            f"[{cls}] catalogue part '{cat_name}' has no published envelope — "
            f"falling back to the population fit for this class.")

    def _num(key: str) -> Optional[float]:
        v = params.get(f"{cls}_motor_{key}")
        return None if v is None else float(v)

    effort, velocity = _num("effort"), _num("velocity")
    gear, power = _num("gear"), _num("power")
    aspect = _num("aspect")          # free in every mode; None = population fit
    # Seed a missing power from torque and speed (same operating point).
    if power is None and effort is not None and velocity is not None:
        power = peak_power_W(effort, velocity)

    point = solve_point(mode, effort=effort, velocity=velocity, gear=gear,
                        power=power, anchor=anchor, laws=laws)
    warnings.extend(f"[{cls}] {n}" for n in point.notes)

    sized = laws.size(point.tau, point.gear, anchor, aspect=aspect)
    errors, warns = laws.check(f"[{cls}]", point.tau, point.omega, point.gear, sized)
    warnings.extend(warns)
    return ClassDesign(cls=cls, point=point, sized=sized, errors=list(errors),
                       warnings=warnings, anchor=anchor, laws=laws)


# ── Curves, for plotting the shift ────────────────────────────────────────────

def speed_axis(designs: Sequence[ClassDesign], n: int = 96,
               pad: float = 1.08) -> list[float]:
    """A shared omega grid covering every class's envelope, padded by ``pad``."""
    hi = max((d.point.omega for d in designs), default=1.0) * pad
    hi = max(hi, 1e-3)
    return [hi * i / (n - 1) for i in range(n)]


def envelope_curve(point: OperatingPoint, omegas: Sequence[float]) -> list[float]:
    """Linear stall-to-no-load torque at each speed; NaN past the no-load speed."""
    return [point.tau * (1.0 - w / point.omega) if w <= point.omega else float("nan")
            for w in omegas]


def frontier_curve(design: ClassDesign, omegas: Sequence[float],
                   laws: Optional[MotorTrends] = None) -> list[float]:
    """Frontier torque at each speed for this class's mass: ``min(td_max*m, 4*pd_max*m/omega)``."""
    front = frontier_of(design.sized.get("category"), laws or design.laws)
    m = design.mass
    td = front["torque_density_Nm_per_kg"]["max"] * m
    pd = 4.0 * front["power_density_W_per_kg"]["max"] * m
    return [td if w <= 0 else min(td, pd / w) for w in omegas]
