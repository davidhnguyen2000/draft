"""Measurement → parameter overrides → a generated twin.

Only two things cross from the original: geometry (link lengths from joint
placements, plus foot/hand/trunk shape) and actuation (per-joint peak torque and
no-load speed). Every mass, envelope and armature is derived by
`draft/trends/feasibility.py`; no mass is copied.

Gear ratio is inferred from the declared speed via the fitted speed trend.
Roles the target lacks are switched off (`<role>_dof: false`). Where measured
spacing would make actuators intersect, `generation/packing.py` rejects it and
`build()` lengthens the link (recorded in `TwinResult.separation_floors`).
Some parameters depend on trend-sized motor envelopes, so `build()` iterates to
a fixed point.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from draft.paths import repo_root
_REPO = repo_root()
from draft.generation.generator import RobotGenerator          # noqa: E402
from draft.trends.feasibility import (LINK_TRENDS_JSON, FeasibilityViolation,  # noqa: E402
                                        motor_fits)

from .measure import (Measurement, RobotMeasurer, mjcf_shape_features,  # noqa: E402
                      shipped_palette)
from .render import urdf_for_mujoco                                 # noqa: E402
from .targets import (GENERATED_HUMANOID_ROLES, GENERATED_QUADRUPED_ROLES,  # noqa: E402

                      Target)

# Minimum drawn link length (e.g. for co-located axes).
EPS_LEN = 0.012

# The base designs' motor classes, which a twin replaces wholesale.
_BASE_CLASS_RE = re.compile(r"^_?[LMS]_motor")

# role → 2-letter motor-class code (class names may not contain '_').
HUMANOID_CLASS = {
    "torso_yaw": "TY", "torso_roll": "TR", "torso_pitch": "TP",
    "hip_pitch": "HP", "hip_roll": "HR", "hip_yaw": "HY", "knee": "KN",
    "ankle_pitch": "AP", "ankle_roll": "AR",
    "shoulder_pitch": "SP", "shoulder_roll": "SR", "shoulder_yaw": "SY",
    "elbow": "EL", "wrist_yaw": "WY", "wrist_pitch": "WP", "wrist_roll": "WR",
}
QUADRUPED_CLASS = {"hip_roll": "HR", "hip_pitch": "HP", "knee": "KN"}


@dataclass
class TwinResult:
    key: str
    label: str
    category: str
    out_dir: Path
    mjcf: Path
    overrides: dict
    feasibility_report: dict
    target: Measurement
    twin: Measurement
    pruned_roles: tuple = ()
    separation_floors: dict = field(default_factory=dict)
    law_relaxed: bool = False
    warnings: list = field(default_factory=list)

    def summary(self) -> dict:
        t, w = self.target, self.twin
        return {
            "key": self.key, "label": self.label, "category": self.category,
            "target_mass_kg": t.total_mass, "twin_mass_kg": w.total_mass,
            "mass_ratio": w.total_mass / t.total_mass if t.total_mass else None,
            "target_dof": t.n_dof, "twin_dof": w.n_dof,
            "pruned_roles": list(self.pruned_roles),
            "separation_floors_m": self.separation_floors,
            "law_relaxed": self.law_relaxed,
            "mjcf": str(self.mjcf),
        }


class TwinSynthesizer:
    """Builds one twin. Subclassed per body plan for the parameter mapping."""

    robot: str = ""
    class_map: dict = {}
    generated_roles: list = []

    def __init__(self, repo: Path = _REPO, base_overrides: dict | None = None):
        self.repo = Path(repo)
        self.robot_dir = self.repo / "src" / "draft" / "robots" / self.robot
        self.base = yaml.safe_load((self.robot_dir / "parameters.yaml").read_text())
        # Experiment-level parameter replacements (e.g. `link_rho` for a density sweep).
        self.base.update(base_overrides or {})
        self.base_overrides = dict(base_overrides or {})
        self.laws = motor_fits()

    #: Each limb as an alternating [role, param, role, param, …] chain, used to
    #: find the link between two colliding actuators.
    CHAINS: list = []

    #: Default slenderness for every class (`<cls>_motor_aspect`), or None for
    #: the catalogue's fitted split. Only r and L move; ``r^2*L`` (hence mass)
    #: is fixed by the airgap relation (`trends/motor_solve.py`).
    MOTOR_ASPECT: float | None = None

    #: role → measured group the actuator must fit inside (the body proximal to
    #: its joint). Empty disables the fit.
    ASPECT_FIT_GROUP: dict = {}

    #: Roles the fit leaves alone. See `aspect_overrides`.
    ASPECT_FIT_SKIP: frozenset = frozenset()

    #: role → (measured half-span feature, role sharing that span): a hard cap
    #: on package length that overrides the box fit. See `_apply_span_limits`.
    ASPECT_SPAN_LIMIT: dict = {}

    #: Per-role slenderness overriding both the fit and `MOTOR_ASPECT`.
    MOTOR_ASPECT_BY_ROLE: dict = {}

    #: family → the tree's own proximal-to-distal role order, used to detect a
    #: permuted axis order against the target's measured chain.
    CHAIN_ORDER: dict = {}

    #: How a permuted family's actuators map to the tree's slots:
    #:   "axis"   by rotation axis: torque per axis is right (default).
    #:   "depth"  by chain position: mass distribution along the chain is right.
    #: On G1/H2, "axis" costs <0.4% of total mass.
    CHAIN_MATCH: str = "axis"

    def _reorder_by_chain(self, m: Measurement, present: dict) -> dict:
        """Re-key a family's actuators when the tree and target order axes differently.

        No-op under ``"axis"``. Under ``"depth"``, applied only when the two
        chains are a permutation of each other.
        """
        if self.CHAIN_MATCH == "axis":
            return dict(present)
        if self.CHAIN_MATCH != "depth":
            raise ValueError(f"CHAIN_MATCH must be 'depth' or 'axis', "
                             f"got {self.CHAIN_MATCH!r}")
        out = dict(present)
        for family, target_roles in (m.chain or {}).items():
            tree_roles = [r for r in self.CHAIN_ORDER.get(family, [])
                          if r in self.class_map]
            target_roles = [r for r in target_roles if r in present]
            if len(tree_roles) < 2 or len(tree_roles) != len(target_roles):
                continue
            if set(tree_roles) == set(target_roles) and tree_roles != target_roles:
                for slot, src in zip(tree_roles, target_roles):
                    out[slot] = present[src]
        return out

    def separating_params(self, role_a, role_b) -> list:
        """Parameters lying between two roles on a limb chain, proximal first."""
        for chain in self.CHAINS:
            roles = chain[::2]
            if role_a in roles and role_b in roles:
                i, j = sorted((roles.index(role_a), roles.index(role_b)))
                return chain[2 * i + 1: 2 * j: 2]
        return []

    def redistribute(self, geom: dict, floors: dict) -> dict:
        """Absorb packing floors within each chain, keeping its total length.

        The excess is taken from the chain's other links in proportion to their
        slack above their own floors; the chain grows only if no slack is left.
        """
        if not floors:
            return geom
        for chain in self.CHAINS:
            params = chain[1::2]
            # Expression-valued parameters are pinned spans, not donors.
            live = [p for p in params if isinstance(geom.get(p), (int, float))]
            if not any(p in floors for p in live):
                continue
            want = {p: float(geom[p]) for p in live}
            lo = {p: float(floors.get(p, EPS_LEN)) for p in live}
            have = {p: max(want[p], lo[p]) for p in live}
            excess = sum(have.values()) - sum(want.values())
            for _ in range(8):
                if excess <= 1e-9:
                    break
                slack = {p: have[p] - lo[p] for p in live}
                total = sum(slack.values())
                if total <= 1e-9:
                    break                       # nothing left to give: the limb grows
                take = min(excess, total)
                for p in live:
                    have[p] -= take * slack[p] / total
                excess -= take
            for p in live:
                geom[p] = round(max(have[p], EPS_LEN), 5)
        return geom

    # ── actuation ────────────────────────────────────────────────────────────
    def motor_overrides(self, m: Measurement) -> tuple[dict, tuple]:
        """One motor class per canonical role, torque and speed straight off the target.

        Roles the target lacks get `<role>_dof: false`. Their motor class is
        still declared so link-radius expressions resolve; it adds no mass.
        Returns (overrides, pruned roles).
        """
        present = {r: (m.torque(r), m.speed(r)) for r in self.class_map
                   if m.torque(r) and m.speed(r)}
        if not present:
            raise ValueError(f"{m.label}: no usable effort/velocity limits")
        present = self._reorder_by_chain(m, present)
        smallest = min(present.values(), key=lambda tv: tv[0])

        fitted = self.aspect_overrides(m, present, smallest)
        out, synthetic = {}, []
        for role, cls in self.class_map.items():
            tau, omega = present.get(role, smallest)
            if role not in present:
                synthetic.append(role)
                out[f"{role}_dof"] = False
            gear = self.laws.gear_from_speed(omega)
            out[f"{cls}_motor_effort"] = round(float(tau), 4)
            out[f"{cls}_motor_velocity"] = round(float(omega), 4)
            out[f"{cls}_motor_gear"] = round(float(gear), 4)
            aspect = self.MOTOR_ASPECT_BY_ROLE.get(
                role, fitted.get(role, self.MOTOR_ASPECT))
            if aspect:
                out[f"{cls}_motor_aspect"] = float(aspect)
            # No trend covers friction/damping; scale them by torque.
            out[f"{cls}_motor_friction"] = round(0.1 * min(1.0, tau / 120.0) + 0.01, 4)
            out[f"{cls}_motor_damping"] = round(0.01 * min(1.0, tau / 120.0) + 0.001, 5)
            out[f"{role}_mot"] = cls
        return out, tuple(synthetic)

    def aspect_overrides(self, m: Measurement, present: dict,
                         smallest: tuple) -> dict:
        """Slenderness per role, solved so the actuator fits the target's link.

        The measured box of the role's group is projected onto the target's
        joint axis: ``L_max`` along it, ``r_max`` across. With the volume
        ``V = 2*pi*q*r^3`` fixed by the trends::

            q >= V / (2*pi*r_max^3)                   (thin enough)
            q <= (L_max / (2*(V/2pi)^(1/3)))^(3/2)    (short enough)

        If both hold, `MOTOR_ASPECT` (or the balanced q) is clamped into the
        window; otherwise ``q = L_max / (2*r_max)``, which overshoots length and
        diameter by the same factor.
        """
        if not self.ASPECT_FIT_GROUP:
            return {}
        self.aspect_fit: dict = {}
        lo, hi = self.laws.aspect_range
        # The target's own joint axes, to project the box onto.
        axes = {j.role: np.asarray(j.axis, float) for j in m.joints
                if j.role and j.side in (None, "L", "FL")}
        for role, group in self.ASPECT_FIT_GROUP.items():
            if role in self.ASPECT_FIT_SKIP or role not in self.class_map:
                continue
            # Geometric envelope (inertia boxes under-report shells).
            dims = [m.features.get(f"fitbox_{group}_{a}") for a in "xyz"]
            axis = axes.get(role)
            if any(v is None for v in dims) or axis is None:
                continue
            dims = [float(v) for v in dims]
            if min(dims) <= 0:
                continue
            # Length along the actuator axis; diameter across the other two.
            k = int(np.argmax(np.abs(axis)))
            rest = [dims[i] for i in range(3) if i != k]
            l_max = dims[k]
            r_max = 0.5 * math.sqrt(rest[0] * rest[1])
            tau, omega = present.get(role, smallest)
            sized = self.laws.size(tau, self.laws.gear_from_speed(omega))
            v = sized["volume"]
            q_thin = v / (2 * math.pi * r_max ** 3)
            q_short = (l_max / (2 * (v / (2 * math.pi)) ** (1 / 3))) ** 1.5
            # The shape of the space: over by the same factor either way.
            q_balanced = l_max / (2 * r_max)
            if q_thin <= q_short:
                q = min(q_short, max(q_thin, self.MOTOR_ASPECT or q_balanced))
            else:
                q = q_balanced
            q = float(np.clip(q, lo, hi))
            r = (v / (2 * math.pi * q)) ** (1 / 3)
            self.aspect_fit[role] = {
                "group": group,
                "axis": "xyz"[k],
                "link_r_mm": round(r_max * 1000, 2),
                "link_span_mm": round(l_max * 1000, 2),
                "aspect_thin_enough": round(q_thin, 3),
                "aspect_short_enough": round(q_short, 3),
                "aspect_balanced": round(q_balanced, 3),
                "aspect_used": round(q, 3),
                "overshoot_length": round(2 * q * r / l_max, 3),
                "overshoot_diameter": round(2 * r / (2 * r_max), 3),
                "motor_r_mm": round(r * 1000, 2),
                "motor_L_mm": round(2 * q * r * 1000, 2),
                # No slenderness fits both bounds; the balanced q is used.
                "oversized": bool(q_thin > q_short),
                "_v": v,
            }
        self._apply_span_limits(m)
        for v in self.aspect_fit.values():
            v.pop("_v", None)
        return {r: v["aspect_used"] for r, v in self.aspect_fit.items()}

    def _apply_span_limits(self, m: Measurement) -> None:
        """Cap package length so the machine's width matches (`ASPECT_SPAN_LIMIT`).

        The pelvis draws both hip-pitch units as one bar, so the hip-roll axis
        cannot sit inboard of `hip_pitch_motor_L + hip_roll_motor_r`. This cap
        overrides the box fit. Recorded as `aspect_width_cap` / `width_limited`.
        """
        lo, hi = self.laws.aspect_range
        for role, (feature, other) in (self.ASPECT_SPAN_LIMIT or {}).items():
            fit = self.aspect_fit.get(role)
            span = m.features.get(feature)
            other_fit = self.aspect_fit.get(other)
            if not fit or span is None or not other_fit:
                continue
            # Subtract the roll actuator's radius; 5 mm headroom keeps the
            # `pin_hip_roll` test in `geometry_overrides` passing.
            room = float(span) - other_fit["motor_r_mm"] / 1000.0 - 0.005
            if room <= 0:
                continue
            v = fit["_v"]
            q_cap = (room / (2 * (v / (2 * math.pi)) ** (1 / 3))) ** 1.5
            q = float(np.clip(min(fit["aspect_used"], q_cap), lo, hi))
            r = (v / (2 * math.pi * q)) ** (1 / 3)
            fit.update(
                aspect_width_cap=round(q_cap, 3),
                width_limited=bool(q_cap < fit["aspect_used"]),
                aspect_used=round(q, 3),
                motor_r_mm=round(r * 1000, 2),
                motor_L_mm=round(2 * q * r * 1000, 2),
                overshoot_length=round(2 * q * r / (fit["link_span_mm"] / 1000), 3),
                overshoot_diameter=round(r / (fit["link_r_mm"] / 1000), 3),
                oversized=bool(2 * q * r > fit["link_span_mm"] / 1000
                               or r > fit["link_r_mm"] / 1000))

    # ── appearance ───────────────────────────────────────────────────────────

    #: One colour scheme per twin (from `assets/march_of_progress.png`): a
    #: neutral for structure, a saturated colour for actuators, a darker detail
    #: ring and an eye accent.
    PALETTES: dict = {
        "g1":  {"neutral": "#86C6EC", "motor": "#7972B6",   # erectus
                "detail": "#2C2872", "eye": "#B0DCC3"},
        "h2":  {"neutral": "#FEC472", "motor": "#DD5847",   # human
                "detail": "#8C332D", "eye": "#4892AF"},
        "go2": {"neutral": "#F6E9B2", "motor": "#79BC79",   # transitional
                "detail": "#0B6947", "eye": "#F3CB52"},
        "b2":  {"neutral": "#AF8261", "motor": "#803F3E",   # primate
                "detail": "#322D2A", "eye": "#E3C69E"},
    }

    #: Parameters each `PALETTES` slot drives; keys a robot lacks are skipped.
    COLOUR_BLOCKING: dict = {
        "neutral": ("link_color", "torso_color", "head_sphere_color",
                    "hand_sphere_color", "foot_color"),
        "motor":   ("motor_color",),
        "detail":  ("mot_detail_color",),
        "eye":     ("left_eye_color", "right_eye_color"),
    }

    #: Use the shipped robot's colours instead of `PALETTES` (cosmetic only).
    MATCH_SHIPPED_COLOURS = False

    @classmethod
    def colour_overrides(cls, palette: dict) -> dict:
        """The shipped robot's colours, when `MATCH_SHIPPED_COLOURS` asks for them.

        Steps a family of colours off the dominant trunk and limb colours.
        Returns `{}` when disabled.
        """
        if not palette or not cls.MATCH_SHIPPED_COLOURS:
            return {}

        def rgba(c, k=1.0):
            r, g, b = (float(np.clip(x * k, 0.0, 1.0)) for x in c)
            return f"{r:.3f} {g:.3f} {b:.3f} 1.000"

        limb = palette.get("_limb") or palette.get("_all")
        trunk = palette.get("_trunk") or limb
        if limb is None:
            return {}
        # Step lighter instead of darker on near-black robots.
        dark = float(np.mean(limb)) < 0.30
        k_motor, k_detail = (1.55, 2.20) if dark else (0.62, 0.34)
        return {
            "link_color": rgba(limb),
            "motor_color": rgba(limb, k_motor),
            "mot_detail_color": rgba(limb, k_detail),
            "torso_color": rgba(trunk),
            "foot_color": rgba(limb, k_motor),
            "head_sphere_color": rgba(trunk, 0.92),
            "hand_sphere_color": rgba(limb, k_motor),
        }

    def palette_overrides(self, key: str) -> dict:
        """This twin's colour scheme, expanded through `COLOUR_BLOCKING`.

        Only keys the robot's `parameters.yaml` declares are emitted.
        """
        scheme = self.PALETTES.get(key) or {}

        def rgba(h: str) -> str:
            h = h.lstrip("#")
            r, g, b = (int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
            return f"{r:.3f} {g:.3f} {b:.3f} 1.000"

        return {k: rgba(colour)
                for slot, colour in scheme.items()
                for k in self.COLOUR_BLOCKING.get(slot, ())
                if k in self.base}

    # ── geometry (subclass) ──────────────────────────────────────────────────
    #: parameter → measured feature it must reproduce; the build measures the
    #: twin and corrects each parameter by the residual.
    FEATURE_OF: dict = {}

    def geometry_overrides(self, m: Measurement, resolved: dict) -> dict:
        raise NotImplementedError

    def measure_twin(self, measurer, mjcf: Path, label: str, category: str):
        """Measure the generated twin, reading `foot_h` off geometry as for the target."""
        m = measurer.measure(mjcf, label)
        shape = mjcf_shape_features(mjcf, self.generated_roles, category)
        if "foot_h" in shape:
            m.features["foot_h"] = shape["foot_h"]
        return m

    def packing_floors(self, report: dict, resolved: dict) -> dict:
        """Minimum link lengths that would separate every interpenetrating pair.

        Returns `{parameter: current length + overlap depth + 1 mm}` for the
        link separating each overlapping pair in the packing report.
        """
        out: dict = {}
        for pair in (report.get("actuator_packing") or {}).get("pairs", []):
            params = self.separating_params(pair.get("a_drives"),
                                            pair.get("b_drives"))
            if not params:
                continue
            # Lengthen the distal link (intermediate ones come from pruned roles).
            param = params[-1]
            current = float(resolved.get(param, 0.0))
            out[param] = max(out.get(param, 0.0),
                             current + float(pair["depth_m"]) + 1e-3)
        return out

    #: Whether the fixed point sizes the trunk section to reproduce the target's
    #: trunk VOLUME as well as its shape. See `trunk_volume_correction`.
    MATCH_TRUNK_VOLUME = False

    def trunk_volume_correction(self, target: Measurement, report: dict,
                                resolved: dict) -> dict:
        """Additive deltas on the trunk section that match the target's volume.

        The twin's trunk is shorter than the vendor's, so the fitted section
        (aspect and chamfer held) is scaled until the emitted inertia-box volume
        equals the target's, which is what the trunk density trend reads. The
        section ends up larger than the vendor's. Deltas are additive on the
        resolved values and clipped per pass.
        """
        if not self.MATCH_TRUNK_VOLUME:
            return {}
        box = target.group_box.get("trunk")
        emitted = (((report.get("segment_densities") or {}).get("derived") or {})
                   .get("torso_rho") or {}).get("volume_convention") or {}
        have = float(emitted.get("inertia_box_volume_m3") or 0.0)
        if box is None or have <= 0:
            return {}
        want = float(np.prod(np.asarray(box, float)))
        if want <= 0:
            return {}
        # Volume ∝ scale² at fixed length.
        scale = float(np.clip(math.sqrt(want / have), 0.5, 2.0))
        out = {}
        for key in ("torso_ly", "torso_lz"):
            cur = _as_float(resolved.get(key), 0.0)
            if cur > 0:
                out[key] = cur * (scale - 1.0)
        return out

    def feature_corrections(self, target: Measurement, twin: Measurement) -> dict:
        """Additive residuals to close the geometry loop, parameter by parameter."""
        out = {}
        for param, feat in self.FEATURE_OF.items():
            a, b = target.features.get(feat), twin.features.get(feat)
            if a is None or b is None:
                continue
            out[param] = out.get(param, 0.0) + float(a) - float(b)
        return out

    @staticmethod
    def strip_base_classes(gen: RobotGenerator) -> None:
        """Drop the base design's L/M/S classes so only the twin's remain.

        `params_override` can only add keys, so leftover classes would otherwise
        be sized and reported.
        """
        for key in [k for k in gen.params if _BASE_CLASS_RE.match(str(k))]:
            gen.params.pop(key)

    # ── build ────────────────────────────────────────────────────────────────
    def build(self, target: Target, m: Measurement, out_dir: Path,
              passes: int = 5, shape_model: Path | None = None) -> TwinResult:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        gen = RobotGenerator(self.robot_dir)
        self.strip_base_classes(gen)

        # Shipped colours if enabled, else the twin's own scheme.
        colours = self.colour_overrides(
            shipped_palette(shape_model, target.role_map, m.category)
            if shape_model else {})
        colours = colours or self.palette_overrides(target.key)
        motors, synthetic = self.motor_overrides(m)
        resolved = dict(self.base)          # seeds the fixed point with base envelopes
        overrides: dict = {}
        relaxed = False
        warnings: list = []

        mjcf = out_dir / f"{self.robot}.xml"
        measurer = RobotMeasurer(self.generated_roles, m.category)
        corrections: dict = {}
        separation: dict = {}          # parameter → minimum length, from packing
        twin = None

        for p in range(passes):
            final = p == passes - 1
            # `base_overrides` must reach the generator too, not just this class.
            overrides = dict(self.base_overrides)
            overrides.update(motors)
            geom = self.geometry_overrides(m, resolved)
            for param, delta in corrections.items():
                # Numbers only: expression-valued parameters are pinned.
                if isinstance(geom.get(param), (int, float)):
                    geom[param] = round(max(geom[param] + delta, EPS_LEN), 5)
            # Packing floors win over feature corrections, absorbed within the limb.
            geom = self.redistribute(geom, separation)
            overrides.update(geom)
            overrides.update(colours)
            # Strict gates; packing and structure gates open until the last pass,
            # since early passes have not yet converged.
            overrides["allow_hypothetical"] = relaxed
            overrides["allow_motor_overlap"] = not final
            overrides["allow_actuator_in_structure"] = not final
            try:
                gen.generate(out_dir, params_override=overrides)
            except FeasibilityViolation as exc:
                # A real platform beyond the frontier is a finding: relax and record.
                relaxed = True
                warnings.extend(exc.messages)
                overrides["allow_hypothetical"] = True
                overrides["allow_motor_overlap"] = True
                overrides["allow_actuator_in_structure"] = True
                gen.generate(out_dir, params_override=overrides)
            resolved = yaml.safe_load((out_dir / "parameters_resolved.yaml").read_text())
            report = yaml.safe_load((out_dir / "feasibility_report.yaml").read_text())
            twin = self.measure_twin(measurer, mjcf, f"{m.label} twin", m.category)
            # Accumulate residual corrections across passes.
            reshaped = {p for chain in self.CHAINS for p in chain[1::2]
                        if any(q in separation for q in chain[1::2])}
            for param, delta in self.feature_corrections(m, twin).items():
                # Skip chains reshaped by packing floors.
                if param in reshaped:
                    continue
                corrections[param] = corrections.get(param, 0.0) + delta
            for param, delta in self.trunk_volume_correction(
                    m, report, resolved).items():
                corrections[param] = corrections.get(param, 0.0) + delta
            for param, floor in self.packing_floors(report, resolved).items():
                separation[param] = max(separation.get(param, 0.0), floor)

        feasibility_report = yaml.safe_load((out_dir / "feasibility_report.yaml").read_text())
        self._separation = {k: round(v, 5) for k, v in separation.items()}

        # Re-measure the MJCF actually on disk.
        twin = self.measure_twin(measurer, mjcf, f"{m.label} twin", m.category)

        # Solver state, so the geometry can be re-derived without the fixed point.
        (out_dir / "twin_solve.yaml").write_text(yaml.safe_dump(
            {"corrections": {k: round(float(v), 6) for k, v in corrections.items()},
             "separation_floors": self._separation,
             # Per role: link bounds, chosen slenderness, and `oversized`.
             "aspect_fit": getattr(self, "aspect_fit", {})}, sort_keys=True))

        (out_dir / "twin_overrides.yaml").write_text(
            yaml.safe_dump({k: v for k, v in overrides.items() if v is not None},
                           sort_keys=True))
        return TwinResult(
            key=target.key, label=target.label, category=m.category,
            out_dir=out_dir, mjcf=mjcf, overrides=overrides,
            feasibility_report=feasibility_report, target=m, twin=twin,
            pruned_roles=synthetic, separation_floors=self._separation,
            law_relaxed=relaxed, warnings=warnings,
        )



def _as_float(value, default: float) -> float:
    """A resolved parameter as a number, or `default`.

    Pass 1 is seeded from `parameters.yaml`, which may hold `${...}` expressions.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _palm_radius(volume: float, link_r: float) -> float:
    """Radius of a stub-plus-ball hand of volume V whose stub is `r` long.

    Solves ``(4/3)πr³ + π·link_r²·r = V`` by Newton from the pure-sphere radius.
    """
    r = (3.0 * volume / (4.0 * math.pi)) ** (1.0 / 3.0)
    a, b = 4.0 / 3.0 * math.pi, math.pi * link_r ** 2
    for _ in range(40):
        f = a * r ** 3 + b * r - volume
        if abs(f) < 1e-12:
            break
        r -= f / (3.0 * a * r ** 2 + b)
    return max(r, 1e-3)


class HumanoidTwinSynthesizer(TwinSynthesizer):
    robot = "humanoid"
    class_map = HUMANOID_CLASS
    generated_roles = GENERATED_HUMANOID_ROLES
    #: the catalogue's median slenderness (`actuator_trends.json:aspect_trend`)
    MOTOR_ASPECT = 0.852
    #: role → measured segment its actuator must fit inside (the proximal body).
    ASPECT_FIT_GROUP = {
        "torso_yaw": "trunk", "torso_roll": "trunk",
        "torso_pitch": "trunk", "hip_pitch": "trunk",
        "hip_roll": "hip_link", "hip_yaw": "hip_link",
        "knee": "thigh",
        "ankle_pitch": "shank", "ankle_roll": "ankle_link",
        "shoulder_pitch": "shoulder_link",
        "shoulder_roll": "shoulder_link",
        "shoulder_yaw": "shoulder_link",
        "elbow": "upper_arm", "wrist_yaw": "forearm",
        "wrist_pitch": "wrist_link",
        "wrist_roll": "wrist_link",
    }
    #: The ankle is linkage-driven from the shank; `ankle_link` is just a bracket.
    ASPECT_FIT_SKIP = frozenset({"ankle_pitch", "ankle_roll"})

    #: Hip-pitch length is capped by the roll-axis half-span (sets hip width).
    ASPECT_SPAN_LIMIT = {"hip_pitch": ("hip_roll_y", "hip_roll")}

    #: Empty: the per-class fit decides.
    MOTOR_ASPECT_BY_ROLE: dict = {}

    #: Scale on the measured head radius (the head carries `head_rho` mass).
    HEAD_FILL = 1.0
    CHAIN_ORDER = {
        "shoulder": ["shoulder_pitch", "shoulder_roll", "shoulder_yaw"],
        "wrist": ["wrist_yaw", "wrist_pitch", "wrist_roll"],
        "hip": ["hip_pitch", "hip_roll", "hip_yaw"],
        "ankle": ["ankle_pitch", "ankle_roll"],
        "torso": ["torso_roll", "torso_pitch", "torso_yaw"],
    }
    #: Chain order in humanoid/tree.yaml. Each limb is split where the tree turns
    #: from lateral to downward, so a floor on one is never paid by the other.
    CHAINS = [
        ["shoulder_pitch", "upper_shoulder_link_length", "shoulder_roll"],
        ["shoulder_roll", "lower_shoulder_link_length",
         "shoulder_yaw", "bicep_link_length",
         "elbow", "forearm_link_length",
         "wrist_yaw", "upper_wrist_link_length",
         "wrist_pitch", "lower_wrist_link_length",
         "wrist_roll"],
        # Spine separate from the leg so spine packing is not paid by the leg.
        ["torso_roll", "upper_core_link_length",
         "torso_pitch", "lower_core_link_length",
         "torso_yaw", "pelvis_link_length", "hip_pitch"],
        ["hip_pitch", "upper_hip_link_length", "hip_roll"],
        ["hip_roll", "lower_hip_link_length",
         "hip_yaw", "upper_leg_link_length",
         "knee", "lower_leg_link_length",
         "ankle_pitch", "ankle_link_length",
         "ankle_roll"],
    ]
    FEATURE_OF = {
        # Leg segments are solved as spans in `geometry_overrides`, not here.
        "bicep_link_length": "bicep", "forearm_link_length": "forearm",
        "upper_hip_link_length": "hip_seg_0", "lower_hip_link_length": "hip_seg_1",
        "upper_shoulder_link_length": "shoulder_seg_0",
        "lower_shoulder_link_length": "shoulder_seg_1",
        "upper_wrist_link_length": "wrist_seg_0",
        "lower_wrist_link_length": "wrist_seg_1",
        "ankle_link_length": "ankle_seg",
    }

    def geometry_overrides(self, m: Measurement, resolved: dict) -> dict:
        f = m.features
        g = lambda k, d=None: f.get(k, d)                              # noqa: E731
        pos = lambda k, d: max(float(g(k, d) or d), EPS_LEN)           # noqa: E731

        # Motor envelopes from the previous pass (hence the fixed point).
        tr_r = float(resolved.get("TR_motor_r", resolved.get("L_motor_r", 0.045)))
        sp_L = float(resolved.get("SP_motor_L", resolved.get("M_motor_L", 0.08)))

        shoulder_z, hip_z = float(g("shoulder_z", 0.3)), float(g("hip_z", -0.1))
        torso_length = max(shoulder_z - hip_z - tr_r, 0.15)

        # Solve lower_core + pelvis from the waist depth; split by waist span.
        waist_drop = shoulder_z - float(g("waist_z", hip_z + 0.05))
        core = torso_length + tr_r - waist_drop
        core = float(np.clip(core, 2 * EPS_LEN, 0.85 * torso_length))
        span = float(g("waist_span", 0.0) or 0.0)
        lower_core = float(np.clip(span, EPS_LEN, core - EPS_LEN))
        pelvis = core - lower_core

        trunk_box = m.group_box.get("trunk")
        # Shoulder inset is solved so the torso width matches the target's trunk.
        shoulder_y = float(g("shoulder_y", 0.1))
        # Pin the ROLL axes (the limbs hang from them), at shoulder and hip.
        sh_lat = float(resolved.get("upper_shoulder_link_length",
                                    g("shoulder_seg_0", EPS_LEN)))
        shoulder_roll_y = float(g("shoulder_roll_y",
                                  shoulder_y + float(g("shoulder_seg_0", EPS_LEN))))
        hip_roll_y = float(g("hip_roll_y", float(g("hip_y", 0.06))
                             + float(g("hip_seg_0", EPS_LEN))))
        # A roll axis can only be pinned outside its actuator bound (e.g.
        # `hip_pitch_motor_L + hip_roll_motor_r`); otherwise the pitch axis is matched.
        sr_r = float(resolved.get("SR_motor_r", resolved.get("M_motor_r", 0.04)))
        hp_L = float(resolved.get("HP_motor_L", resolved.get("L_motor_L", 0.08)))
        hr_r = float(resolved.get("HR_motor_r", resolved.get("L_motor_r", 0.05)))
        pin_shoulder_roll = shoulder_roll_y >= sp_L / 2 + sr_r + 0.001
        pin_hip_roll = hip_roll_y >= hp_L + hr_r + 0.001
        trunk_w = float(trunk_box[1]) if trunk_box is not None and trunk_box[1] > 0 \
            else 2 * shoulder_y - sp_L
        # the axis the wall is measured back from, per branch
        wall_base = (shoulder_roll_y - sh_lat) if pin_shoulder_roll else shoulder_y
        # Inset in [0, 1]; 1 puts the whole actuator inside the torso wall.
        inset = float(np.clip(0.5 - (2 * wall_base - trunk_w) / (2 * sp_L), 0.0, 1.0))
        # An expression, so the roll axis stays put if actuator aspect changes.
        shoulder_width = (
            "${max(2 * (%.5f - shoulder_pitch_motor_L * (0.5 - "
            "shoulder_motor_inset) - upper_shoulder_link_length), 0.06)}"
            % shoulder_roll_y if pin_shoulder_roll else
            "${max(2 * (%.5f - shoulder_pitch_motor_L"
            " * (0.5 - shoulder_motor_inset)), 0.06)}" % shoulder_y)
        # Hand radius from girth, capped by hand length and the last wrist
        # actuator's radius (WR).
        hand_len = pos("hand_length", 0.07)
        wrist_r = float(resolved.get("WR_motor_r", 0.0)) or 0.035
        hand_r = float(np.clip(0.5 * float(g("hand_girth", 0.5 * hand_len)),
                               0.012, min(0.25 * hand_len, 1.1 * wrist_r)))
        # Redistribute the same volume into a palm with a stub as long as its
        # radius (volume-preserving); fallback when `hand_blob_*` is missing.
        link_r = wrist_r
        stub = max(hand_len - 2 * hand_r, EPS_LEN)
        volume = math.pi * link_r ** 2 * stub + 4 / 3 * math.pi * hand_r ** 3
        hand_r = _palm_radius(volume, link_r)
        depth = (float(trunk_box[2]) if trunk_box is not None and trunk_box[2] > 0
                 else 0.6 * trunk_w)

        ov = {
            # spine
            "torso_length": round(torso_length, 5),
            "lower_core_link_length": round(lower_core, 5),
            "pelvis_link_length": round(pelvis, 5),
            "upper_core_link_length": round(max(0.4 * lower_core, EPS_LEN), 5),
            "shoulder_width": shoulder_width,
            "shoulder_motor_inset": round(inset, 5),
            "shoulder_depth": round(float(np.clip(depth, 0.08, 0.45)), 5),
            # Pins the hip ROLL axes to the target's (expression, as shoulder_width).
            "hip_width": ("${max(2 * (%.5f - upper_hip_link_length), 0.02)}"
                          % hip_roll_y if pin_hip_roll else
                          round(2 * float(g("hip_y", 0.06)), 5)),
            # leg
            "upper_hip_link_length": round(pos("hip_seg_0", EPS_LEN), 5),
            "lower_hip_link_length": round(pos("hip_seg_1", EPS_LEN), 5),
            # Solved so hip-roll-to-knee matches.
            "upper_leg_link_length": ("${max(%.5f - lower_hip_link_length, %.3f)}"
                                      % (pos("thigh_total", pos("thigh", 0.2)),
                                         EPS_LEN)),
            # Solved so knee-to-ankle-roll matches, absorbing the ankle packing floor.
            "lower_leg_link_length": ("${max(%.5f - ankle_link_length, %.3f)}"
                                      % (pos("shank_total", pos("shank", 0.2)),
                                         EPS_LEN)),
            "ankle_link_length": round(pos("ankle_seg", EPS_LEN), 5),
            "foot_length": round(float(g("foot_length", 0.2)), 5),
            "foot_width": round(float(g("foot_width", 0.09)), 5),
            # Used directly, not corrected: the emitted ankle-to-sole has a floor
            # set by the ankle actuator's radius.
            "foot_h": round(float(g("foot_h", 0.05)), 5),
            "foot_offset_x": round(0.17 * float(g("foot_length", 0.2)), 5),
            # arm
            "upper_shoulder_link_length": round(pos("shoulder_seg_0", EPS_LEN), 5),
            "lower_shoulder_link_length": round(pos("shoulder_seg_1", EPS_LEN), 5),
            "bicep_link_length": round(pos("bicep", 0.1), 5),
            "forearm_link_length": round(pos("forearm", 0.1), 5),
            "upper_wrist_link_length": round(pos("wrist_seg_0", EPS_LEN), 5),
            "lower_wrist_link_length": round(pos("wrist_seg_1", EPS_LEN), 5),
            # Matched to the sphere the renderer draws for the target's hand
            # (`measure._hand_blob`); the palm solve above is the fallback.
            "hand_r": round(float(g("hand_blob_r", hand_r)), 5),
            "hand_link_length": round(max(
                float(g("hand_blob_reach", hand_r)), EPS_LEN), 5),
            # head: measured off the shipped robot (it carries mass)
            "neck_length": round(float(g("neck_length", 0.40 * torso_length)), 5),
            "head_radius": round(self.HEAD_FILL
                                 * float(g("head_radius", 0.18 * torso_length)), 5),
        }
        return ov


class QuadrupedTwinSynthesizer(TwinSynthesizer):
    robot = "quadruped"
    class_map = QUADRUPED_CLASS
    generated_roles = GENERATED_QUADRUPED_ROLES
    #: Roll and pitch units live in the hip housing; the belt-driven knee unit
    #: sits at the hip end of the thigh (see quadruped/tree.yaml).
    ASPECT_FIT_GROUP = {"hip_roll": "hip_link",
                        "hip_pitch": "hip_link",
                        "knee": "thigh"}
    # Quadruped only: its trunk is a single prism, so one scale is well-posed.
    MATCH_TRUNK_VOLUME = True
    CHAIN_ORDER = {"hip": ["hip_roll", "hip_pitch"], "knee": ["knee"]}
    # The two hip links are orthogonal, so they are separate chains.
    CHAINS = [["hip_roll", "hip_abduction_link_length", "hip_pitch"],
              ["hip_pitch", "hip_roll_link_length", "knee"]]
    FEATURE_OF = {
        "upper_leg_link_length": "thigh", "lower_leg_link_length": "shank",
        # fore-aft, roll axis → pitch axis
        "hip_abduction_link_length": "hip_dx",
        # lateral, pitch axis → the plane the thigh hangs in
        "hip_roll_link_length": "leg_plane_y",
    }

    def geometry_overrides(self, m: Measurement, resolved: dict) -> dict:
        f = m.features
        hr_L = float(resolved.get("HR_motor_L", resolved.get("L_motor_L", 0.06)))
        hp_r = float(resolved.get("HP_motor_r", resolved.get("L_motor_r", 0.05)))
        kn_L = float(resolved.get("KN_motor_L", resolved.get("L_motor_L", 0.06)))

        thigh = max(float(f.get("thigh", 0.22)), EPS_LEN)
        shank = max(float(f.get("shank", 0.22)), EPS_LEN)
        trunk = f.get("trunk_box") or [0.3, 0.2, 0.11]

        # Limb radii come from the structural trend (`link_mass_class` in tree.yaml).

        return {
            "upper_leg_link_length": round(thigh, 5),
            "lower_leg_link_length": round(shank, 5),
            # Fore-aft roll→pitch offset (0 on Go2/B2), floored so the trunk end
            # wall clears the pitch and knee units. An expression, so it tracks
            # actuator shapes and stays out of the correction loop.
            "hip_abduction_link_length": (
                "${max(%.5f, hip_roll_motor_L + 1.1 * max(hip_pitch_motor_r,"
                " knee_motor_r))}" % max(float(f.get("hip_dx", 0.0)), EPS_LEN)),
            # Lateral, pitch axis → thigh plane, plus the knee motor's half length.
            "hip_roll_link_length": round(max(
                float(f.get("leg_plane_y", 0.12)) - float(f.get("abad_y", 0.06))
                + kn_L / 2, EPS_LEN), 5),
            # Trim on top of the abduction link's fore-aft offset.
            "hip_pitch_x_offset": 0.0,
            # Grow the trunk mesh to cover the hip-roll units (see quadruped/mesh.yaml).
            "torso_mesh_cover": 1.0,
            # Roll actuator ends flush with the torso end wall. With `torso_lx`
            # pinning the hip-pitch axes to the measured wheelbase, the trunk is
            # shorter than the vendor's; `MATCH_TRUNK_VOLUME` restores its volume
            # through the section.
            "hip_roll_inset": "${hip_roll_motor_L}",
            "shoulder_width": round(2 * float(f.get("abad_y", 0.06)), 5),
            # Solved so the hip-pitch axes (at `torso_lx/2 - hip_roll_inset +
            # hip_abduction_link_length`) land at the target's `abad_x + hip_dx`.
            "torso_lx": ("${max(2 * (%.5f + hip_roll_motor_L"
                         " - hip_abduction_link_length), 0.08)}"
                         % (float(f.get("abad_x", 0.15))
                            + float(f.get("hip_dx", 0.0)))),
            # Section and chamfer from `measure.trunk_section_fit` (the vendor's
            # shell); the inertia box `trunk_box` is the fallback.
            "torso_ly": round(float(np.clip(
                f.get("trunk_fit_ly") or trunk[1], 0.08, 0.6)), 5),
            "torso_lz": round(float(np.clip(
                f.get("trunk_fit_lz") or trunk[2], 0.05, 0.4)), 5),
            "torso_chamfer": round(float(np.clip(
                f.get("trunk_fit_chamfer") or self.base["torso_chamfer"],
                0.01, 0.49)), 4),
            "size_metric_expr": "${upper_leg_link_length + lower_leg_link_length}",
        }


SYNTHESIZERS = {"humanoid": HumanoidTwinSynthesizer,
                "quadruped": QuadrupedTwinSynthesizer}


def shipped_geometry_model(target: Target, out_dir: Path) -> Path | None:
    """The shipped model as MuJoCo compiles it, preparing a URDF when needed."""
    src = target.mjcf or target.urdf
    if src is None:
        return None
    src = Path(src)
    if src.suffix.lower() == ".xml" and src.exists():
        return src
    return urdf_for_mujoco(src, out_dir / "shipped_mjcf") if src.exists() else None


def build_twin(target: Target, out_root: Path, repo: Path = _REPO,
               chain_match: str | None = None,
               base_overrides: dict | None = None) -> TwinResult:
    """Measure one target and generate its twin under `out_root/<key>/`.

    ``chain_match`` overrides :attr:`TwinSynthesizer.CHAIN_MATCH`;
    ``base_overrides`` replaces `parameters.yaml` entries for this build only.
    """
    meas = RobotMeasurer(target.role_map, target.category).measure(
        target.urdf, target.label)
    out_dir = Path(out_root) / target.key
    # Shape features need the model as MuJoCo compiles it (URDFs are prepared).
    shape_model = shipped_geometry_model(target, out_dir)
    if shape_model:
        meas.features.update(
            mjcf_shape_features(shape_model, target.role_map, target.category))
    synth = SYNTHESIZERS[target.category](repo, base_overrides=base_overrides)
    if chain_match is not None:
        synth.CHAIN_MATCH = chain_match
    return synth.build(target, meas, out_dir, shape_model=shape_model)
