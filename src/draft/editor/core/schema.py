"""Classify a robot's parameters by kind and owner, with no GUI toolkit involved.

A :class:`ParamKind` decides the widget; a :class:`ParamOwner` decides whether
there is one: ``FREE`` (the designer's), ``CALIBRATED`` (solved from
datasets/robot_descriptions; editable but filed apart) or ``LAW`` (derived by a
fitted trend at generation time; shown read-only).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterator

from draft.paths import repo_root
_REPO_ROOT = repo_root()
from draft.trends.feasibility import CALIBRATED_PARAMS, LinkTrends, link_fits


#: Fallback when a robot declares no `<cls>_motor_effort` at all.
MOTOR_CHOICES: tuple[str, ...] = ("L", "M", "S")

# Same rule as generation/generator.py:_CLASS_RE; `hip_pitch_motor_effort` is a
# per-joint key, not a class.
_CLASS_RE = re.compile(r"^([A-Za-z][A-Za-z0-9]*)_motor_effort$")

#: `motor_rho` only covers geoms outside every actuator class; each class
#: carries its own trend-solved density (trends/feasibility.py:MotorTrends).
_MOTOR_RHO_SOURCE = ("fallback only — each actuator class carries the density the "
                     "mass trend solved for it, so the emitted cylinder weighs what "
                     "the trend charged")


class ParamKind(Enum):
    """What sort of value a parameter holds — hence what edits it."""

    LENGTH = "length"          # a link dimension, in metres
    MOTOR_CLASS = "motor_class"  # `<joint>_mot: L|M|S` — which class drives a joint
    DENSITY = "density"        # `<x>_rho`, kg/m³
    COLOR = "color"            # "r g b a" string
    FLAG = "flag"              # bool
    TEXT = "text"              # a population name, a contact triple, a `${expr}` rule
    NUMBER = "number"          # every other scalar — geometry
    ACTUATOR = "actuator"      # `<cls>_motor_*` — owned by trends/motor_solve.py
    STRUCTURED = "structured"  # list/dict: joint ranges, contact triples, fingers
    PRIVATE = "private"        # leading `_`: catalogue anchors, saved poses


class ParamOwner(Enum):
    FREE = "free"
    CALIBRATED = "calibrated"
    LAW = "law"


#: Kinds the session keeps as scalar values and a frontend renders as fields.
EDITABLE_KINDS = frozenset({ParamKind.LENGTH, ParamKind.MOTOR_CLASS, ParamKind.DENSITY,
                            ParamKind.COLOR, ParamKind.FLAG, ParamKind.TEXT,
                            ParamKind.NUMBER})

#: Display group per kind; `NUMBER` splits further on ownership (`_group_for`).
_GROUPS: dict[ParamKind, str] = {
    ParamKind.LENGTH: "Link Lengths",
    ParamKind.MOTOR_CLASS: "Motor Classes",
    ParamKind.DENSITY: "Density (kg/m³)",
    ParamKind.COLOR: "Colors",
    ParamKind.FLAG: "Switches",
    ParamKind.TEXT: "Expressions & Text",
    ParamKind.NUMBER: "Geometry",
    ParamKind.ACTUATOR: "Actuator Design",
}

#: Group order as the editor presents it. A group with no members is skipped.
GROUP_ORDER: tuple[str, ...] = (
    "Link Lengths", "Motor Classes", "Actuator Design", "Geometry",
    "Calibrated (datasets/robot_descriptions)", "Density (kg/m³)", "Switches",
    "Expressions & Text", "Colors",
)


# ── Colour round trip ─────────────────────────────────────────────────────────

def rgba_to_tuple(s: str) -> tuple[int, int, int, int]:
    """'0.5 0.5 0.5 1.0' → (128, 128, 128, 255)"""
    parts = str(s).strip().split()
    if len(parts) < 4:
        return (128, 128, 128, 255)
    return tuple(max(0, min(255, int(float(p) * 255))) for p in parts[:4])  # type: ignore[return-value]


def tuple_to_rgba(t) -> str:
    """(128, 128, 128, 255) → '0.502 0.502 0.502 1.000'"""
    return " ".join(f"{v / 255:.3f}" for v in t)


def text_param(s: Any) -> Any:
    """A text field's value: a string, unless it is a bare number (so a rule
    pinned to a constant still works inside other expressions)."""
    s = str(s).strip()
    if "${" in s:
        return s
    try:
        return float(s)
    except ValueError:
        return s


# ── Classification ────────────────────────────────────────────────────────────

def motor_classes(data: dict) -> tuple[str, ...]:
    """Motor classes this robot declares, in order (not assumed to be L/M/S)."""
    found = [m.group(1) for m in (_CLASS_RE.match(str(k)) for k in data)
             if m and "_" not in m.group(1)]
    return tuple(dict.fromkeys(found)) or MOTOR_CHOICES


def law_owned_densities(params: dict) -> dict[str, str]:
    """`<x>_rho` keys a fitted trend derives for this design, mapped to why.

    The population named by `mass_composition_reference` decides which apply.
    """
    out: dict[str, str] = {}
    if "motor_rho" in params:
        out["motor_rho"] = _MOTOR_RHO_SOURCE
    pop = params.get("mass_composition_reference")
    if pop in LinkTrends.POPULATIONS:
        for key, law in link_fits(pop).segment_density_params(params).items():
            out[key] = (f"{pop} {law['segment']} structural effective density, "
                        f"datasets/robot_descriptions median of {law['n']} measured segments "
                        f"(p10–p90 {law['p10']:.0f}–{law['p90']:.0f})")
    return out


def _kind_of(key: str, value: Any, classes: tuple[str, ...]) -> ParamKind:
    """One parameter's kind: name tests first, then the value's type.

    Types must round-trip: `<role>_dof: false` is tested with `v is False`.
    """
    if str(key).startswith("_"):
        return ParamKind.PRIVATE
    if isinstance(value, (list, dict)):
        return ParamKind.STRUCTURED
    if isinstance(value, str) and "${" in value:
        # A derived `${...}` rule: keep it text so a numeric widget cannot
        # replace it with a constant.
        return ParamKind.TEXT
    if "length" in key:
        return ParamKind.LENGTH
    if key.endswith("_mot"):
        return ParamKind.MOTOR_CLASS
    if key.endswith("_rho"):
        return ParamKind.DENSITY
    if key.endswith("_color"):
        return ParamKind.COLOR
    if any(key.startswith(f"{c}_motor_") for c in classes):
        # Owned by the actuator solve; the trends overwrite these at generation.
        return ParamKind.ACTUATOR
    if isinstance(value, bool):
        return ParamKind.FLAG
    if not isinstance(value, (int, float)):
        return ParamKind.TEXT
    return ParamKind.NUMBER


def _group_for(kind: ParamKind, owner: ParamOwner) -> str:
    if kind is ParamKind.NUMBER and owner is ParamOwner.CALIBRATED:
        return "Calibrated (datasets/robot_descriptions)"
    return _GROUPS.get(kind, "Other")


@dataclass(frozen=True)
class ParamSpec:
    """One parameter, classified. `default` is the robot's shipped value."""

    key: str
    kind: ParamKind
    owner: ParamOwner
    default: Any
    group: str
    hint: str = ""
    #: For LAW-owned parameters: which fit derives the number, in words.
    source: str = ""

    @property
    def editable(self) -> bool:
        return self.kind in EDITABLE_KINDS and self.owner is not ParamOwner.LAW


class RobotSchema:
    """Every parameter of one robot, classified once from `parameters.yaml`.

    Fixed thereafter: loading another design re-seeds values, not the schema.
    """

    def __init__(self, params: dict) -> None:
        self.motor_classes = motor_classes(params)
        # Read from the shipped parameters: changing `mass_composition_reference`
        # does not re-file densities until the editor is reopened.
        self._law_densities = law_owned_densities(params)

        self.specs: dict[str, ParamSpec] = {}
        for key, value in params.items():
            kind = _kind_of(str(key), value, self.motor_classes)
            owner, hint, source = self._ownership(str(key), kind)
            self.specs[str(key)] = ParamSpec(
                key=str(key), kind=kind, owner=owner, default=value,
                group=_group_for(kind, owner), hint=hint, source=source)

    def _ownership(self, key: str, kind: ParamKind) -> tuple[ParamOwner, str, str]:
        if kind is ParamKind.DENSITY and key in self._law_densities:
            return ParamOwner.LAW, "", self._law_densities[key]
        if kind is ParamKind.ACTUATOR:
            return ParamOwner.LAW, "", "solved by core/motor_solve.py from the class's two stated inputs"
        calibrated = CALIBRATED_PARAMS.get(key)
        if calibrated:
            hint = (f"Calibrated, not chosen — {calibrated}. Re-run that stage "
                    f"rather than nudging it here.") if kind is ParamKind.DENSITY else calibrated
            return ParamOwner.CALIBRATED, hint, ""
        if kind is ParamKind.DENSITY:
            return (ParamOwner.FREE,
                    "No measured counterpart in datasets/robot_descriptions — this design's to state.", "")
        return ParamOwner.FREE, "", ""

    # ── queries ───────────────────────────────────────────────────────────────

    def __iter__(self) -> Iterator[ParamSpec]:
        return iter(self.specs.values())

    def of_kind(self, *kinds: ParamKind) -> list[ParamSpec]:
        return [s for s in self.specs.values() if s.kind in kinds]

    @property
    def editable(self) -> list[ParamSpec]:
        return [s for s in self.specs.values() if s.editable]

    @property
    def law_owned_densities(self) -> dict[str, str]:
        return dict(self._law_densities)

    def groups(self) -> dict[str, list[ParamSpec]]:
        """Editable specs by display group, in `GROUP_ORDER`, empties dropped."""
        out: dict[str, list[ParamSpec]] = {g: [] for g in GROUP_ORDER}
        for spec in self.specs.values():
            if spec.editable:
                out.setdefault(spec.group, []).append(spec)
        return {g: v for g, v in out.items() if v}

    def coerce(self, key: str, value: Any) -> Any:
        """A widget's value, as the type the pipeline should receive."""
        spec = self.specs.get(key)
        if spec is None:
            return value
        if spec.kind is ParamKind.FLAG:
            return bool(value)
        if spec.kind is ParamKind.TEXT:
            return text_param(value)
        if spec.kind is ParamKind.MOTOR_CLASS:
            return str(value)
        if spec.kind is ParamKind.COLOR:
            return value if isinstance(value, str) else tuple_to_rgba(value)
        if spec.kind in (ParamKind.LENGTH, ParamKind.NUMBER, ParamKind.DENSITY):
            try:
                return float(value)
            except (ValueError, TypeError):
                return spec.default
        return value

    def link_density_band(self, values: dict) -> tuple[float, float] | None:
        """The measured structural band `link_rho` must land in, or None."""
        pop = values.get("mass_composition_reference")
        if "link_rho" not in self.specs or pop not in LinkTrends.POPULATIONS:
            return None
        return link_fits(pop).link_density_band()
