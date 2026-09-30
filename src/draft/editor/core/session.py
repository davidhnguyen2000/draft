"""One robot being edited: its values, its output folders, and its build.

A session owns the design; a frontend owns widgets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from draft.paths import repo_root
_REPO_ROOT = repo_root()
from draft.trends import motor_solve
from draft.generation.generator import RobotGenerator
from draft.editor.core.schema import ParamKind, ParamOwner, RobotSchema
from draft.editor.shared import RobotTypeConfig, load_yaml, new_timestamped_dir, to_python


# Output YAMLs in a generated folder, not designs to load back in.
# `*_inertial_properties.yaml` is prefixed by the root body's name.
_NON_PARAMETER_YAML = frozenset({"parameters_resolved.yaml", "feasibility_report.yaml"})


def is_parameter_yaml(path: Path) -> bool:
    return (path.name not in _NON_PARAMETER_YAML
            and not path.name.endswith("_inertial_properties.yaml"))


# ── Output folders ────────────────────────────────────────────────────────────

class OutputFolders:
    """The generated folders one robot type can be edited into, and which is live."""

    def __init__(self, base: Path, config: RobotTypeConfig) -> None:
        self._base = base
        self._config = config
        self.dirs: dict[str, Path] = {}
        self.selected: str = "(none)"
        self.refresh()

    def refresh(self) -> list[str]:
        dirs: dict[str, Path] = {}
        base, prefix = self._base, self._config.output_prefix
        mjcf = self._config.mjcf_filename
        if not base.is_dir():
            # Nothing generated yet; `new_folder()` creates the base.
            self.dirs = {}
            self.selected = self.names[0]
            return self.names

        # `<prefix>_<timestamp>`: the digit keeps robots whose names extend this
        # prefix out of the list.
        for p in sorted(base.glob(f"{prefix}_[0-9]*"), reverse=True):
            if p.is_dir():
                dirs[p.name] = p
        # Fallback: pick up any renamed folder that contains the expected MJCF.
        for p in sorted(base.iterdir(), reverse=True):
            if p.is_dir() and p.name not in dirs and (p / mjcf).is_file():
                dirs[p.name] = p

        self.dirs = dirs
        if self.selected not in dirs:
            self.selected = self.names[0]
        return self.names

    @property
    def names(self) -> list[str]:
        return list(self.dirs) or ["(none)"]

    @property
    def selected_dir(self) -> Path | None:
        return self.dirs.get(self.selected)

    def select(self, name: str) -> Path | None:
        self.selected = name
        return self.selected_dir

    @property
    def base(self) -> Path:
        """Where this robot's generated folders live."""
        return self._base

    def new_folder(self) -> Path:
        d = new_timestamped_dir(self._base, self._config.output_prefix)
        self.refresh()
        self.selected = d.name
        return d

    def ensure_selected(self) -> Path:
        """The live folder, creating a fresh one rather than failing silently."""
        d = self.selected_dir
        if d is None:
            return self.new_folder()
        d.mkdir(parents=True, exist_ok=True)
        return d

    def design_yaml(self, folder: Path | None = None) -> Path | None:
        """The design YAML in a generated folder (skipping output reports), if any."""
        d = folder if folder is not None else self.selected_dir
        if d is None:
            return None
        for candidate in ("parameters_from_gui.yaml", "parameters.yaml"):
            p = d / candidate
            if p.is_file():
                return p
        extras = sorted(p for p in d.glob("*.yaml") if is_parameter_yaml(p))
        return extras[0] if extras else None

    def mjcf(self, folder: Path | None = None) -> Path | None:
        d = folder if folder is not None else self.selected_dir
        if d is None:
            return None
        p = d / self._config.mjcf_filename
        return p if p.is_file() else None


# ── Actuator inputs ───────────────────────────────────────────────────────────

class MotorInputs:
    """The `<cls>_motor_*` quantities a design states, per actuator class.

    `<cls>_motor_mode` picks which two of (torque, speed, gear, power) are
    stated; `trends/motor_solve.py` derives the rest (see :meth:`designs`).
    """

    #: Gearbox losses: no fitted trend covers them, so they stay asserted inputs.
    ASSERTED: tuple[str, ...] = ("friction", "damping")

    def __init__(self, classes, data: dict | None = None) -> None:
        self.classes = tuple(classes)
        self.params: dict[str, Any] = {}
        if data is not None:
            self.seed(data)

    def seed(self, data: dict) -> None:
        """Re-derive each class's stated inputs from `data` through the solve.

        A class `data` does not describe is left unchanged.
        """
        for cls in self.classes:
            try:
                design = motor_solve.design_class(cls, data)
            except (ValueError, KeyError, TypeError):
                continue
            p = design.point
            self.params.update({f"{cls}_motor_mode": p.mode,
                                f"{cls}_motor_effort": float(p.tau),
                                f"{cls}_motor_velocity": float(p.omega),
                                f"{cls}_motor_gear": float(p.gear),
                                f"{cls}_motor_power": float(p.power),
                                # Slenderness L/D; population fit when unstated.
                                f"{cls}_motor_aspect": float(design.sized["aspect"])})
            for key in self.ASSERTED:
                v = data.get(f"{cls}_motor_{key}")
                if v is not None:
                    self.params[f"{cls}_motor_{key}"] = float(v)

    def set(self, params: dict) -> None:
        self.params.update(params)

    def collect(self) -> dict[str, Any]:
        return dict(self.params)

    def solve_params(self) -> dict[str, Any]:
        """The stated inputs the actuator solve reads."""
        return dict(self.params)

    def design(self, cls: str) -> motor_solve.ClassDesign:
        return motor_solve.design_class(cls, self.solve_params())

    def designs(self) -> list[motor_solve.ClassDesign]:
        params = self.solve_params()
        return [motor_solve.design_class(cls, params) for cls in self.classes]


# ── Generation ────────────────────────────────────────────────────────────────

@dataclass
class FeasibilitySummary:
    """`feasibility_report.yaml` reduced to what an editor shows at a glance."""

    total_mass_kg: float = 0.0
    mass_kg: dict[str, float] = field(default_factory=dict)
    checks: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def read(cls, output_dir: Path) -> "FeasibilitySummary | None":
        try:
            report = load_yaml(output_dir / "feasibility_report.yaml")
        except Exception:
            return None
        return cls(
            total_mass_kg=float(report.get("total_mass_kg", 0.0) or 0.0),
            mass_kg=dict(report.get("mass_kg") or {}),
            checks=list((report.get("mass_composition_check") or {}).get("checks", [])),
            warnings=list(report.get("warnings") or []),
        )


@dataclass
class GenerateResult:
    ok: bool
    output_dir: Path
    mjcf_path: Path | None = None
    error: str | None = None
    report: FeasibilitySummary | None = None


def build_robot(config: RobotTypeConfig, params: dict,
                output_dir: Path) -> GenerateResult:
    """Write the design next to its output and run the generator over it.

    The parameters are written first, so a design the build rejects is kept.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = to_python(params)
    (output_dir / "parameters_from_gui.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    robot_dir = config.robot_dir or config.yaml_path.parent
    try:
        RobotGenerator(robot_dir).generate(output_dir, params_override=payload)
    except Exception as exc:
        return GenerateResult(False, output_dir, error=str(exc))

    mjcf = output_dir / config.mjcf_filename
    if not mjcf.is_file():
        return GenerateResult(False, output_dir, error=f"MJCF not produced at {mjcf}")
    return GenerateResult(True, output_dir, mjcf_path=mjcf,
                          report=FeasibilitySummary.read(output_dir))


# ── Session ───────────────────────────────────────────────────────────────────

class EditorSession:
    """A robot's current design: schema-classified values plus what builds them.

    * ``values``     — the editable scalars, one per :class:`ParamSpec`.
    * ``structured`` — list/dict parameters (joint ranges, contact triples);
      carried through because `tree.yaml` references them.
    * ``private``    — `_`-prefixed keys.
    * ``motors``     — the `<cls>_motor_*` inputs (:class:`MotorInputs`).

    Law-owned densities are carried at their loaded value; the key's presence
    is what the trend matches on.
    """

    def __init__(self, config: RobotTypeConfig, generated_base: Path,
                 folders: OutputFolders | None = None) -> None:
        self.config = config
        # Pass `folders` to share one folder list with the frontend.
        self.folders = folders or OutputFolders(generated_base, config)

        self.values: dict[str, Any] = {}
        self.structured: dict[str, Any] = {}
        self.private: dict[str, Any] = {}
        self.law_densities: dict[str, Any] = {}
        self.reschema(self.starting_params())

    def starting_params(self) -> dict[str, Any]:
        """The robot's shipped table with its preset applied on top."""
        params = load_yaml(self.config.yaml_path)
        preset = self.config.preset_path
        if preset is not None:
            try:
                params = {**params, **(load_yaml(preset) or {})}
            except Exception:
                pass                       # a bad preset costs the preset only
        return params

    def reschema(self, params: dict[str, Any]) -> None:
        """Re-derive the schema from `params`, then seed it.

        Lets a design declare motor classes the shipped table does not.
        """
        self.schema = RobotSchema(params)
        self.motors = MotorInputs(self.schema.motor_classes)
        self._seed(params, replace=True)

    # ── values ────────────────────────────────────────────────────────────────

    def _seed(self, data: dict, replace: bool) -> None:
        """Take what `data` says about keys this schema knows; ignore other scalars."""
        for key, spec in self.schema.specs.items():
            if key not in data:
                continue
            if spec.kind is ParamKind.ACTUATOR:
                continue                      # seeded below, via the solve
            if spec.kind is ParamKind.PRIVATE:
                continue
            if spec.kind is ParamKind.STRUCTURED:
                self.structured[key] = data[key]
            elif spec.owner is ParamOwner.LAW:
                self.law_densities[key] = data[key]
            else:
                self.values[key] = self.schema.coerce(key, data[key])

        # Structured/private keys are taken wholesale, not through the schema.
        incoming_structured = {k: v for k, v in data.items()
                               if isinstance(v, (list, dict)) and not str(k).startswith("_")}
        incoming_private = {k: v for k, v in data.items() if str(k).startswith("_")}
        if replace:
            self.structured = incoming_structured
            self.private = incoming_private
        else:
            self.structured.update(incoming_structured)
            self.private = incoming_private
        self.motors.seed(data)

    def set(self, key: str, value: Any) -> None:
        self.values[key] = self.schema.coerce(key, value)

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, self.law_densities.get(key, default))

    def load_params(self, data: dict) -> None:
        """Re-seed from a loaded design, keeping structured keys already carried."""
        self._seed(data, replace=False)

    def reset_to_defaults(self) -> None:
        """Back to the starting design: `parameters.yaml` plus the robot's preset
        (`g1` for the humanoid, `cheetah` for the quadruped). Re-derives the schema."""
        self.reschema(self.starting_params())

    def load_selected_folder(self) -> Path | None:
        yp = self.folders.design_yaml()
        if yp is None:
            return None
        try:
            data = load_yaml(yp)
        except Exception:
            return None
        self.load_params(data)
        return yp

    # ── the design, assembled ─────────────────────────────────────────────────

    def collect(self) -> dict[str, Any]:
        """Exactly what would be generated — and what `save_as` writes."""
        params: dict[str, Any] = dict(self.structured)
        params.update(self.values)
        params.update(self.law_densities)
        params.update(self.private)
        # Last: the actuator solve owns every `<cls>_motor_*` key. Catalogue
        # anchors are dropped; actuators are sized by the population fit.
        for key in [k for k in params if str(k).endswith("_motor_catalog")]:
            params.pop(key)
        params.update(self.motors.collect())
        return params

    @property
    def allow_hypothetical(self) -> bool:
        """Whether a beyond-frontier actuator is a warning rather than a refusal."""
        return bool(self.values.get("allow_hypothetical", False))

    # ── actions ───────────────────────────────────────────────────────────────

    def generate(self) -> GenerateResult:
        """Build the current design into the live folder."""
        output_dir = self.folders.ensure_selected()
        result = build_robot(self.config, self.collect(), output_dir)
        self.folders.refresh()
        return result

    def save_as(self, name: str) -> Path:
        """Build this design into its own named folder (unlike `generate`, which
        reuses the live one). Never touches the shipped `parameters.yaml`."""
        slug = "".join(c if (c.isalnum() or c in "-_") else "-"
                       for c in name.strip()).strip("-") or "design"
        out = self.folders.base / f"{self.config.output_prefix}_{slug}"
        out.mkdir(parents=True, exist_ok=True)
        result = build_robot(self.config, self.collect(), out)
        self.folders.refresh()
        self.folders.select(out.name)
        return result.output_dir if result.ok else out
