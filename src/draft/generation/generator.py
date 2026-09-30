"""RobotGenerator: parameters.yaml + tree.yaml (+ mesh.yaml) → MJCF.

A robot directory `src/draft/robots/<name>/` holds `parameters.yaml`, `tree.yaml`
(or a `tree.py` defining `get_tree(params)`), and optionally `mesh.yaml` (or a
`mesh.py` RootMesh subclass) for the root body. The fitted trends size and check
the assembled robot. Run:

    RobotGenerator(robot_dir('<name>')).generate(output_dir)
    python scripts/generate_robot.py --robot <name>
"""

from __future__ import annotations

import importlib.util
import math
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from .expr_eval import attrdict, resolve
from .mesh_spec import MeshSpec
from ..trends.feasibility import (FeasibilityViolation, LinkTrends, link_fits)
from ..trends.motor_solve import (DERIVED_FIELDS, FREE_INPUTS, MODE_INPUTS,
                          design_class)
from .spec_builder import SpecEmitter
from .tree_ops import apply_distal_taper, expand_mirrors, measured_taper
from . import packing
from ..trends.motor_catalog import resolve_catalog_motors


# ── YAML helpers ───────────────────────────────────────────────────────────────

def _load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f) or {}


# ── Mesh class loader ──────────────────────────────────────────────────────────

def _ensure_src_path(robot_dir: Path) -> None:
    """Add the enclosing `src/` to sys.path so a robot's own `tree.py`/`mesh.py` can import."""
    for parent in robot_dir.resolve().parents:
        if parent.name == 'src':
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            return


def _load_root_mesh(robot_dir: Path):
    """The root-body mesh from `mesh.yaml` (preferred) or a `mesh.py` class, or None.

    Either way the instance has `.name`, `.density_param` and `.build_and_save(params, out)`.
    """
    mesh_yaml = robot_dir / 'mesh.yaml'
    if mesh_yaml.exists():
        return MeshSpec.load(mesh_yaml)

    mesh_py = robot_dir / 'mesh.py'
    if not mesh_py.exists():
        return None

    _ensure_src_path(robot_dir)
    spec = importlib.util.spec_from_file_location('_robot_mesh', mesh_py)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Find any RootMesh subclass by duck-typing (avoids cross-import issues)
    for name in dir(mod):
        obj = getattr(mod, name)
        try:
            if (isinstance(obj, type)
                    and obj.__name__ != 'RootMesh'
                    and hasattr(obj, 'get_layers')
                    and hasattr(obj, 'build_and_save')):
                return obj()
        except Exception:
            continue
    return None


def _load_tree_fn(robot_dir: Path):
    """`get_tree` from the robot's `tree.py`, or None if it has none."""
    tree_py = robot_dir / 'tree.py'
    if not tree_py.exists():
        return None

    _ensure_src_path(robot_dir)
    spec = importlib.util.spec_from_file_location('_robot_tree', tree_py)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, 'get_tree', None)


# ── Physical-law application ───────────────────────────────────────────────────

# A motor CLASS key is a bare token before `_motor_effort` (L, M, S …), not a
# per-joint key like `hip_pitch_motor_effort`.
_CLASS_RE = re.compile(r'^([A-Za-z][A-Za-z0-9]*)_motor_effort$')

# Fields the fitted trends own; values stated in parameters.yaml are overridden.
_DERIVED_MOTOR_FIELDS = tuple(f for f in DERIVED_FIELDS if f != 'mass')

# Operating-point inputs a mode may derive rather than read; overrides are recorded.
_MODE_SOLVED_FIELDS = ('effort', 'velocity', 'gear', 'power')


def reference_population(params: dict) -> str:
    """Population named by ``mass_composition_reference``, defaulting to 'humanoid'.

    The fallback only sets the density band; an undeclared design is not audited.
    """
    ref = params.get('mass_composition_reference')
    return ref if ref in LinkTrends.POPULATIONS else 'humanoid'


def _motor_class_names(params: dict) -> list[str]:
    """Motor size classes declared in params, in declaration order."""
    out = []
    for key in params:
        m = _CLASS_RE.match(str(key))
        if m and '_' not in m.group(1):
            out.append(m.group(1))
    return out


def size_actuators(params: dict, strict: bool = True) -> tuple[dict, dict]:
    """Size every motor class from its operating point via the fitted trends.

    A class states two of (torque, speed, gear, power), chosen by
    ``<cls>_motor_mode``; `trends/motor_solve.py` derives the rest plus envelope,
    density and armature. Overridden stated values are recorded in the report.
    Raises FeasibilityViolation past the catalogued frontier unless
    ``allow_hypothetical: true``.
    """
    params = dict(params)
    allow = bool(params.get('allow_hypothetical', False))
    report: dict = {'classes': {}, 'errors': [], 'warnings': []}

    for cls in _motor_class_names(params):
        design = design_class(cls, params)
        point, sized = design.point, design.sized

        ignored = []
        for key, derived in design.as_params().items():
            if key.endswith('_mode'):
                continue
            field = key.rsplit('_', 1)[-1]
            # `aspect` is always stated, never solved.
            solved = (field in _DERIVED_MOTOR_FIELDS
                      or (field not in MODE_INPUTS[point.mode]
                          and field not in FREE_INPUTS))
            old = params.get(key)
            if (solved and old is not None
                    and abs(float(old) - derived) > 1e-9 * max(1.0, abs(derived))):
                ignored.append({'param': key, 'declared': float(old),
                                'derived': round(derived, 8)})
            params[key] = derived

        report['errors'].extend(design.errors)
        report['warnings'].extend(design.warnings)
        dens = design.densities()
        report['classes'][cls] = {
            'anchor': sized['anchor'], 'category': sized['category'],
            'designed_from': point.mode,
            'gear_ratio': round(point.gear, 3), 'gear_source': point.gear_source,
            'effort_Nm': round(point.tau, 4),
            'velocity_rad_s': round(point.omega, 6),
            'mass_kg': round(sized['mass'], 6),
            'radius_m': round(sized['r'], 6), 'length_m': round(sized['L'], 6),
            'volume_cm3': round(sized['volume'] * 1e6, 2),
            'aspect_LD': round(sized['aspect'], 4),
            'aspect_source': sized['aspect_source'],
            'density_kg_m3': round(sized['rho'], 2),
            'rotor_inertia_kgm2': round(sized['J_rotor'], 12),
            'rotor_inertia_from': sized['J_rotor_source'],
            'armature_kgm2': round(sized['armature'], 9),
            'torque_density_Nm_per_kg': round(dens['torque_density']['value'], 1),
            'power_density_W_per_kg': round(dens['power_density']['value'], 1),
            'peak_power_W': round(point.power, 1),
            'frontier_tau_max_Nm': round(design.ceilings()['tau_max'], 1),
            'overridden_by_law': ignored,
        }

    if 'link_rho' in params:
        # Checked against the design's own population's density band.
        report['errors'].extend(
            link_fits(reference_population(params)).check_link_density(
                float(params['link_rho'])))

    if report['errors'] and strict and not allow:
        raise FeasibilityViolation(report['errors'])
    if allow and report['errors']:
        report['warnings'] = ([f'allow_hypothetical: {m}' for m in report['errors']]
                              + report['warnings'])
        report['errors'] = []
    return params, report


def _inertia_box_volume(props: dict) -> float | None:
    """Volume of the solid box with the body's principal inertias.

    Same construction as `datasets/robot_descriptions/scripts/02_parse_descriptions.py`:
    a² = 6(I_yy + I_zz - I_xx)/m (and cyclic). None for a non-physical tensor.
    """
    import numpy as _np
    m = float(props.get('mass') or 0.0)
    I = props.get('inertia_com')
    if m <= 0 or I is None:
        return None
    d = _np.linalg.eigvalsh(_np.asarray(I, dtype=float))
    sq = [6.0 * (d[(i + 1) % 3] + d[(i + 2) % 3] - d[i]) / m for i in range(3)]
    if any(v <= 0 for v in sq):
        return None
    a, b, c = (math.sqrt(v) for v in sq)
    return float(a * b * c)


def derive_segment_densities(params: dict) -> tuple[dict, dict]:
    """Set every ``<seg>_rho`` the design's population has measured, from the trends.

    Only for a design declaring ``mass_composition_reference``; ``link_rho`` is untouched.
    """
    params = dict(params)
    report: dict = {'population': params.get('mass_composition_reference'),
                    'derived': {}, 'warnings': []}
    if params.get('mass_composition_reference') not in LinkTrends.POPULATIONS:
        report['note'] = ('no measured population declared — segment densities '
                          'left as stated')
        return params, report

    laws = link_fits(reference_population(params))
    for key, law in laws.segment_density_params(params).items():
        declared = params.get(key)
        params[key] = law['rho']
        entry = {'segment': law['segment'], 'rho_kg_m3': round(law['rho'], 3),
                 'measured_p10_p90': [law['p10'], law['p90']], 'n': law['n'],
                 'source': f"{law['population']} {law['segment']} structural "
                           f"effective density (datasets/robot_descriptions)"}
        if declared is not None and abs(float(declared) - law['rho']) > 1e-9:
            entry['declared'] = float(declared)
        report['derived'][key] = entry
    return params, report


def _joint_axis(joint: dict) -> tuple | None:
    """A tree joint's axis as a unit vector in its body frame, or None."""
    a = joint.get('axis')
    if isinstance(a, str):
        return {'x': (1.0, 0.0, 0.0), 'y': (0.0, 1.0, 0.0),
                'z': (0.0, 0.0, 1.0)}.get(a.strip().lstrip('+-'))
    if isinstance(a, (list, tuple)) and len(a) == 3:
        v = [float(x) for x in a]
        n = math.sqrt(sum(x * x for x in v))
        return tuple(x / n for x in v) if n > 0 else None
    return None


def _assign_structural_mass(tree: dict, fits: LinkTrends,
                            population: str | None) -> None:
    """Give every `link_mass_class` member its class trend's mass and density.

    A jointed body starts a segment; jointless descendants join it. The target
    mass is `LinkTrends.structural_mass` at the joint-to-joint length (axis-to-axis
    for parallel axes; members' own length for terminal/root segments), split by
    member length. Writes ``link_mass_kg``/``link_mass_basis``/``link_rho``.
    """
    segs: dict[str, dict] = {}

    def pos(node):
        return [float(v) for v in (node.get('pos') or (0.0, 0.0, 0.0))]

    def walk(node, seg, offset, is_root=False):
        j = node.get('joint')
        if not is_root and isinstance(j, dict) and j.get('name'):
            here, below = segs[seg]['axis'], _joint_axis(j)
            if here is not None and below is not None and abs(
                    sum(a * b for a, b in zip(here, below))) > 0.999:
                along = sum(o * a for o, a in zip(offset, here))
                offset = [o - along * a for o, a in zip(offset, here)]
            segs[seg]['child_joints'].append(math.sqrt(sum(v * v for v in offset)))
            seg, offset = node['name'], [0.0, 0.0, 0.0]
            segs[seg] = {'members': [], 'child_joints': [], 'root': False,
                         'axis': _joint_axis(j)}
        g = node.get('geometry') or {}
        if g.get('link_mass_class') and float(g.get('link_l') or 0.0) > 0.0:
            segs[seg]['members'].append(node)
        for child in node.get('children', []) or []:
            walk(child, seg, [a + b for a, b in zip(offset, pos(child))])

    segs[tree['name']] = {'members': [], 'child_joints': [], 'root': True,
                          'axis': None}
    walk(tree, tree['name'], [0.0, 0.0, 0.0], is_root=True)

    for seg_name, s in segs.items():
        by_cls: dict[str, list] = {}
        for n in s['members']:
            by_cls.setdefault(str(n['geometry']['link_mass_class']), []).append(n)
        for cls, nodes in by_cls.items():
            lengths = [float(n['geometry']['link_l']) for n in nodes]
            groups = ([([n], [L]) for n, L in zip(nodes, lengths)] if s['root']
                      else [(nodes, lengths)])
            for grp, lens in groups:
                seg_len = (max(s['child_joints'])
                           if s['child_joints'] and not s['root'] else sum(lens))
                pop_ok = population in LinkTrends.POPULATIONS
                law = fits.structural_mass(cls, seg_len) if pop_ok else None
                rho = fits.member_density(cls) if pop_ok else None
                if law is None or rho is None:
                    raise ValueError(
                        f"{grp[0].get('name')}: link_mass_class {cls!r} has no "
                        f"structural {'linear density' if law is None else 'density'} "
                        f"law for population {population!r} "
                        f"(datasets/robot_descriptions link_trends.json)")
                for n, L in zip(grp, lens):
                    g = n['geometry']
                    share = L / sum(lens)
                    g['link_mass_kg'] = law['mass_kg'] * share
                    g['link_mass_basis'] = {**law, 'segment_root': seg_name,
                                            'share': round(share, 4)}
                    g.setdefault('link_rho', rho['rho'])


def collect_joints(node: dict, out: list | None = None) -> list[dict]:
    """Every actuated joint in a resolved tree, with its motor class.

    Includes joints a `module:` node declares under `joints:`.
    """
    out = [] if out is None else out
    j = node.get('joint')
    if isinstance(j, dict) and 'motor_class' in j:
        out.append({'name': j.get('name', node.get('name')),
                    'motor_class': str(j['motor_class']),
                    'effort': j.get('effort'), 'velocity': j.get('velocity')})
    prefix = str(node.get('prefix', f"{node.get('name', '')}_"))
    for jname, cls in (node.get('joints') or {}).items():
        out.append({'name': f'{prefix}{jname}', 'motor_class': str(cls),
                    'effort': None, 'velocity': None})
    for child in node.get('children', []) or []:
        collect_joints(child, out)
    return out


def actuator_mass_audit(joints: list[dict], params: dict,
                        emitted_motor_kg: float) -> dict:
    """Compare the actuator mass the trends charged with the motor-geom mass emitted."""
    charged = sum(float(params[f"{j['motor_class']}_motor_mass"]) for j in joints)
    ratio = emitted_motor_kg / charged if charged > 0 else None
    return {
        'n_joints': len(joints),
        'charged_by_motor_law_kg': round(charged, 4),
        'emitted_motor_geom_kg': round(emitted_motor_kg, 4),
        'emitted_over_charged': round(ratio, 4) if ratio else None,
    }


def _z_span(node: dict, z: float = 0.0, acc: list | None = None) -> float:
    """Vertical span of the joint frames in the neutral pose (no euler in any tree)."""
    acc = [] if acc is None else acc
    z = z + float((node.get('pos') or [0, 0, 0])[2])
    acc.append(z)
    for child in node.get('children', []) or []:
        _z_span(child, z, acc)
    return max(acc) - min(acc)


def size_metric(params: dict, tree: dict, population: str) -> float | None:
    """The size the allometry checks use: ``size_metric_expr`` if stated, else the joint z-span.

    A quadruped without the expression gets None (its z-span is not predictive).
    """
    expr = params.get('size_metric_expr')
    if expr:
        return float(resolve(str(expr), params))
    if population == 'quadruped':
        return None
    return _z_span(tree) or None


# ── Parameter expansion ────────────────────────────────────────────────────────

def _resolve_param_exprs(params: dict) -> dict:
    """Resolve ``${}`` parameter values in repeated passes (``*_expr`` keys are left alone)."""
    out = dict(params)
    pending = {k: v for k, v in out.items()
               if isinstance(v, str) and '${' in v and not k.endswith('_expr')}
    while pending:
        done = []
        for key, expr in pending.items():
            try:
                out[key] = resolve(expr, out)
            except ValueError:
                continue      # depends on a parameter not resolved yet
            done.append(key)
        if not done:
            raise ValueError('Parameter expressions cannot be resolved (unknown '
                             'name or circular reference): '
                             + ', '.join(f'{k}: {v}' for k, v in sorted(pending.items())))
        for key in done:
            pending.pop(key)
    return out


def _expand_motor_params(params: dict) -> dict:
    """Copy class values onto joints: `hip_pitch_mot: L` → `hip_pitch_motor_r` = `L_motor_r`, etc."""
    expanded = dict(params)
    classes = set(_motor_class_names(params))
    global_rho = float(params.get('motor_rho', 2700.0))
    for key, val in list(params.items()):
        if not (key.endswith('_mot') and isinstance(val, str) and val in classes):
            continue
        joint = key[:-4]   # strip '_mot'
        cls   = val
        fields = ('r', 'L', 'effort', 'velocity', 'friction', 'damping', 'armature')
        for f in fields:
            expanded.setdefault(f'{joint}_motor_{f}', params[f'{cls}_motor_{f}'])
        # Per-class rho, else the global motor_rho.
        cls_rho = float(params.get(f'{cls}_motor_rho', global_rho))
        expanded.setdefault(f'{joint}_motor_rho', cls_rho)
    return expanded


# ── Tree resolution ────────────────────────────────────────────────────────────

def _resolve_value(v: Any, params: dict) -> Any:
    if isinstance(v, dict):
        return {k: _resolve_value(vv, params) for k, vv in v.items()}
    if isinstance(v, list):
        return [_resolve_value(x, params) for x in v]
    if isinstance(v, str):
        return resolve(v, params)
    return v

def disabled_roles(params: dict) -> set:
    """Joint roles switched off via ``<role>_dof: false`` (joint and actuator removed, link kept)."""
    return {str(k)[:-4] for k, v in params.items()
            if str(k).endswith('_dof') and v is False}


def prune_disabled_joints(node: dict, roles: set, removed: list | None = None) -> list:
    """Strip joints whose role is off and the actuator geom tagged `motor_for` that role.

    Matched by name suffix, so one flag covers both mirrored sides.
    """
    removed = [] if removed is None else removed
    if not roles:
        return removed
    j = node.get('joint')
    if isinstance(j, dict):
        jname = str(j.get('name', node.get('name', '')))
        if any(jname == r or jname.endswith(f'_{r}') for r in roles):
            node['joint'] = None
            removed.append(jname)
    g = node.get('geometry')
    if isinstance(g, dict) and g.get('motor_for') in roles:
        g['motor_class'] = None
    for child in node.get('children', []) or []:
        prune_disabled_joints(child, roles, removed)
    return removed


def _resolve_tree(node: dict, params: dict, scope: dict | None = None) -> dict:
    """Resolve all ${} expressions in a tree node, recursing into children.

    ``scope`` holds the loop variables of any enclosing ``for_each``; they shadow
    parameters of the same name for the duration of that subtree.
    """
    ns = params if not scope else {**params, **scope}
    result = {}
    for k, v in node.items():
        if k == 'children':
            result[k] = _expand_children(v or [], params, scope)
        else:
            result[k] = _resolve_value(v, ns)
    return result


def _expand_children(children: list, params: dict, scope: dict | None) -> list:
    """Resolve a `children` list, instantiating every `for_each` entry::

        - for_each: ${items}            # any expression yielding a list
          as: f                         # item name (default: item)
          where: ${f.kind == 'a'}       # optional filter
          node:                         # subtree instantiated per item
            name: ${f.name}_link
    """
    out: list[dict] = []
    for child in children:
        if 'for_each' not in child:
            out.append(_resolve_tree(child, params, scope))
            continue
        base = params if not scope else {**params, **scope}
        items = _resolve_value(child['for_each'], base)
        if not isinstance(items, (list, tuple)):
            raise ValueError(f"for_each must resolve to a list, got "
                             f"{type(items).__name__}: {child['for_each']!r}")
        alias = str(child.get('as', 'item'))
        template = child.get('node')
        if not isinstance(template, dict):
            raise ValueError("a for_each entry needs a `node:` subtree to repeat")
        for index, item in enumerate(items):
            inner = {**(scope or {}), alias: attrdict(item),
                     f'{alias}_index': index}
            if 'where' in child and not _resolve_value(child['where'],
                                                       {**params, **inner}):
                continue
            out.append(_resolve_tree(template, params, inner))
    return out


# ── Scene.xml writer ───────────────────────────────────────────────────────────

def _write_scene(output_dir: Path, robot_name: str) -> None:
    (output_dir / 'scene.xml').write_text(
        f'<mujoco>\n'
        f'  <include file="./{robot_name}.xml"/>\n'
        f'  <worldbody>\n'
        f'    <geom name="floor" type="plane" size="10 10 0.1" pos="0 0 0"\n'
        f'          rgba="0.8 0.8 0.8 1" contype="1" conaffinity="1"/>\n'
        f'    <light name="top_light" pos="0 0 5" dir="0 0 -1" diffuse="0.6 0.6 0.6"/>\n'
        f'  </worldbody>\n'
        f'</mujoco>\n',
        encoding='utf-8',
    )


# ── Public API ─────────────────────────────────────────────────────────────────

class RobotGenerator:
    """Generates MJCF from a robot directory (parameters.yaml, tree.yaml, optional mesh.yaml).

    Usage: ``RobotGenerator(robot_dir('humanoid')).generate(Path('generated/humanoid_test'))``
    """

    def __init__(self, robot_dir: Path) -> None:
        self.robot_dir = Path(robot_dir)
        self.params       = _load_yaml(self.robot_dir / 'parameters.yaml')
        self.mesh         = _load_root_mesh(self.robot_dir)

        # tree.py, if present, takes precedence over tree.yaml.
        self._tree_fn = _load_tree_fn(self.robot_dir)
        self.tree     = ({} if self._tree_fn is not None
                         else _load_yaml(self.robot_dir / 'tree.yaml'))

    @property
    def name(self) -> str:
        return self.robot_dir.name

    def _build_tree(self, params: dict) -> dict:
        """Resolve ${}, expand mirrors, drop disabled joints. Pure — re-callable."""
        raw_tree = self._tree_fn(params) if self._tree_fn is not None else self.tree
        tree = expand_mirrors(_resolve_tree(raw_tree, params))
        # After mirroring, so one `<role>_dof: false` covers both sides.
        prune_disabled_joints(tree, disabled_roles(params))
        return tree

    def generate(self,
                 output_dir: Path,
                 params_override: dict | None = None,
                 strict: bool = True) -> Path:
        """Generate MJCF into output_dir and return it.

        Sized and checked by the fitted trends (`trends/feasibility.py`); with
        ``strict``, a violation raises ``FeasibilityViolation``.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        params = dict(self.params)
        if params_override:
            params.update(params_override)

        # 1. Motor classes: catalogue names → torque/speed, then the fitted trends
        #    derive envelope, mass and reflected rotor inertia.
        params = resolve_catalog_motors(params)
        params, actuator_report = size_actuators(params, strict=strict)
        # Segment densities, before the expression pass so `${}` rules can use them.
        params, density_report = derive_segment_densities(params)
        params = _expand_motor_params(params)
        params = _resolve_param_exprs(params)

        # 2. First tree pass — needed to count the actuators the pack must feed.
        expanded = self._build_tree(params)
        joints = collect_joints(expanded)

        # 3. Trunk density: the design's population's measured trunk structure.
        mesh_volume = None
        if self.mesh is not None:
            # Re-saved below if the density changes.
            _, mesh_props = self.mesh.build_and_save(params, output_dir)
            mesh_volume = float(mesh_props['volume'])
        population = reference_population(params)
        pop_fits = link_fits(population)

        # 3a. The measured trunk density is m / V_inertia_box; rescale by
        #     V_box / V_mesh so the lofted mesh weighs what the measurement implies.
        trunk_shape = None
        if mesh_volume and 'torso_rho' in density_report.get('derived', {}):
            box_v = _inertia_box_volume(mesh_props)
            if box_v and box_v > 0:
                factor = box_v / mesh_volume
                measured = float(params['torso_rho'])
                params['torso_rho'] = measured * factor
                trunk_shape = {
                    'measured_rho_kg_m3': round(measured, 2),
                    'mesh_volume_m3': round(mesh_volume, 6),
                    'inertia_box_volume_m3': round(box_v, 6),
                    'shape_factor': round(factor, 4),
                    'applied_rho_kg_m3': round(params['torso_rho'], 2),
                    'why': 'the measured density is mass / inertia-equivalent-box '
                           'volume; the emitted geom is charged on MESH volume, so '
                           'the ratio is applied to keep the mass the measurement '
                           'implies',
                }
                density_report['derived']['torso_rho']['volume_convention'] = trunk_shape
                expanded = self._build_tree(params)

        # 4. Root mesh: re-save at the trend-derived density so its inertial
        #    properties YAML agrees with the MJCF.
        mesh_stl = None
        if self.mesh is not None:
            if trunk_shape is not None:
                self.mesh.build_and_save(params, output_dir)
            mesh_stl = f'{self.mesh.name}_mesh.stl'

        # 4b. Distal taper targets: only chains whose fitted CI excludes 1.0.
        taper_targets = {}
        if (params.get('taper_mode')
                and params.get('mass_composition_reference') in LinkTrends.POPULATIONS):
            for chain in ('leg', 'arm'):
                law = pop_fits.distal_taper(chain)
                if not law:
                    continue
                lo, hi = law['per_step_ci95']
                if lo < 1.0 < hi:
                    continue
                taper_targets[chain] = law['per_step']
        # 4a. Structural masses for `link_mass_class` members (emitter solves radius).
        _assign_structural_mass(expanded, pop_fits,
                                params.get('mass_composition_reference'))

        taper_report = apply_distal_taper(
            expanded, {**params, 'taper_rho_band': pop_fits.link_density_band()},
            taper_targets)
        if taper_report:
            actuator_report['warnings'].extend(taper_report['warnings'])

        # Compile before writing: packing reads actuator poses from the compiled model.
        emitter = SpecEmitter(self.name, mesh_stl, robot_dir=self.robot_dir,
                              asset_dir=output_dir)
        emitter.emit(expanded, params)
        model = emitter.compile(mesh_volume)
        actuator_report['warnings'].extend(emitter.module_warnings)
        if emitter.stated_members:
            actuator_report['warnings'].append(
                f"{len(emitter.stated_members)} structural member(s) declare no "
                f"link_mass_class, so their radius and density are stated rather "
                f"than derived: {', '.join(sorted(emitter.stated_members))}")
        for geom, e in (emitter.structural_link_report() or {}).items():
            if e['density_raised_at_cap'] and e['density_kg_m3'] > 2 * e['class_density_kg_m3']:
                actuator_report['warnings'].append(
                    f"{geom}: to weigh its {e['emitted_member_kg']:.3f} kg share of the "
                    f"{e['class']} trend inside the actuators it joins, the member's "
                    f"density is {e['density_kg_m3']:.0f} kg/m³ against the class's "
                    f"{e['class_density_kg_m3']:.0f}")
            if e['floored_at_5pct']:
                actuator_report['warnings'].append(
                    f"{geom}: the {e['fixed_in_segment_kg']:.3f} kg already in the "
                    f"segment exceeds the {e['class']} trend's {e['target_kg']:.3f} kg, "
                    f"so the member is floored at 5% of the target")
        xml_path = output_dir / f'{self.name}.xml'
        xml_path.write_text(emitter.to_string(), encoding='utf-8')
        # Module assets go beside the model before the round-trip reload.
        for dest, src in emitter.module_assets.items():
            shutil.copy2(src, output_dir / dest)
        roundtrip = emitter.roundtrip_check(xml_path, float(model.body_mass.sum()))

        # 5. Audit mass composition against the design's own population (if declared).
        masses = emitter.mass_report(mesh_volume)
        # Surface any disagreement between the role split and MuJoCo's masses.
        if masses.get('disagreement'):
            actuator_report['warnings'].append(masses['disagreement'])
        audited = params.get('mass_composition_reference') in LinkTrends.POPULATIONS
        audit = (pop_fits.audit(masses['by_role'],
                                size_m=size_metric(params, expanded, population),
                                mass_by_body=masses['by_body'],
                                trunk_patterns=params.get('trunk_bodies'))
                 if audited else
                 {'reference_population': params.get('mass_composition_reference'),
                  'total_mass_kg': masses['total'], 'checks': [],
                  'note': 'no measured population declared — mass composition '
                          'not audited (set mass_composition_reference to one of '
                          f'{", ".join(LinkTrends.POPULATIONS)})'})
        # 5b. Actuator-vs-actuator packing.
        packing_report = packing.check(
            emitter.motor_geoms,
            exempt=set(params.get('packing_exempt_geoms') or ()))
        pack_msgs = packing.messages(packing_report)
        actuator_report['warnings'].extend(packing.soft_messages(packing_report))
        if pack_msgs:
            if strict and not params.get('allow_motor_overlap'):
                raise FeasibilityViolation(pack_msgs)
            actuator_report['warnings'].extend(f'allow_motor_overlap: {m}'
                                            for m in pack_msgs)

        # 5c. Actuator-vs-root-mesh; only `structure_overlap_allowed` roles may overlap.
        struct_overlap = packing.structure_overlap(
            emitter.model, emitter.model_data,
            float(params.get('torso_rho', 1000.0)),
            role_of={m['name']: m.get('drives') for m in emitter.motor_geoms})
        actuator_report['warnings'].extend(packing.structure_messages(
            struct_overlap, masses['by_role'].get('torso', 0.0)))
        struct_msgs = packing.structure_violations(
            struct_overlap, params.get('structure_overlap_allowed') or ())
        if struct_msgs:
            if strict and not params.get('allow_actuator_in_structure'):
                raise FeasibilityViolation(struct_msgs)
            actuator_report['warnings'].extend(
                f'allow_actuator_in_structure: {m}' for m in struct_msgs)

        act_audit = actuator_mass_audit(joints, params,
                                        masses['by_role'].get('motor', 0.0))
        ratio = act_audit['emitted_over_charged']
        if ratio is not None and abs(ratio - 1.0) > 0.05:
            actuator_report['warnings'].append(
                f"emitted motor geoms weigh {act_audit['emitted_motor_geom_kg']:.2f} kg, "
                f"{ratio:.2f}x the {act_audit['charged_by_motor_law_kg']:.2f} kg the "
                f"actuator trends charge for {act_audit['n_joints']} declared joints — "
                f"the tree draws an actuator more than once, or overrides a "
                f"trend-derived motor_r/motor_L. Every mass trend below is being applied "
                f"to a robot the motor trend did not price.")

        report = {
            'robot': self.name,
            'actuator_checks': actuator_report,
            'segment_densities': density_report,
            'actuator_mass_audit': act_audit,
            'actuator_packing': packing_report,
            'actuator_in_structure': struct_overlap,
            'mass_kg': {k: round(v, 4) for k, v in sorted(masses['by_role'].items())},
            'total_mass_kg': round(masses['total'], 4),
            'mass_composition_check': audit,
            'distal_taper': taper_report,
            'link_render': emitter.link_render_report(params.get('link_rho')),
            'link_inset': emitter.link_inset_report(),
            'structural_links': emitter.structural_link_report(),
            'foot_render': emitter.foot_render_report(),
            'xml_roundtrip': roundtrip,
        }
        # Audit the taper whether or not `taper_mode` applied it.
        if audited:
            for chain, got in sorted(measured_taper(expanded, params).items()):
                law = pop_fits.distal_taper(chain)
                if not law:
                    continue
                lo, hi = law['per_step_ci95']
                audit.setdefault('checks', []).append({
                    'quantity': f'{chain}_distal_taper',
                    'value': round(got, 4),
                    'status': 'ok' if lo <= got <= hi else 'outside_ci95',
                    'measured': law['per_step'], 'ci95': [lo, hi],
                    'r2': law['r2'], 'n': law['n'], 'n_robots': law['n_robots'],
                    'unit': 'x per chain step (structural mass per metre)',
                })
        warnings = list(actuator_report['warnings']) + pop_fits.warnings(audit)
        report['warnings'] = warnings
        (output_dir / 'feasibility_report.yaml').write_text(
            yaml.safe_dump(report, sort_keys=False), encoding='utf-8')
        (output_dir / 'parameters_resolved.yaml').write_text(
            yaml.safe_dump({k: v for k, v in sorted(params.items())},
                           sort_keys=False), encoding='utf-8')
        for msg in warnings:
            print(f'WARNING {msg}')

        # Copy any extra .xml files from the robot directory.
        for xml_asset in self.robot_dir.glob('*.xml'):
            shutil.copy2(xml_asset, output_dir / xml_asset.name)

        _write_scene(output_dir, self.name)
        print(f'Generated {self.name} → {output_dir}')
        return output_dir

    @staticmethod
    def make_output_dir(base_dir: Path, prefix: str) -> Path:
        """Create a timestamped subdirectory under base_dir."""
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        d  = Path(base_dir) / f'{prefix}_{ts}'
        n  = 1
        while d.exists():
            d = Path(base_dir) / f'{prefix}_{ts}_{n:02d}'
            n += 1
        d.mkdir(parents=True)
        return d
