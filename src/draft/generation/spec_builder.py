"""Builds a MuJoCo model from a resolved kinematic tree via `mujoco.MjSpec`.

The model is compiled before it is written, so actuator poses and masses are
read back from MuJoCo. A tree node with `module:` attaches a real MJCF file
(see `_attach_module`). The emitted XML is `spec.to_xml()`.
"""
from __future__ import annotations

import math
from pathlib import Path

import mujoco

from .tree_ops import (LINK_MOTOR_PENETRATION_M, avec, axis_center,
                       drawn_radius, euler_for_axis, link_span, proximal_inset,
                       geom_volume, motor_props, rgba, slim_density)


_GEOM_TYPE = {
    'sphere':   mujoco.mjtGeom.mjGEOM_SPHERE,
    'cylinder': mujoco.mjtGeom.mjGEOM_CYLINDER,
    'capsule':  mujoco.mjtGeom.mjGEOM_CAPSULE,
    'box':      mujoco.mjtGeom.mjGEOM_BOX,
    'mesh':     mujoco.mjtGeom.mjGEOM_MESH,
}

_SENSOR_TYPE = {
    'gyro':          (mujoco.mjtSensor.mjSENS_GYRO,          mujoco.mjtObj.mjOBJ_SITE),
    'accelerometer': (mujoco.mjtSensor.mjSENS_ACCELEROMETER, mujoco.mjtObj.mjOBJ_SITE),
    'velocimeter':   (mujoco.mjtSensor.mjSENS_VELOCIMETER,   mujoco.mjtObj.mjOBJ_SITE),
    'framepos':      (mujoco.mjtSensor.mjSENS_FRAMEPOS,      mujoco.mjtObj.mjOBJ_SITE),
    'framequat':     (mujoco.mjtSensor.mjSENS_FRAMEQUAT,     mujoco.mjtObj.mjOBJ_SITE),
    'subtreeangmom': (mujoco.mjtSensor.mjSENS_SUBTREEANGMOM, mujoco.mjtObj.mjOBJ_BODY),
    'subtreecom':    (mujoco.mjtSensor.mjSENS_SUBTREECOM,    mujoco.mjtObj.mjOBJ_BODY),
    'subtreelinvel': (mujoco.mjtSensor.mjSENS_SUBTREELINVEL, mujoco.mjtObj.mjOBJ_BODY),
}

#: MuJoCo's own default when a geom states neither mass nor density.
_DEFAULT_DENSITY = 1000.0


def _size3(vals) -> list:
    """MjSpec size fields are always length 3; pad what we are given."""
    out = [float(v) for v in vals]
    return (out + [0.0, 0.0, 0.0])[:3]


def _set_euler(elem, euler) -> None:
    """Set Euler angles; `alt.type` must be switched too or the angles are ignored."""
    if not euler:
        return
    elem.alt.type = mujoco.mjtOrientation.mjORIENTATION_EULER
    elem.alt.euler = _size3(euler)


def _module_joints(bodies) -> list:
    """Every joint in a module's body tree (`spec.joints` is empty before compile)."""
    out = []
    for b in bodies:
        out.extend(b.joints)
        out.extend(_module_joints(b.bodies))
    return out


def _perdof(v: float) -> list:
    """Per-DOF vector for a hinge's damping/stiffness: value in slot 0 only."""
    return [float(v), 0.0, 0.0]


class SpecEmitter:
    """Builds an `mujoco.MjSpec` from a resolved, mirror-expanded tree dict."""

    def __init__(self, model_name: str, mesh_stl: str | None = None,
                 robot_dir: Path | None = None, asset_dir: Path | None = None):
        self.spec = mujoco.MjSpec()
        self.spec.modelname = model_name
        self._mesh_name = Path(mesh_stl).stem if mesh_stl else None
        self._mesh_file = mesh_stl
        self._robot_dir = Path(robot_dir) if robot_dir else None
        self._asset_dir = Path(asset_dir) if asset_dir else None
        self._params: dict = {}

        self._motor_joints: list[tuple[str, dict]] = []
        # geom name → mass role (actuator / structure / trunk …).
        self._geom_roles: dict[str, str] = {}
        # geom name → (mass_kg, body_name), checked against the compiled model.
        self._geom_mass: dict[str, tuple] = {}
        self._link_render: dict[str, dict] = {}
        #: Links trimmed at their actuator housing (see tree_ops.link_span).
        self._link_inset: dict[str, dict] = {}
        self._structural_links: dict[str, dict] = {}
        #: Members emitted with a stated radius/density rather than a class law.
        self.stated_members: list[str] = []
        #: body name -> the actuator it draws, {mp, axis, ctr}; read by children
        #: for `tree_ops.proximal_inset`.
        self._body_motor: dict[str, dict] = {}
        #: body name -> its parent body's name, for that lookup.
        self._parent_body: dict[str, str] = {}
        self._foot_render: dict[str, dict] = {}
        # Actuator cylinders by geom name; world poses filled in after compile.
        self._motor_records: dict[str, dict] = {}
        self._body_pairs: list[tuple[str, str]] = []
        self._model: mujoco.MjModel | None = None
        self._model_data = None
        #: Warnings raised while attaching modules.
        self.module_warnings: list[str] = []
        #: Module asset files, {filename in emitted XML: source path}; copied beside the model.
        self.module_assets: dict[str, Path] = {}
        # Module mesh name → volume (m³), measured from the file.
        self._module_mesh_vol: dict[str, float] = {}
        # tree node name → the module's actual root body name (for contact excludes).
        self._module_roots: dict[str, str] = {}
        self._mesh_volume: float | None = None
        self._setup()

    # ── Model-level setup ──────────────────────────────────────────────────────

    def _setup(self) -> None:
        self.spec.compiler.degree = False           # angles in radians
        self.spec.compiler.autolimits = True
        opt = self.spec.option
        opt.timestep = 0.002
        opt.iterations = 50
        opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        opt.tolerance = 1e-10

        # Root defaults; overridden per robot in `emit` from parameters.yaml.
        d = self.spec.default
        d.geom.contype = 1
        d.geom.conaffinity = 1
        d.geom.condim = 3
        d.geom.friction = [0.7, 0.1, 0.1]
        d.joint.damping = _perdof(0.01)

        self._visual = self.spec.add_default('visual', self.spec.default)
        self._visual.geom.contype = 0
        self._visual.geom.conaffinity = 0
        self._visual.geom.group = 2

        if self._mesh_name:
            # Pass the STL as bytes so the XML keeps a bare relative filename.
            self.spec.add_mesh(name=self._mesh_name, file=self._mesh_file)
            if self._asset_dir:
                stl = self._asset_dir / self._mesh_file
                if stl.exists():
                    self.spec.assets = {self._mesh_file: stl.read_bytes()}

    # ── Entry point ────────────────────────────────────────────────────────────

    def emit(self, tree: dict, params: dict) -> None:
        self._params = params
        # Optional contact overrides from parameters.yaml.
        d = self.spec.default
        if 'geom_friction' in params:
            d.geom.friction = _size3(str(params['geom_friction']).split())
        if 'geom_solimp' in params:
            d.geom.solimp = ([float(x) for x in str(params['geom_solimp']).split()]
                             + [0.0] * 5)[:5]
        if 'geom_solref' in params:
            d.geom.solref = ([float(x) for x in str(params['geom_solref']).split()]
                             + [0.0] * 2)[:2]

        self._build_body(self.spec.worldbody, tree)
        self._emit_contact_excludes(tree)

        # No <actuator> block: the consumer owns the controller. Peak torque goes
        # on the joint's actuatorfrcrange, no-load speed into a custom numeric.
        for jname, mp in self._motor_joints:
            self.spec.add_numeric(name=f'{jname}_velocity_limit',
                                  data=[float(mp['velocity'])])

        for s in tree.get('sensors', []) or []:
            self._emit_sensor(s)

    # ── Body dispatch ──────────────────────────────────────────────────────────

    def _build_body(self, parent, node: dict):
        name = node['name']
        t = node.get('type', 'basic')

        # A module node is a whole MJCF file, attached rather than built.
        if 'module' in node:
            return self._attach_module(parent, node)

        body = parent.add_body(name=name,
                               pos=_size3(node.get('pos', [0, 0, 0])))
        self._parent_body[name] = str(getattr(parent, 'name', '') or '')
        _set_euler(body, node.get('euler'))

        if t == 'root':
            body.add_freejoint(name=f'{name}_freejoint')
        elif isinstance(node.get('joint'), dict):
            self._emit_joint(body, node, node['joint'])

        if   t == 'root':   self._emit_root_geoms(body, node)
        elif t == 'basic':  self._emit_basic_geoms(body, node)
        elif t == 'sphere': self._emit_sphere_geoms(body, node)
        elif t == 'foot':   self._emit_foot_geoms(body, node)
        elif t == 'geom':   self._emit_geom(body, node)

        for site in node.get('sites', []) or []:
            body.add_site(name=site['name'],
                          pos=_size3(site.get('pos', [0, 0, 0])), group=5)

        for child in node.get('children', []) or []:
            self._build_body(body, child)
        return body

    # ── Modules ────────────────────────────────────────────────────────────────

    def _attach_module(self, parent, node: dict):
        """Attach an MJCF file as a subtree. Node keys:

          module:  .xml path relative to the robot directory (a standalone model).
          prefix:  namespace for its names (default `<name>_`).
          set:     `<kind>.<name>.<attr>: value` overrides applied before attach,
                   e.g. `body.distal.pos`, `joint.mcp.range`.
          joints:  `<joint name>: <motor class>`; priced and given class dynamics.
          motors:  `<geom name>: <motor class>`; resized to the class envelope.
          roles:   `<geom name>: <mass role>`; `_motor`/`_link` names need none.

        The file is re-read on every call (attach consumes the sub-spec).
        """
        path = (self._robot_dir or Path('.')) / str(node['module'])
        if not path.exists():
            raise FileNotFoundError(f"module '{node['module']}' not found at {path}")
        sub = mujoco.MjSpec.from_file(str(path))
        # Before `set:`, so per-element overrides still win.
        self._stamp_contact(sub)

        for key, val in (node.get('set') or {}).items():
            kind, _, rest = str(key).partition('.')
            ename, _, attr = rest.rpartition('.')
            if not (kind and ename and attr):
                raise ValueError(
                    f"module set key '{key}' must read <kind>.<name>.<attr>, "
                    f"e.g. body.distal.pos")
            lookup = getattr(sub, kind, None)
            if not callable(lookup):
                raise ValueError(
                    f"module set key '{key}' names an unknown element kind "
                    f"'{kind}' — expected one of body, geom, joint, site")
            # MjSpec returns None for an unknown name; report it clearly.
            elem = lookup(ename)
            if elem is None:
                raise ValueError(
                    f"module '{node['module']}' has no {kind} named '{ename}' "
                    f"(from set key '{key}')")
            cur = getattr(elem, attr)
            # Vector attributes are fixed-width in MjSpec; match what is there.
            if hasattr(cur, '__len__') and not isinstance(cur, (str, bytes)):
                vals = [float(v) for v in (val if isinstance(val, (list, tuple))
                                           else str(val).split())]
                setattr(elem, attr, (vals + [0.0] * len(cur))[:len(cur)])
            else:
                setattr(elem, attr, type(cur)(val) if cur is not None else val)

        prefix = str(node.get('prefix', f"{node['name']}_"))
        # Record masses before attach adds the prefix.
        pending = [(f'{prefix}{g.name}', g) for g in sub.geoms if g.name]
        roles = {str(k): str(v) for k, v in (node.get('roles') or {}).items()}
        # Module total mass from MuJoCo, to check the per-geom role split against.
        module_mass = float(sub.copy().compile().body_mass.sum())

        sub_bodies = list(sub.worldbody.bodies)
        if not sub_bodies:
            raise ValueError(f"module '{node['module']}' declares no body")
        self._module_roots[node['name']] = f'{prefix}{sub_bodies[0].name}'
        # Read before attaching; sub-spec handles do not survive `attach`.
        module_joint_names = {j.name for j in _module_joints(sub_bodies)}

        frame = parent.add_frame(pos=_size3(node.get('pos', [0, 0, 0])))
        _set_euler(frame, node.get('euler'))
        self._claim_module_assets(sub, prefix, path)
        self.spec.attach(sub, prefix=prefix, frame=frame)

        for full_name, g in pending:
            gtype = str(mujoco.mjtGeom(g.type).name).replace('mjGEOM_', '').lower()
            # An unset fromto reads back as [nan, 0, ...].
            ft = list(g.fromto)
            fromto = ft if all(v == v for v in ft) else None
            mesh_vol = self._module_mesh_vol.get(str(g.meshname)) if gtype == 'mesh' else None
            mass = (float(g.mass) if float(g.mass) > 0 else
                    float(g.density if g.density else _DEFAULT_DENSITY)
                    * geom_volume(gtype, list(g.size), fromto, mesh_vol))
            bare = full_name[len(prefix):]
            role = roles.get(bare, roles.get(full_name))
            if role is None:
                role = ('motor' if bare.endswith('_motor') else
                        'detail' if bare.endswith('_detail') else 'link')
            self._geom_roles[full_name] = role
            self._geom_mass[full_name] = (mass, node['name'])
            if role == 'motor':
                self._motor_records[full_name] = {
                    'name': full_name, 'body': node['name'],
                    'motor_class': None, 'drives': None,
                    'path': (node['name'],),
                }

        accounted = sum(m for n, (m, _) in self._geom_mass.items()
                        if n.startswith(prefix))
        if module_mass > 0 and abs(accounted - module_mass) / module_mass > 0.01:
            self.module_warnings.append(
                f"module '{node['module']}' compiles to {module_mass:.4f} kg but "
                f"{accounted:.4f} kg was attributed to roles — an unnamed geom, or "
                f"a shape whose volume this builder does not compute. Every mass "
                f"law below is applied to the first figure and audited on the "
                f"second, so they have to agree.")

        # `motors:` resizes module actuator geoms to the trend-derived class envelope.
        for gname, cls in (node.get('motors') or {}).items():
            mp = motor_props(str(cls), self._params)
            full = f'{prefix}{gname}'
            g = self.spec.geom(full)
            g.size = _size3([mp['r'], mp['L'] / 2])
            g.density = mp['rho']
            # Leave `mass` unset (NaN = use density); 0.0 would mean massless.
            vol = geom_volume('cylinder', [mp['r'], mp['L'] / 2])
            self._geom_roles[full] = 'motor'
            self._geom_mass[full] = (mp['rho'] * vol, node['name'])
            self._motor_records[full] = {
                'name': full, 'body': node['name'],
                'motor_class': str(cls), 'drives': None,
                'r': float(mp['r']), 'L': float(mp['L']),
            }

        declared = {str(k) for k in (node.get('joints') or {})}
        actual = module_joint_names
        undeclared = sorted(actual - declared)
        if undeclared:
            self.module_warnings.append(
                f"module '{node['module']}' has joint(s) {', '.join(undeclared)} that "
                f"the assembly does not declare under `joints:`. They enter the model "
                f"unpriced by the actuator trends and unactuated — no torque limit, no "
                f"armature, no reflected inertia.")
        missing = sorted(declared - actual)
        if missing:
            raise ValueError(
                f"module '{node['module']}' has no joint(s) named "
                f"{', '.join(missing)}, declared under `joints:`")

        for jname, cls in (node.get('joints') or {}).items():
            mp = motor_props(str(cls), self._params)
            j = self.spec.joint(f'{prefix}{jname}')
            j.armature = mp['armature']
            j.damping = _perdof(mp['damping'])
            j.frictionloss = mp['friction']
            j.actfrclimited = mujoco.mjtLimited.mjLIMITED_TRUE
            j.actfrcrange = [-mp['effort'], mp['effort']]
            self._motor_joints.append((f'{prefix}{jname}', mp))
        return None

    def _stamp_contact(self, sub) -> None:
        """Apply the assembly's contact parameters to every module geom.

        Per geom, since a module's default classes are already resolved at parse time.
        """
        p = self._params
        friction = (_size3(str(p['geom_friction']).split())
                    if 'geom_friction' in p else None)
        solimp = (([float(x) for x in str(p['geom_solimp']).split()] + [0.0] * 5)[:5]
                  if 'geom_solimp' in p else None)
        solref = (([float(x) for x in str(p['geom_solref']).split()] + [0.0] * 2)[:2]
                  if 'geom_solref' in p else None)
        for g in sub.geoms:
            if friction is not None:
                g.friction = friction
            if solimp is not None:
                g.solimp = solimp
            if solref is not None:
                g.solref = solref

    def _claim_module_assets(self, sub, prefix: str, module_path: Path) -> None:
        """Register a module's mesh files for copying beside the emitted model.

        Filenames get the prefix only when two modules use one name for different files.
        """
        base = Path(getattr(sub, 'modelfiledir', '') or module_path.parent)
        meshdir = str(getattr(sub, 'meshdir', '') or '')
        for mesh in sub.meshes:
            ref = str(mesh.file or '')
            if not ref:
                continue                     # inline vertices, nothing to copy
            src = (base / meshdir / ref) if meshdir else (base / ref)
            src = src.resolve()
            if not src.is_file():
                self.module_warnings.append(
                    f"module '{module_path.name}' references asset '{ref}', which is "
                    f"not at {src}. The emitted model will not load.")
                continue
            dest = Path(ref).name
            if self.module_assets.get(dest, src) != src:
                dest = f'{prefix}{dest}'     # same name, different file
            self.module_assets[dest] = src
            mesh.file = dest
            try:
                import trimesh
                loaded = trimesh.load(str(src), force='mesh')
                scale = float(getattr(mesh, 'scale', [1.0, 1.0, 1.0])[0] or 1.0)
                vol = abs(float(loaded.volume)) * scale ** 3
                # Under both names: geoms are read after attach renames meshes.
                self._module_mesh_vol[str(mesh.name)] = vol
                self._module_mesh_vol[f'{prefix}{mesh.name}'] = vol
            except Exception as exc:
                self.module_warnings.append(
                    f"module '{module_path.name}': could not measure the volume of "
                    f"'{ref}' ({type(exc).__name__}), so its mass is missing from the "
                    f"role breakdown the link-trend audit reads.")
        # Files will sit beside the model, so drop the module's meshdir.
        if meshdir:
            sub.meshdir = ''

    # ── Joint ──────────────────────────────────────────────────────────────────

    def _emit_joint(self, body, node: dict, jspec: dict) -> None:
        jname = jspec.get('name', node['name'])
        cls = str(jspec['motor_class'])
        overrides = {k: jspec[k] for k
                     in ('effort', 'velocity', 'friction', 'damping', 'armature')
                     if k in jspec}
        mp = motor_props(cls, self._params, overrides)
        lo, hi = jspec['range']
        j = body.add_joint(name=jname, type=mujoco.mjtJoint.mjJNT_HINGE,
                           axis=_size3(avec(str(jspec['axis']))))
        j.limited = mujoco.mjtLimited.mjLIMITED_TRUE
        j.range = [float(lo), float(hi)]
        j.actfrclimited = mujoco.mjtLimited.mjLIMITED_TRUE
        j.actfrcrange = [-mp['effort'], mp['effort']]
        j.damping = _perdof(mp['damping'])
        j.frictionloss = mp['friction']
        j.armature = mp['armature']
        self._motor_joints.append((jname, mp))

    # ── Geom helpers ───────────────────────────────────────────────────────────

    def _add_geom(self, body, name: str, gtype: str, size, role: str,
                  pos=(0, 0, 0), euler=None, color=None, density: float | None = None,
                  mass: float | None = None, fromto=None, cls=None,
                  friction=None) -> None:
        """Create a geom and record what it weighs and what role it plays."""
        g = body.add_geom(name=name, type=_GEOM_TYPE[gtype], size=_size3(size))
        if fromto is not None:
            g.fromto = [float(x) for x in fromto]
        else:
            g.pos = _size3(pos)
        _set_euler(g, euler)
        if color is not None:
            g.rgba = rgba(color)
        if cls is not None:
            g.classname = cls
        if friction is not None:
            g.friction = _size3(friction)
        if mass is not None:
            g.mass = float(mass)
            resolved = float(mass)
        else:
            rho = _DEFAULT_DENSITY if density is None else float(density)
            g.density = rho
            resolved = rho * geom_volume(gtype, _size3(size), fromto,
                                         self._mesh_volume)
        self._geom_roles[name] = role
        self._geom_mass[name] = (resolved, body.name)

    def _draw_link(self, geom_name: str, g: dict, r_struct: float,
                   link_l: float, mp: dict | None, link_axis: str = 'z',
                   motor_axis: str | None = None,
                   node: dict | None = None) -> float:
        parent = (self._body_motor.get(self._parent_body.get(node['name'], ''))
                  if node else None)
        r_draw = drawn_radius(g, self._params, r_struct, mp, link_axis,
                              motor_axis, (parent or {}).get('mp'),
                              (parent or {}).get('axis'))
        if abs(r_draw - r_struct) > 1e-12:
            self._link_render[geom_name] = {
                'r_structural_m': round(r_struct, 6),
                'r_drawn_m': round(r_draw, 6),
                'length_m': round(link_l, 6),
            }
        return r_draw

    def _structural_cap(self, g: dict, node: dict, mp: dict | None,
                        link_axis: str, motor_axis: str | None) -> float:
        """Widest a structural member may be (`tree_ops.structural_radius_cap`, then `link_r_max`)."""
        from .tree_ops import structural_radius_cap
        parent = (self._body_motor.get(self._parent_body.get(node['name'], ''))
                  or {})
        cap = structural_radius_cap(mp, motor_axis, parent.get('mp'),
                                    parent.get('axis'), link_axis)
        if 'link_r_max' in g:
            cap = min(cap, float(g['link_r_max']))
        return cap

    def _structural_link_r(self, geom_name: str, g: dict, mass_kg: float,
                           rho: float, length: float, fixed_kg: float = 0.0,
                           cap: float = math.inf) -> float:
        """(radius, density) at which a member weighs its share of the trend mass.

        `fixed_kg` (e.g. an end sphere) is paid first; the member keeps at least 5%
        of the target. Above `cap` the radius is capped and the density raised.
        """
        cyl_kg = mass_kg - fixed_kg
        floored = cyl_kg < 0.05 * mass_kg
        cyl_kg = max(cyl_kg, 0.05 * mass_kg)
        r = math.sqrt(cyl_kg / (rho * math.pi * length))
        capped = r > cap
        if capped:
            # Keep the mass: cap the radius and raise the density.
            r = cap
            member_rho = cyl_kg / (math.pi * r * r * length)
        else:
            member_rho = rho
        law = g.get('link_mass_basis') or {}
        self._structural_links[geom_name] = {
            'class': law.get('segment'),
            'basis': law.get('basis'),
            'segment_root': law.get('segment_root'),
            'share_of_segment': law.get('share'),
            'joint_to_joint_m': round(float(law.get('length_m', length)), 6),
            'lambda_kg_per_m': round(float(law.get('lambda_kg_per_m', 0.0)), 4),
            'target_kg': round(mass_kg, 4),
            'fixed_in_segment_kg': round(fixed_kg, 4),
            'structural_length_m': round(length, 6),
            'class_density_kg_m3': round(rho, 2),
            'density_kg_m3': round(member_rho, 2),
            'r_m': round(r, 6),
            'r_cap_m': round(cap, 6) if math.isfinite(cap) else None,
            'emitted_member_kg': round(member_rho * math.pi * r * r * length, 4),
            'density_raised_at_cap': bool(capped),
            'floored_at_5pct': bool(floored),
        }
        return r, member_rho

    def structural_link_report(self) -> dict | None:
        """Per member of a measured class: the trend target and the radius solved for it."""
        return dict(sorted(self._structural_links.items())) or None

    # ── Geom emitters ──────────────────────────────────────────────────────────

    def _emit_root_geoms(self, body, node: dict) -> None:
        if not self._mesh_name:
            return
        g = body.add_geom(name=f'{node["name"]}_mesh',
                          type=mujoco.mjtGeom.mjGEOM_MESH)
        g.meshname = self._mesh_name
        g.rgba = rgba(node.get('color', '0.5 0.5 0.5 1'))
        rho = float(self._params.get(node.get('density_param', 'torso_rho'), 100.0))
        g.density = rho
        self._geom_roles[f'{node["name"]}_mesh'] = 'torso'
        # Mesh volume is patched in by `mass_report`.
        self._geom_mass[f'{node["name"]}_mesh'] = (('mesh', rho), body.name)

    def _proximal_inset(self, node: dict, link_axis: str, link_sign: int,
                        link_unit, penetration: float) -> float:
        """Where this body's member starts to clear its parent's actuator (`tree_ops.proximal_inset`).

        Skipped (0) for a body with its own `euler`.
        """
        parent = self._body_motor.get(self._parent_body.get(node['name'], ''))
        if parent is None or node.get('euler'):
            return 0.0
        pos = [float(v) for v in _size3(node.get('pos', [0, 0, 0]))]
        # Parent actuator position along this link's axis, from this body's origin.
        along = link_sign * sum((parent['ctr'][i] - pos[i]) * link_unit[i]
                                for i in range(3))
        return proximal_inset(parent['mp'], parent['axis'], along, link_axis,
                              penetration=penetration)

    def _emit_basic_geoms(self, body, node: dict) -> None:
        g = node.get('geometry', {})
        name = node['name']
        p = self._params

        link_axis = str(g.get('link_axis', 'z'))
        link_l = float(g['link_l'])
        link_r = float(g.get('link_r', p.get('link_r', 0.02)))
        link_rho = float(g.get('link_rho', p.get('link_rho', 500.0)))
        link_color = g.get('link_color', p.get('link_color', '0.5 0.5 0.5 1'))
        motor_axis = str(g.get('motor_axis', 'x'))
        # `motor_class: null`: actuator pruned with its joint; the link stays.
        motor_cls = g.get('motor_class')
        mp = motor_props(str(motor_cls), p) if motor_cls else None
        # Drawn link radius uses the class values, before per-link overrides.
        mp_draw = mp

        # Per-link motor overrides; `motor_rho: 0` draws an actuator already
        # charged elsewhere (e.g. a belt-driven knee) without weighing it again.
        if mp is not None and {'motor_r', 'motor_L', 'motor_rho'} & set(g):
            mp = dict(mp)
            for key, prop in (('motor_r', 'r'), ('motor_L', 'L'), ('motor_rho', 'rho')):
                if key in g:
                    mp[prop] = float(g[key])

        # Motor at the link tip (or body origin); base axes only, for mirroring.
        link_base = link_axis.lstrip('+-')
        link_sign = -1 if link_axis.startswith('-') else 1
        link_unit = avec(link_base)
        motor_base = motor_axis.lstrip('+-')
        mot_offset = shift = mot_ctr = mot_euler = None
        if mp is not None:
            axes_align = mp['L'] / 2 if link_base == motor_base else 0
            mot_offset = 0.0 if link_l <= 0 else link_l - axes_align
            # `motor_offset` pulls the actuator back along the link toward the parent.
            mot_offset -= float(g.get('motor_offset', 0.0))
            # `motor_shift` translates the actuator perpendicular to its link.
            shift = [float(v) for v in (g.get('motor_shift') or (0.0, 0.0, 0.0))]
            mot_ctr = tuple(link_sign * v * mot_offset + sv
                            for v, sv in zip(link_unit, shift))
            mot_euler = euler_for_axis(motor_axis)

        # `link_l: 0`: the body is just its actuator, no structural member.
        if link_l > 0:
            def span(r_d: float) -> tuple[float, float]:
                # Trim the member at its actuators (`tree_ops.link_span`).
                if not p.get('link_motor_inset', True):
                    return 0.0, link_l
                pen = float(p.get('link_motor_penetration', LINK_MOTOR_PENETRATION_M))
                return link_span(
                    link_l, mot_offset if mot_offset is not None else link_l, mp,
                    link_axis, motor_axis, shift, r_d, penetration=pen,
                    t0=self._proximal_inset(node, link_axis, link_sign,
                                            link_unit, pen))

            mass_kg = g.get('link_mass_kg')
            if mass_kg is not None:
                # Radius is solved from the class mass; start from the cap.
                cap = self._structural_cap(g, node, mp_draw, link_axis, motor_axis)
                link_r = cap if math.isfinite(cap) else link_r
            else:
                self.stated_members.append(f'{name}_link')
            r_draw = self._draw_link(f'{name}_link', g, link_r, link_l, mp_draw,
                                     link_axis, motor_axis, node)
            t0, t1 = span(r_draw)
            if mass_kg is not None:
                link_r, link_rho = self._structural_link_r(
                    f'{name}_link', g, float(mass_kg), link_rho,
                    max(t1 - t0, 1e-6), cap=cap)
                self._link_render.pop(f'{name}_link', None)
                r_draw = self._draw_link(f'{name}_link', g, link_r, link_l, mp_draw,
                                         link_axis, motor_axis, node)
                t0, t1 = span(r_draw)
            seg = max(t1 - t0, 1e-6)
            if seg < link_l - 1e-9:
                self._link_inset[f'{name}_link'] = {
                    'joint_to_joint_m': round(link_l, 6),
                    'structural_m': round(seg, 6),
                    'removed_m': round(link_l - seg, 6),
                    'removed_frac': round(1.0 - seg / link_l, 4),
                }
            self._add_geom(
                body, f'{name}_link', 'cylinder', [r_draw, seg / 2], 'link',
                pos=axis_center(link_axis, t0 + seg / 2),
                euler=euler_for_axis(link_axis), color=link_color,
                density=slim_density(link_rho, link_r, r_draw))

        if mp is None:
            return

        self._add_geom(body, f'{name}_motor', 'cylinder', [mp['r'], mp['L'] / 2],
                       'motor', pos=mot_ctr, euler=mot_euler,
                       color=mp['color'], density=mp['rho'])
        self._motor_records[f'{name}_motor'] = {
            'name': f'{name}_motor', 'body': name,
            'motor_class': str(motor_cls), 'drives': g.get('motor_for'),
            'r': float(mp['r']), 'L': float(mp['L']),
        }
        self._body_motor[name] = {'mp': mp, 'axis': motor_axis,
                                  'ctr': [float(v) for v in mot_ctr]}
        # Detail ring — visual only, weightless.
        self._add_geom(body, f'{name}_detail', 'cylinder',
                       [mp['r'] * 0.8, mp['L'] * 0.55], 'detail',
                       pos=mot_ctr, euler=mot_euler, color=mp['detail'],
                       density=0.0, cls=self._visual)

    def _emit_sphere_geoms(self, body, node: dict) -> None:
        g = node.get('geometry', {})
        name = node['name']
        p = self._params

        link_axis = str(g.get('link_axis', 'z'))
        link_l = float(g['link_l'])
        link_r = float(g.get('link_r', p.get('link_r', 0.02)))
        link_rho = float(g.get('link_rho', p.get('link_rho', 500.0)))
        link_color = g.get('link_color', p.get('link_color', '0.5 0.5 0.5 1'))
        sphere_r = float(g['sphere_r'])
        sphere_dir = int(g.get('sphere_dir', 1))
        sphere_color = g.get('sphere_color', link_color)

        av = avec(link_axis)
        cyl_ctr = tuple(v * sphere_dir * link_l / 2 for v in av)
        sph_ctr = tuple(v * sphere_dir * link_l for v in av)

        # No motor of its own: drawn radius keys off the joint's motor class.
        jcls = str((node.get('joint') or {}).get('motor_class', '')
                   or g.get('motor_class', ''))
        mp = motor_props(jcls, p) if jcls else None
        # The cylinder's own density; the sphere keeps `sphere_rho` / `link_rho`.
        member_rho = link_rho
        if g.get('link_mass_kg') is not None:
            # The end sphere is in the same segment, so it is paid for first.
            sphere_kg = (float(g.get('sphere_rho', link_rho))
                         * 4.0 / 3.0 * math.pi * sphere_r ** 3)
            link_r, member_rho = self._structural_link_r(
                f'{name}_link', g, float(g['link_mass_kg']), link_rho, link_l,
                fixed_kg=sphere_kg,
                cap=self._structural_cap(g, node, mp, link_axis,
                                         str(g.get('motor_axis', link_axis))))
        elif link_l > 0:
            self.stated_members.append(f'{name}_link')
        r_draw = self._draw_link(f'{name}_link', g, link_r, link_l, mp,
                                 link_axis, str(g.get('motor_axis', link_axis)),
                                 node)

        self._add_geom(body, f'{name}_link', 'cylinder', [r_draw, link_l / 2],
                       'link', pos=cyl_ctr, euler=euler_for_axis(link_axis),
                       color=link_color,
                       density=slim_density(member_rho, link_r, r_draw))

        # Head/hand spheres take their own measured segment density.
        sphere_rho = float(g.get('sphere_rho', link_rho))
        role = ('head' if 'head' in name else 'hand' if 'hand' in name else 'link')
        # Optional per-geom friction from the geometry block.
        friction = ([float(v) for v in g['friction']] if 'friction' in g else None)
        self._add_geom(body, f'{name}_sphere', 'sphere', [sphere_r], role,
                       pos=sph_ctr, color=sphere_color, density=sphere_rho,
                       friction=friction)

    def _emit_foot_geoms(self, body, node: dict) -> None:
        g = node.get('geometry', {})
        name = node['name']
        p = self._params

        foot_l = float(g['foot_l'])
        foot_w = float(g['foot_w'])
        foot_h = float(g.get('foot_h', p.get('foot_h', 0.025)))
        offset_x = float(g.get('foot_offset_x', p.get('foot_offset_x', 0.03)))
        color = g.get('color', p.get('foot_color', '0.5 0.5 0.5 1'))
        link_rho = float(p.get('link_rho', 500.0))
        mot_cls = str((node.get('joint') or {}).get('motor_class', 'S'))
        mot_r = float(p.get(f'{mot_cls}_motor_r', 0.04))

        # A foot is a sole plate on a riser (ankle axis down to the plate).
        box_h = float(g.get('foot_plate_h', p.get('foot_plate_h', 0.015)))
        # Riser fills `foot_h` minus the plate, but at least the ankle motor radius.
        # Keep in step with tree.yaml's `max(foot_h, mot_r + plate)` stance height.
        riser_h = max(foot_h - box_h, mot_r)
        riser_r = float(g.get('riser_r', mot_r))
        box_rho = float(g.get('foot_rho', p.get('foot_rho', link_rho)))
        # Mass uses the whole foot envelope, which is what `foot_rho` was measured on.
        box_mass = box_rho * foot_l * foot_w * foot_h

        # Weightless: `box_mass` already covers the riser's volume.
        self._add_geom(body, f'{name}_riser', 'cylinder', [riser_r, riser_h / 2],
                       'foot', pos=(0, 0, -riser_h / 2),
                       color=g.get('riser_color', color), density=0.0)
        self._add_geom(body, f'{name}_box', 'box',
                       [foot_l / 2, foot_w / 2, box_h / 2], 'foot',
                       pos=(offset_x, 0, -riser_h - box_h / 2), color=color,
                       mass=box_mass)
        self._foot_render[f'{name}_box'] = {
            'ankle_to_sole_m': round(riser_h + box_h, 6),
            'foot_h_measured_m': round(foot_h, 6),
            'ankle_motor_r_m': round(mot_r, 6),
            'riser_height_m': round(riser_h, 6),
            'riser_radius_m': round(riser_r, 6),
            'plate_thickness_m': round(box_h, 6),
            'plate_rho_kg_m3': round(box_rho, 2),
            'plate_mass_kg': round(box_mass, 6),
            # Negative when the ankle motor is taller than the stated foot depth.
            'ankle_to_sole_deficit_m': round(foot_h - (riser_h + box_h), 6),
        }

        bottom_z = -(riser_h + box_h)
        sites_xy = [
            (foot_l / 2 + offset_x,  foot_w / 2),
            (foot_l / 2 + offset_x, -foot_w / 2),
            (-foot_l / 2 + offset_x,  foot_w / 2),
            (-foot_l / 2 + offset_x, -foot_w / 2),
            (foot_l / 2 + offset_x,  0),
        ]
        names = [f'{name}_sphere_{i}' for i in range(4)] + [f'{name}_toe_sphere']
        for sname, (sx, sy) in zip(names, sites_xy):
            body.add_site(name=sname, pos=[sx, sy, bottom_z], group=3)

    def _emit_geom(self, body, node: dict) -> None:
        g = node.get('geometry', {})
        name = node['name']
        shape = g.get('shape', 'cylinder')
        color = g.get('color', '0.5 0.5 0.5 1')
        rho = float(g.get('rho', self._params.get('motor_rho', 2700.0)))
        r = float(g['r'])
        mass = float(g['mass']) if 'mass' in g else None

        # Default role `other`: raw primitives are not counted as limb structure.
        role = str(g.get('role', 'other'))

        # `decorative: true`: drawn, weightless and non-colliding.
        kw = {}
        if bool(g.get('decorative', False)):
            rho, mass, kw = 0.0, None, {'cls': self._visual}

        if shape == 'sphere':
            self._add_geom(body, f'{name}_geom', 'sphere', [r], role,
                           color=color, density=rho, mass=mass, **kw)
        elif shape in ('cylinder', 'capsule'):
            if 'fromto' in g:
                self._add_geom(body, f'{name}_geom', shape, [r], role,
                               fromto=[float(x) for x in g['fromto']],
                               color=color, density=rho, mass=mass, **kw)
            else:
                l = float(g['l'])
                self._add_geom(body, f'{name}_geom', shape, [r, l / 2], role,
                               euler=euler_for_axis(str(g.get('axis', 'z'))),
                               color=color, density=rho, mass=mass, **kw)
        elif shape == 'box':
            lx = float(g.get('lx', r))
            ly = float(g.get('ly', r))
            lz = float(g.get('lz', g.get('l', r)))
            self._add_geom(body, f'{name}_geom', 'box', [lx / 2, ly / 2, lz / 2],
                           role, color=color, density=rho, mass=mass, **kw)

    # ── Contacts and sensors ───────────────────────────────────────────────────

    def _collect_body_pairs(self, node: dict, parent: str | None = None,
                            grandparent: str | None = None) -> list[tuple[str, str]]:
        """Parent and grandparent body pairs to exclude from contact.

        Parent pairs are also covered by `filterparent`; grandparent pairs are not.
        """
        pairs: list[tuple[str, str]] = []
        name = self._module_roots.get(node['name'], node['name'])
        if 'module' in node:
            # Only the seam to a module is excluded; its internals are its own.
            if parent:
                pairs.append((parent, name))
            return pairs
        if parent:
            pairs.append((parent, name))
        if grandparent:
            pairs.append((grandparent, name))
        for child in node.get('children', []) or []:
            pairs.extend(self._collect_body_pairs(child, name, parent))
        return pairs

    def _emit_contact_excludes(self, tree: dict) -> None:
        for b1, b2 in self._collect_body_pairs(tree):
            self.spec.add_exclude(name=f'{b1}__{b2}', bodyname1=b1, bodyname2=b2)

    def _emit_sensor(self, s: dict) -> None:
        stype = str(s['type'])
        if stype not in _SENSOR_TYPE:
            raise ValueError(f'unsupported sensor type: {stype}')
        kind, objtype = _SENSOR_TYPE[stype]
        objname = s.get('site') or s.get('body') or s.get('objname')
        self.spec.add_sensor(name=s.get('name', stype), type=kind,
                             objtype=objtype, objname=str(objname))

    # ── Compilation ────────────────────────────────────────────────────────────

    def compile(self, mesh_volume_m3: float | None = None) -> mujoco.MjModel:
        """Compile, then record actuator world poses from `mj_forward` at the neutral pose."""
        self._mesh_volume = mesh_volume_m3
        self._model = self.spec.compile()
        model = self._model
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        # Kept for `packing.structure_overlap`.
        self._model_data = data

        for name, rec in self._motor_records.items():
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid < 0:
                continue
            size = model.geom_size[gid]
            rec['center'] = tuple(float(v) for v in data.geom_xpos[gid])
            rec['R'] = [float(v) for v in data.geom_xmat[gid]]
            # Cylinder axis is local z: half-extents (r, r, L/2).
            rec['half'] = (float(size[0]), float(size[0]), float(size[1]))
            rec.setdefault('r', float(size[0]))
            rec.setdefault('L', float(size[1]) * 2.0)
            bid = model.geom_bodyid[gid]
            rec['path'] = self._body_path(model, bid)
        return model

    @staticmethod
    def _body_path(model, bid: int) -> tuple:
        """Body chain from the root (used to find consecutive actuators)."""
        chain = []
        while bid > 0:
            chain.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid))
            bid = model.body_parentid[bid]
        return tuple(reversed(chain))

    @property
    def motor_geoms(self) -> list:
        return list(self._motor_records.values())

    @property
    def model(self):
        return self._model

    @property
    def model_data(self):
        return getattr(self, '_model_data', None)

    # ── Mass accounting ────────────────────────────────────────────────────────

    def mass_report(self, mesh_volume_m3: float | None = None) -> dict:
        """Emitted mass by role, body and geom; the total is checked against the compiled model."""
        by_role: dict[str, float] = {}
        by_body: dict[str, float] = {}
        per_geom: dict[str, float] = {}

        for name, (mass, bname) in self._geom_mass.items():
            if isinstance(mass, tuple) and mass[0] == 'mesh':
                mass = float(mass[1]) * float(mesh_volume_m3 or 0.0)
            mass = float(mass)
            if mass <= 0:
                continue
            role = self._geom_roles.get(name, 'other')
            if role == 'detail':
                continue
            by_role[role] = by_role.get(role, 0.0) + mass
            by_body[bname] = by_body.get(bname, 0.0) + mass
            per_geom[name] = mass

        total = sum(by_role.values())
        report = {'by_role': by_role, 'by_body': by_body, 'per_geom': per_geom,
                  'total': total}
        if self._model is not None:
            compiled = float(self._model.body_mass.sum())
            report['compiled_total_kg'] = compiled
            if total > 0 and abs(compiled - total) / total > 0.01:
                report['disagreement'] = (
                    f'role accounting totals {total:.4f} kg but the compiled '
                    f'model weighs {compiled:.4f} kg')
        return report

    # ── Reports ────────────────────────────────────────────────────────────────

    def link_render_report(self, link_rho: float | None = None) -> dict | None:
        """How far drawn links were slimmed; their XML density is raised to keep mass."""
        if not self._link_render:
            return None
        rho = float(link_rho if link_rho is not None
                    else self._params.get('link_rho', 0.0))
        ratios = [e['r_drawn_m'] / e['r_structural_m']
                  for e in self._link_render.values() if e['r_structural_m'] > 0]
        return {
            'n_links_slimmed': len(self._link_render),
            'structural_link_rho_kg_m3': round(rho, 2),
            'drawn_radius_over_structural': [round(min(ratios), 4),
                                             round(max(ratios), 4)],
            'emitted_density_kg_m3': [round(rho / max(ratios) ** 2, 1),
                                      round(rho / min(ratios) ** 2, 1)],
            'mass_effect': 'none — density compensates exactly; ρ·π·r²·L is invariant',
            'inertia_effect': ('slimmer cylinders: transverse inertia within a few '
                               'percent, axial inertia scales with (r_drawn/r_struct)² '
                               'and is dominated by the joint armature anyway'),
            'links': dict(sorted(self._link_render.items())),
        }

    def link_inset_report(self) -> dict | None:
        """Structural length each link gave up where it meets its actuators."""
        if not self._link_inset:
            return None
        removed = [e['removed_frac'] for e in self._link_inset.values()]
        return {
            'n_links_inset': len(self._link_inset),
            'removed_fraction_of_length': [round(min(removed), 4),
                                           round(max(removed), 4)],
            'rule': ('a structural link stops at its actuator housing face plus '
                     'a bolt flange, rather than running to the joint centre '
                     'through the motor — see core/tree_ops.py:link_span'),
            'penetration_m': float(self._params.get(
                'link_motor_penetration', LINK_MOTOR_PENETRATION_M)),
            'scope': ('the DISTAL end only: this body\'s own actuator shares its '
                      'frame so the overlap is fixed, while the actuator at the '
                      'proximal end is on the far side of a joint and how much it '
                      'interpenetrates depends on the angle — core/packing.py '
                      'governs that pair instead'),
            'links': dict(sorted(self._link_inset.items())),
        }

    def foot_render_report(self) -> dict | None:
        return dict(sorted(self._foot_render.items())) or None

    def to_string(self) -> str:
        return self.spec.to_xml()

    @staticmethod
    def roundtrip_check(path: Path, expected_kg: float) -> dict:
        """Reload the written XML and report mass drift from 6-significant-figure rounding."""
        model = mujoco.MjModel.from_xml_path(str(path))
        got = float(model.body_mass.sum())
        rel = abs(got - expected_kg) / expected_kg if expected_kg else 0.0
        return {
            'written_mass_kg': round(got, 6),
            'in_memory_mass_kg': round(expected_kg, 6),
            'relative_drift': float(f'{rel:.3g}'),
            'note': ('MuJoCo writes 6 significant figures; this is the cost of '
                     'that rounding, not a modelling difference'),
        }
