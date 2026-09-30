"""A root-body mesh described by `mesh.yaml`: parallel cross-sections lofted along one axis::

    name:          torso        # → torso_mesh.stl
    axis:          z            # the sections stack along this axis
    density_param: torso_rho    # which parameter holds the density
    chamfer:       0.15         # default corner chamfer for `lx`/`ly` sections
    layers:
      - {pos: "${-torso_length}", lx: "${torso_lx}", ly: "${torso_ly}"}
      - {pos: 0, lx: "${shoulder_depth}", ly: "${shoulder_width}", cx: "${offset}"}

A layer is either `lx`/`ly` (chamfered rectangle, optional `cx`/`cy`, `chamfer`)
or `polygon` (explicit `[[u, v], ...]`, optional `inset`). Values are `${expr}`s.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .expr_eval import resolve
from .layered_mesh import LayeredMeshBuilder, inset_polygon, rect_outline

#: Keys a layer may carry; anything else is rejected as a typo.
_LAYER_KEYS = {'pos', 'lx', 'ly', 'cx', 'cy', 'chamfer', 'polygon', 'inset'}


class MeshSpec:
    """A `mesh.yaml`, resolved against parameters and built into an STL."""

    def __init__(self, spec: dict, source: Path | None = None) -> None:
        self.source = source
        self.name = str(spec.get('name', 'root'))
        self.axis = str(spec.get('axis', 'z'))
        self.density_param = str(spec.get('density_param', 'torso_rho'))
        self.chamfer = float(spec.get('chamfer', 0.15))
        self.layers = list(spec.get('layers') or [])
        if len(self.layers) < 2:
            raise ValueError(
                f"{source or 'mesh.yaml'}: a lofted mesh needs at least 2 layers, "
                f"got {len(self.layers)}")
        for layer in self.layers:
            unknown = set(layer) - _LAYER_KEYS
            if unknown:
                raise ValueError(
                    f"{source or 'mesh.yaml'}: unknown layer key(s) "
                    f"{', '.join(sorted(unknown))} — expected any of "
                    f"{', '.join(sorted(_LAYER_KEYS))}")

    @classmethod
    def load(cls, path: Path) -> 'MeshSpec':
        return cls(yaml.safe_load(path.read_text()) or {}, source=path)

    # ── Resolution ─────────────────────────────────────────────────────────────

    @staticmethod
    def _val(layer: dict, key: str, params: dict, default: Any = None) -> Any:
        if key not in layer:
            return default
        return resolve(layer[key], params)

    def planes(self, params: dict) -> list[dict]:
        """The resolved cross-sections, in the form LayeredMeshBuilder wants."""
        out = []
        for i, layer in enumerate(self.layers):
            pos = self._val(layer, 'pos', params)
            if pos is None:
                raise ValueError(f'{self.source}: layer {i} has no `pos`')
            if 'polygon' in layer:
                verts = [[float(u), float(v)] for u, v in self._val(layer, 'polygon', params)]
                inset = float(self._val(layer, 'inset', params, 0.0) or 0.0)
                if inset:
                    verts = inset_polygon(verts, inset)
            elif 'lx' in layer and 'ly' in layer:
                verts = rect_outline(
                    float(self._val(layer, 'lx', params)),
                    float(self._val(layer, 'ly', params)),
                    float(self._val(layer, 'chamfer', params, self.chamfer)),
                    float(self._val(layer, 'cx', params, 0.0) or 0.0),
                    float(self._val(layer, 'cy', params, 0.0) or 0.0))
            else:
                raise ValueError(
                    f'{self.source}: layer {i} states neither `polygon` nor `lx`+`ly`')
            out.append({'position': float(pos), 'vertices': verts})
        return out

    # ── Build ──────────────────────────────────────────────────────────────────

    def build_and_save(self, params: dict, output_dir) -> tuple:
        """Write `<name>_mesh.stl` and its inertial properties; same interface as `RootMesh`."""
        density = float(params.get(self.density_param,
                                   params.get('torso_rho', 1000.0)))
        builder = LayeredMeshBuilder(axis=self.axis, planes=self.planes(params),
                                     density=density, name=self.name)
        return builder.build_and_save(Path(output_dir))
