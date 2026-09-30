"""Abstract base class for a root-body mesh written in Python (`mesh.py`).

Subclasses set `loft_axis` and implement `get_layers(params)`; lofting, inertia
and export are handled by `LayeredMeshBuilder`. The shipped robots use
`mesh.yaml` (see `mesh_spec`) instead.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import trimesh

from .layered_mesh import LayeredMeshBuilder, chamfered_rect_polygon, inertial_properties


# ── Layer dataclass ────────────────────────────────────────────────────────────

@dataclass
class Layer:
    """A single cross-section in a loft mesh.

    pos      : position along the loft axis
    vertices : 2-D polygon as [[u, v], ...] ordered CCW when viewed from +axis
    """
    pos: float
    vertices: list[list[float]]


# ── Convenience constructors ───────────────────────────────────────────────────

def rect_layer(pos: float, lx: float, ly: float,
               chamfer: float = 0.15,
               cx: float = 0.0, cy: float = 0.0) -> Layer:
    """Create a Layer with a chamfered-rectangle cross-section."""
    return Layer(pos=pos, vertices=chamfered_rect_polygon(lx, ly, chamfer, cx, cy))


def polygon_layer(pos: float, vertices: list[list[float]]) -> Layer:
    """Create a Layer from an explicit list of 2-D [u, v] vertices."""
    return Layer(pos=pos, vertices=vertices)


# ── Base class ─────────────────────────────────────────────────────────────────

class RootMesh(ABC):
    """Parametric root-body mesh; all layers must have the same vertex count.

    Class vars: `loft_axis` ('x'/'y'/'z'), `name` (file prefix), `density_param`
    (params key for density, kg/m³).
    """

    loft_axis:     str = 'z'
    name:          str = 'root'
    density_param: str = 'torso_rho'

    @abstractmethod
    def get_layers(self, params: dict) -> list[Layer]:
        """Return loft cross-sections computed from params."""
        ...

    # ── Private helpers ────────────────────────────────────────────────────────

    def _builder(self, params: dict) -> LayeredMeshBuilder:
        layers = self.get_layers(params)
        density = params.get(self.density_param, params.get('torso_rho', 100.0))
        return LayeredMeshBuilder(
            axis=self.loft_axis,
            planes=[{'position': l.pos, 'vertices': l.vertices} for l in layers],
            density=density,
            name=self.name,
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def build(self, params: dict) -> trimesh.Trimesh:
        """Return a watertight trimesh.Trimesh for the given parameters."""
        return self._builder(params).build()

    def build_and_save(self, params: dict, output_dir) -> tuple[trimesh.Trimesh, dict]:
        """Write `{name}_mesh.stl` and `{name}_inertial_properties.yaml`; return (mesh, props)."""
        return self._builder(params).build_and_save(output_dir)
