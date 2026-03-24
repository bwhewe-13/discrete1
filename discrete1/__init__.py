"""discrete1.

Top-level package for the discrete1 library.

This package exports a small convenience API for creating spatial/energy/
angular grids and accessing material definitions. Importing the package
will expose commonly-used helpers so downstream code can do::

    import discrete1

to have access to the main functionality.
"""

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from discrete1 import boundary1d, external1d  # noqa: F401

# Creating medium maps, energy grids, and angular grids
from discrete1.main import (  # noqa: F401
    angular_x,
    energy_grid,
    energy_velocity,
    spatial1d,
)

# Creating materials
from discrete1.materials import materials  # noqa: F401


class Geometry(IntEnum):
    """Supported one-dimensional transport geometries."""

    SLAB = 1
    SPHERE = 2


@dataclass
class MaterialData:
    """Container for material cross-section data."""

    xs_total: np.ndarray  # (n_materials, n_groups)
    xs_scatter: np.ndarray  # (n_materials, n_groups, n_groups) or (..., L+1)
    xs_fission: np.ndarray  # (n_materials, n_groups)
    chi: np.ndarray | None = None


@dataclass
class GeometryData:
    """Container for spatial mesh and boundary-condition metadata."""

    medium_map: np.ndarray  # (cells_x,) int32
    delta_x: np.ndarray  # (cells_x,)
    bc_x: list
    geometry: Geometry = Geometry.SLAB


@dataclass
class AngularData:
    """Container for angular quadrature and optional Legendre data."""

    angle_x: np.ndarray  # (angles,)
    angle_w: np.ndarray  # (angles,)
    P: np.ndarray | None = None  # (L+1, angles), computed on first use
    P_weights: np.ndarray | None = None  # (L+1, angles)
