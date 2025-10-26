"""discrete1.

Top-level package for the discrete1 library.

This package exports a small convenience API for creating spatial/energy/
angular grids and accessing material definitions. Importing the package
will expose commonly-used helpers so downstream code can do::

    import discrete1

to have access to the main functionality.
"""

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
