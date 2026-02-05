"""Utilities for building grids and basic problem parameters.

This module collects small helper functions used to build angular and
energy quadratures, convert energy grids to velocities, construct time
stepping grids, and build 1-D spatial material maps. The helpers are
lightweight and intended for use by example scripts and tests.

Public helpers
- angular_x
- energy_grid
- energy_velocity
- gamma_time_steps
- spatial1d
"""

import os
from importlib.resources import files

import numpy as np

import discrete1.constants as const
from discrete1.utils.hybrid import energy_coarse_index

DATA_PATH = str(files("discrete1").joinpath("sources/energy"))
if not os.path.exists(DATA_PATH):
    DATA_PATH = str(files("discrete1").joinpath("../tests"))


def angular_x(angles, bc_x=[0, 0]):
    """Generate angular ordinates and normalized weights.

    Uses Gauss-Legendre quadrature to compute angular ordinates (mu)
    and weights for the specified number of angles. If reflective
    boundary conditions are present (``bc_x``), the ordering of
    ordinates and weights is adjusted so that the incoming directions
    correspond to the solver's expected ordering.

    Parameters
    ----------
    angles : int
        Number of angular ordinates (n).
    bc_x : sequence of two ints, optional
        Boundary condition indicators for left/right (default [0, 0]).

    Returns
    -------
    tuple
        (angle_x, angle_w) where ``angle_x`` is array of ordinates and
        ``angle_w`` is the corresponding normalized weights.
    """

    angle_x, angle_w = np.polynomial.legendre.leggauss(angles)
    angle_w /= np.sum(angle_w)

    # Ordering for reflective boundaries
    if np.sum(bc_x) > 0.0:
        if bc_x == [1, 0]:
            idx = angle_x.argsort()
        elif bc_x == [0, 1]:
            idx = angle_x.argsort()[::-1]
        angle_x = angle_x[idx].copy()
        angle_w = angle_w[idx].copy()

    return angle_x, angle_w


def energy_grid(grid, groups_fine, groups_coarse=None, optimize=True):
    """Build energy grid edges and index mapping for coarse/fine grids.

    The function loads predefined energy grids shipped with the
    package and returns the energy edges (MeV) and the array mapping
    problem group indices to locations within the fine grid. When
    ``groups_coarse`` is provided the function also returns the coarse
    grid index mapping.

    Parameters
    ----------
    grid : int
        Choice of stock grid (supported: 87, 361, 618).
    groups_fine : int
        Number of fine groups in the problem.
    groups_coarse : int, optional
        Number of coarse groups to compute an additional mapping.
    optimize : bool, optional
        If True, attempt to load precomputed index mappings from
        packaged data; otherwise compute indices on-the-fly.

    Returns
    -------
    edges_g : numpy.ndarray
        Energy bin edges in MeV (length grid + 1).
    edges_gidx_fine : numpy.ndarray
        Integer indices mapping fine-group boundaries into the
        chosen energy grid.
    edges_gidx_coarse : numpy.ndarray, optional
        (Only returned when ``groups_coarse`` is not None.) Coarse
        group index mapping.
    """
    # Create energy grid
    if grid in [87, 361, 618]:
        edges_g = np.load(os.path.join(DATA_PATH, "energy_grids.npz"))[str(grid)]

        # Collect grid boundary indices
        fgrid = str(grid).zfill(3)
        edges_data = np.load(os.path.join(DATA_PATH, f"G{fgrid}_grid_index.npz"))
    # Calculate the indices for the specific fine grid
    if (grid in [87, 361, 618]) and (optimize):
        # Predefined fine grid index
        try:
            label_fine = str(groups_fine).zfill(3)
            edges_gidx_fine = edges_data[label_fine].copy()

        except KeyError:
            edges_gidx_fine = energy_coarse_index(len(edges_g) - 1, groups_fine)

    else:
        edges_gidx_fine = energy_coarse_index(len(edges_g) - 1, groups_fine)

    # Convert to correct type
    edges_gidx_fine = edges_gidx_fine.astype(np.int32)

    if groups_coarse is None:
        return edges_g, edges_gidx_fine

    # Calculate the indices for the specific coarse grid
    if (grid in [87, 361, 618]) and (optimize):
        # Predefined coarse grid index
        try:
            label_coarse = str(groups_coarse).zfill(3)
            edges_gidx_coarse = edges_data[label_coarse].copy()

        except KeyError:
            edges_gidx_coarse = energy_coarse_index(groups_fine, groups_coarse)

    else:
        edges_gidx_coarse = energy_coarse_index(groups_fine, groups_coarse)

    # Convert to correct type
    edges_gidx_coarse = edges_gidx_coarse.astype(np.int32)

    return edges_g, edges_gidx_fine, edges_gidx_coarse


def energy_velocity(groups, edges_g=None):
    """Compute particle speeds (cm/s) at group centers from energy edges.

    Parameters
    ----------
    groups : int
        Number of energy groups. Used only if ``edges_g`` is None.
    edges_g : array_like, optional
        Energy group edges in MeV. If omitted, a unit vector is
        returned for convenience.

    Returns
    -------
    numpy.ndarray
        Speeds at group center energies in cm/s.
    """

    if np.all(edges_g is None):
        return np.ones((groups,))

    centers_gg = 0.5 * (edges_g[1:] + edges_g[:-1])
    gamma = (const.EV_TO_JOULES * centers_gg) / (
        const.MASS_NEUTRON * const.LIGHT_SPEED**2
    ) + 1
    velocity = const.LIGHT_SPEED / gamma * np.sqrt(gamma**2 - 1) * 100
    return velocity


def gamma_time_steps(edges_t, gamma=0.5, half_step=True):
    """Insert intermediate (gamma) time steps between existing edges.

    This helper takes a time-edge array and returns a combined array
    that includes half-steps (or gamma-weighted sub-steps) between
    original edges. It's used by time integrators that require
    sub-step evaluations (for example TR-BDF2).

    Parameters
    ----------
    edges_t : array_like
        Time edge array of length (n_steps + 1).
    gamma : float, optional
        Fraction used to compute intermediate times (default 0.5).
    half_step : bool, optional
        If True, the first intermediate step is set to the exact
        midpoint between the surrounding times.

    Returns
    -------
    numpy.ndarray
        Combined time array of length (n_steps * 2 + 1).
    """

    if gamma == 0.5:
        half_steps = 0.5 * (edges_t[1:] + edges_t[:-1])
    else:
        half_steps = edges_t[:-1] + np.diff(edges_t) * gamma
    # Combine half steps
    combined_steps = np.sort(np.concatenate((edges_t, half_steps)))
    if half_step:
        combined_steps[1] = 0.5 * (combined_steps[0] + combined_steps[2])
    return combined_steps


def spatial1d(layers, edges_x, labels=False, check=True):
    """Create one-dimensional medium map.

    :param layers: list of lists where each layer is a new material. A
        layer is comprised of an index (int), material name (str), and
        the width (str) in the form [index, material, width]. The width
        is the starting and ending points of the material (in cm)
        separated by a dash. If there are multiple regions, a comma can
        separate them. I.E. layer = [0, "plutonium", "0 - 2, 3 - 4"].
    :param edges_x: Array of length I + 1 with the location of the cell edges
    :return: One-dimensional array of length I, identifying the locations
        of the materials
    """
    if labels:
        # Initialize label map
        medium_map = -1 * np.ones((len(edges_x) - 1))
    else:
        # Initialize medium_map
        medium_map = -1 * np.ones((len(edges_x) - 1), dtype=np.int32)
    # Iterate over all layers
    for layer in layers:
        for region in layer[2].split(","):
            start, stop = region.split("-")
            idx1 = np.argmin(np.fabs(float(start) - edges_x))
            idx2 = np.argmin(np.fabs(float(stop) - edges_x))
            medium_map[idx1:idx2] = layer[0]
    # Verify all cells are filled
    if check:
        assert np.all(medium_map != -1)

    return medium_map
