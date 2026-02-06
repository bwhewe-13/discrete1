"""Fixed source 1D transport problems.

This module provides solvers for fixed source (non-fissile) 1D transport problems
using discrete ordinates methods. It includes source iteration and dynamic mode
decomposition techniques.
"""

import numpy as np

from discrete1 import multi_group as mg
from discrete1 import tools


def source_iteration(
    xs_total,
    xs_scatter,
    xs_fission,
    external,
    boundary,
    medium_map,
    delta_x,
    angle_x,
    angle_w,
    bc_x,
    chi=None,
    geometry=1,
    angular=False,
    edges=0,
):
    """Solve fixed source problem using source iteration.

    Uses the multi-group source iteration method to solve for scalar flux
    (and optionally angular flux or edge fluxes) in a fixed source transport
    problem.

    Parameters
    ----------
    xs_total : numpy.ndarray
        Total macroscopic cross section (cells, groups).
    xs_scatter : numpy.ndarray
        Scattering cross section (cells, groups).
    xs_fission : numpy.ndarray
        Fission cross section (cells, groups).
    external : numpy.ndarray
        External source (steps/1, cells, angles, groups).
    boundary : numpy.ndarray
        Boundary conditions (steps/1, 2, angles, groups).
    medium_map : numpy.ndarray
        Material region map (cells,).
    delta_x : numpy.ndarray
        Cell widths (cells,).
    angle_x : numpy.ndarray
        Discrete ordinates (angles,).
    angle_w : numpy.ndarray
        Quadrature weights (angles,).
    bc_x : int
        Boundary condition type (0=vacuum, 1=reflective).
    geometry : int, optional
        Geometry type (1=slab, 2=cylindrical, 3=spherical).
    angular : bool, optional
        If True, return angular flux instead of scalar.
    edges : int, optional
        If 1, return fluxes at cell edges; if 0, at cell centers.

    Returns
    -------
    numpy.ndarray
        Scalar flux at cell centers (cells, groups) if angular=False and
        edges=0, otherwise returns angular or edge flux as specified.
    """

    # Initialize flux
    cells_x = medium_map.shape[0]
    groups = xs_total.shape[1]
    flux_old = np.zeros((cells_x, groups))

    # Combine scattering and fission
    xs_matrix = tools.transfer_matrix(xs_scatter, xs_fission, chi)

    # Run source iteration for scalar flux centers
    flux = mg.source_iteration(
        flux_old,
        xs_total,
        xs_matrix,
        external,
        boundary,
        medium_map,
        delta_x,
        angle_x,
        angle_w,
        bc_x,
        geometry,
    )

    if (angular is False) and (edges == 0):
        return flux

    # For angular flux or scalar flux edges
    return known_source_calculation(
        flux,
        xs_total,
        xs_matrix,
        external,
        boundary,
        medium_map,
        delta_x,
        angle_x,
        angle_w,
        bc_x,
        geometry,
        angular,
        edges,
    )


def dynamic_mode_decomp(
    xs_total,
    xs_scatter,
    xs_fission,
    external,
    boundary,
    medium_map,
    delta_x,
    angle_x,
    angle_w,
    bc_x,
    chi=None,
    geometry=1,
    angular=False,
    edges=0,
    R=2,
    K=10,
):
    """Solve fixed source problem using dynamic mode decomposition.

    Uses dynamic mode decomposition for reduced-order approximation of scalar
    flux in fixed source transport problems.

    Parameters
    ----------
    xs_total : numpy.ndarray
        Total macroscopic cross section (cells, groups).
    xs_scatter : numpy.ndarray
        Scattering cross section (cells, groups).
    xs_fission : numpy.ndarray
        Fission cross section (cells, groups).
    external : numpy.ndarray
        External source (steps/1, cells, angles, groups).
    boundary : numpy.ndarray
        Boundary conditions (steps/1, 2, angles, groups).
    medium_map : numpy.ndarray
        Material region map (cells,).
    delta_x : numpy.ndarray
        Cell widths (cells,).
    angle_x : numpy.ndarray
        Discrete ordinates (angles,).
    angle_w : numpy.ndarray
        Quadrature weights (angles,).
    bc_x : int
        Boundary condition type (0=vacuum, 1=reflective).
    geometry : int, optional
        Geometry type (1=slab, 2=cylindrical, 3=spherical).
    angular : bool, optional
        If True, return angular flux instead of scalar.
    edges : int, optional
        If 1, return fluxes at cell edges; if 0, at cell centers.
    R : int, optional
        Number of modes to retain.
    K : int, optional
        Number of snapshots for decomposition.

    Returns
    -------
    numpy.ndarray
        Scalar flux at cell centers (cells, groups) if angular=False and
        edges=0, otherwise returns angular or edge flux as specified.
    """

    # Initialize flux
    cells_x = medium_map.shape[0]
    groups = xs_total.shape[1]
    flux_old = np.zeros((cells_x, groups))

    # Combine scattering and fission
    xs_matrix = tools.transfer_matrix(xs_scatter, xs_fission, chi)

    # Run dynamic mode decomposition for scalar flux centers
    flux = mg.dynamic_mode_decomp(
        flux_old,
        xs_total,
        xs_matrix,
        external,
        boundary,
        medium_map,
        delta_x,
        angle_x,
        angle_w,
        bc_x,
        geometry,
        R,
        K,
    )

    if (angular is False) and (edges == 0):
        return flux

    # For angular flux or scalar flux edges
    return known_source_calculation(
        flux,
        xs_total,
        xs_matrix,
        external,
        boundary,
        medium_map,
        delta_x,
        angle_x,
        angle_w,
        bc_x,
        geometry,
        angular,
        edges,
    )


def known_source_calculation(
    flux,
    xs_total,
    xs_scatter,
    external,
    boundary,
    medium_map,
    delta_x,
    angle_x,
    angle_w,
    bc_x,
    geometry,
    angular,
    edges,
):
    """Calculate angular flux or edge flux from scalar flux.

    Computes angular flux at cell centers/edges or scalar flux at cell edges
    from a known scalar flux solution.

    Parameters
    ----------
    flux : numpy.ndarray
        Scalar flux at cell centers (cells, groups).
    xs_total : numpy.ndarray
        Total macroscopic cross section (cells, groups).
    xs_scatter : numpy.ndarray
        Scattering cross section (cells, groups).
    external : numpy.ndarray
        External source (cells, angles, groups).
    boundary : numpy.ndarray
        Boundary conditions (2, angles, groups).
    medium_map : numpy.ndarray
        Material region map (cells,).
    delta_x : numpy.ndarray
        Cell widths (cells,).
    angle_x : numpy.ndarray
        Discrete ordinates (angles,).
    angle_w : numpy.ndarray
        Quadrature weights (angles,).
    bc_x : int
        Boundary condition type (0=vacuum, 1=reflective).
    geometry : int
        Geometry type (1=slab, 2=cylindrical, 3=spherical).
    angular : bool
        If True, return angular flux; if False, return scalar flux.
    edges : int
        If 1, return flux at cell edges; if 0, at cell centers.

    Returns
    -------
    numpy.ndarray
        Angular flux (cells, angles, groups) if angular=True, or
        scalar flux at edges (cells+1, groups) if angular=False and edges=1.
    """

    cells_x, groups = flux.shape
    angles = angle_x.shape[0]

    # Create (sigma_s + sigma_f) * phi + external source
    source = np.zeros((cells_x, angles, groups))
    tools._source_total(source, flux, xs_scatter, medium_map, external)

    # Scalar Edges
    if (angular is False) and (edges == 1):
        return mg.known_source_scalar(
            xs_total,
            source,
            boundary,
            medium_map,
            delta_x,
            angle_x,
            angle_w,
            bc_x,
            geometry,
            edges,
        )

    # Angular centers or edges
    return mg.known_source_angular(
        xs_total,
        source,
        boundary,
        medium_map,
        delta_x,
        angle_x,
        angle_w,
        bc_x,
        geometry,
        edges,
    )
