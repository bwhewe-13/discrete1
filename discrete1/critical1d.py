"""One-dimensional criticality drivers.

This module contains helpers for running k-eigenvalue power iterations in
1D slab geometries. Both standard and hybrid (coarse/fine) drivers are
provided.
"""

import numpy as np

import discrete1
from discrete1 import multi_group as mg
from discrete1 import tools
from discrete1.utils import hybrid as hytools

count_kk = 100
change_kk = 1e-06


def power_iteration(
    xs_total,
    xs_scatter,
    xs_fission,
    medium_map,
    delta_x,
    angle_x,
    angle_w,
    bc_x,
    geometry=1,
):
    """Run power iteration for 1D multigroup problems.

    Parameters
    ----------
    xs_total, xs_scatter, xs_fission : numpy.ndarray
        Cross section arrays indexed by material.
    medium_map : array_like
        Spatial medium mapping (length I).
    delta_x : array_like
        Cell widths.
    angle_x, angle_w : array_like
        Angular ordinates and weights.
    bc_x : list-like
        Boundary condition indicators.
    geometry : int, optional
        Geometry selector (1=slab, 2=sphere).

    Returns
    -------
    tuple
        (flux, keff) converged scalar flux and multiplication factor.
    """

    # Set boundary source
    boundary = np.zeros((2, 1, 1))

    # Initialize and normalize flux
    cells_x = medium_map.shape[0]
    flux_old = np.random.rand(cells_x, xs_total.shape[1])
    keff = np.linalg.norm(flux_old)
    flux_old /= np.linalg.norm(keff)

    # Initialize power source
    source = np.zeros((cells_x, 1, xs_total.shape[1]))

    converged = False
    count = 0
    change = 0.0

    while not (converged):
        # Update power source term
        tools._fission_source(flux_old, xs_fission, source, medium_map, keff)

        # Solve for scalar flux
        flux = mg.source_iteration(
            flux_old,
            xs_total,
            xs_scatter,
            source,
            boundary,
            medium_map,
            delta_x,
            angle_x,
            angle_w,
            bc_x,
            geometry,
        )

        # Update keffective
        keff = tools._update_keffective(flux, flux_old, xs_fission, medium_map, keff)

        # Normalize flux
        flux /= np.linalg.norm(flux)

        # Check for convergence
        change = np.linalg.norm((flux - flux_old) / flux / cells_x)
        print(f"Count: {count:>2}\tKeff: {keff:.8f}", end="\r")
        converged = (change < change_kk) or (count >= count_kk)
        count += 1

        flux_old = flux.copy()

    print(f"\nConvergence: {change:2.6e}")
    return flux, keff


def hybrid_power_iteration(
    xs_total_u,
    xs_scatter_u,
    xs_fission_u,
    medium_map,
    delta_x,
    angle_xu,
    angle_wu,
    bc_x,
    angles_c,
    groups_c,
    energy_grid,
    geometry=1,
):
    """Run the hybrid coarse/fine power iteration driver.

    This wraps the workflow needed to compute collided (coarse) and
    uncollided (fine) contributions and combines them for a hybrid solve.
    Returns a converged uncollided flux and keff.
    """

    # Collect hybrid parameters
    _, coarse_idx, factor = hytools.indexing(*energy_grid)

    # Calculate collided terms
    collided = _collided_terms(
        xs_total_u,
        xs_scatter_u,
        angle_xu,
        angle_wu,
        bc_x,
        angles_c,
        groups_c,
        energy_grid,
    )
    xs_total_c, xs_scatter_c, angle_xc, angle_wc = collided

    # Set boundary source
    boundary = np.zeros((2, 1, 1))

    # Initialize and normalize flux
    cells_x = medium_map.shape[0]
    flux_u = np.random.rand(cells_x, xs_total_u.shape[1])
    keff = np.linalg.norm(flux_u)
    flux_u /= np.linalg.norm(keff)
    flux_old = flux_u.copy()

    # Initialize power source
    flux_c = np.random.rand(cells_x, xs_total_c.shape[1])
    flux_c /= np.linalg.norm(flux_c)
    source_c = np.zeros((cells_x, 1, xs_total_c.shape[1]))

    converged = False
    count = 0
    change = 0.0

    while not (converged):
        # Update power source term
        tools._hybrid_fission_source(
            flux_u, xs_fission_u, source_c, medium_map, keff, coarse_idx
        )

        # Solve for scalar flux
        flux_c = mg.source_iteration(
            flux_c,
            xs_total_c,
            xs_scatter_c,
            source_c,
            boundary,
            medium_map,
            delta_x,
            angle_xc,
            angle_wc,
            bc_x,
            geometry,
        )

        tools._hybrid_combine_fluxes(flux_u, flux_c, coarse_idx, factor)

        # Update keffective
        keff = tools._update_keffective(
            flux_u, flux_old, xs_fission_u, medium_map, keff
        )

        # Normalize flux
        flux_u /= np.linalg.norm(flux_u)

        # Check for convergence
        change = np.linalg.norm((flux_u - flux_old) / flux_u / cells_x)
        print(f"Count: {count:>2}\tKeff: {keff:.8f}")  # , end="\r")
        converged = (change < change_kk) or (count >= count_kk)
        count += 1

        flux_old = flux_u.copy()

    print(f"\nConvergence: {change:2.6e}")
    return flux_u, keff


def _collided_terms(
    xs_total_u,
    xs_scatter_u,
    angle_xu,
    angle_wu,
    bc_x,
    angles_c,
    groups_c,
    energy_grid,
):
    # Get hybrid parameters
    edges_g, edges_gidx_u, edges_gidx_c = energy_grid

    # Check for same number of energy groups
    if groups_c == xs_total_u.shape[1]:
        xs_total_c = xs_total_u.copy()
        xs_scatter_c = xs_scatter_u.copy()
    else:
        xs_collided = hytools.coarsen_materials(
            xs_total_u, xs_scatter_u, None, edges_g[edges_gidx_u], edges_gidx_c
        )
        xs_total_c, xs_scatter_c, _ = xs_collided

    # Check for same number of angles
    if angles_c == angle_xu.shape[0]:
        angle_xc = angle_xu.copy()
        angle_wc = angle_wu.copy()
    else:
        angle_xc, angle_wc = discrete1.angular_x(angles_c, bc_x)

    return xs_total_c, xs_scatter_c, angle_xc, angle_wc
