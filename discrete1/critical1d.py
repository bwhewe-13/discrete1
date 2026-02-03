"""One-dimensional criticality drivers.

This module provides solvers for k-eigenvalue problems in one-dimensional
slab and spherical geometries using power iteration methods. It includes:

- Standard power iteration for multigroup criticality calculations
- Data collection utilities for DJINN machine learning model training
- DJINN-accelerated power iteration with surrogate model predictions
- Hybrid coarse/fine mesh power iteration for improved efficiency

All drivers solve the multigroup neutron transport equation with vacuum or
reflective boundary conditions to determine the critical multiplication
factor (k-effective) and corresponding neutron flux distribution.
"""

import numpy as np

import discrete1
from discrete1 import multi_group as mg
from discrete1 import tools
from discrete1.utils import hybrid as hytools

count_kk = 100
change_kk = 1e-06

count_pp = 20
change_pp = 1e-05


def power_iteration(
    xs_total,
    xs_scatter,
    xs_fission,
    medium_map,
    delta_x,
    angle_x,
    angle_w,
    bc_x,
    chi=None,
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
    chi : numpy.ndarray, optional
        Fission Neutron Distribution. Must be included if xs_fission is nusigf.
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
        if chi is None:
            tools.fission_mat_prod(flux_old, xs_fission, source, medium_map, keff)
        else:
            tools.fission_vec_prod(flux_old, chi, xs_fission, source, medium_map, keff)

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
        if chi is None:
            keff = tools._update_keff_mat(flux, flux_old, xs_fission, medium_map, keff)
        else:
            keff = tools._update_keff_vec(
                flux, flux_old, chi, xs_fission, medium_map, keff
            )

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


def collect_power_iteration(
    xs_total,
    xs_scatter,
    xs_fission,
    medium_map,
    delta_x,
    angle_x,
    angle_w,
    bc_x,
    filepath,
    chi=None,
    geometry=1,
):
    """Collect training data for DJINN models during power iteration.

    Runs a standard k-eigenvalue power iteration while collecting and saving
    flux snapshots at each iteration step. The saved data includes scalar flux
    distributions, cross-section arrays, and spatial medium mapping, which can
    be used to train DJINN (Deep Joint Informed Neural Network) surrogate models
    for accelerated neutron transport calculations.

    Parameters
    ----------
    xs_total : numpy.ndarray
        Total cross sections indexed by [material, group].
    xs_scatter : numpy.ndarray
        Scattering cross sections indexed by [material, group, group].
    xs_fission : numpy.ndarray
        Fission cross sections indexed by [material, group].
    medium_map : array_like
        Spatial material index mapping of length I (number of cells).
    delta_x : array_like
        Cell widths for spatial discretization.
    angle_x : array_like
        Discrete ordinates (angular quadrature points).
    angle_w : array_like
        Angular quadrature weights.
    bc_x : list-like
        Boundary condition indicators [left, right] (0=vacuum, 1=reflective).
    filepath : str
        Directory path where training data files will be saved.
    chi : numpy.ndarray, optional
        Fission Neutron Distribution. Must be included if xs_fission is nusigf.
    geometry : int, optional
        Geometry type (1=slab, 2=sphere). Default is 1.

    Returns
    -------
    flux : numpy.ndarray
        Converged scalar flux distribution indexed by [cell, group].
    keff : float
        Converged effective multiplication factor.

    Notes
    -----
    The function saves the following NumPy arrays to disk:
    - `flux_fission_model.npy` : Flux snapshots from each iteration
    - `fission_cross_sections.npy` : Fission cross section data
    - `scatter_cross_sections.npy` : Scattering cross section data
    - `medium_map.npy` : Spatial material mapping

    Additionally, source iteration data is saved during the transport sweeps
    via `mg.source_iteration_collect()` for training DJINN scatter models.
    """

    # Set boundary source
    boundary = np.zeros((2, 1, 1))

    # Initialize and normalize flux
    cells_x = medium_map.shape[0]
    groups = xs_total.shape[1]

    flux_old = np.random.rand(cells_x, groups)
    keff = np.linalg.norm(flux_old)
    flux_old /= np.linalg.norm(keff)

    # Initialize power source
    source = np.zeros((cells_x, 1, groups))
    tracked_flux = np.zeros((count_kk, cells_x, groups))

    converged = False
    count = 0
    change = 0.0

    while not (converged):
        # Update power source term
        if chi is None:
            tools.fission_mat_prod(flux_old, xs_fission, source, medium_map, keff)
        else:
            tools.fission_vec_prod(flux_old, chi, xs_fission, source, medium_map, keff)

        # Solve for scalar flux
        flux = mg.source_iteration_collect(
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
            count,
            filepath,
        )

        # Update keffective
        if chi is None:
            keff = tools._update_keff_mat(flux, flux_old, xs_fission, medium_map, keff)
        else:
            keff = tools._update_keff_vec(
                flux, flux_old, chi, xs_fission, medium_map, keff
            )

        # Normalize flux
        flux /= np.linalg.norm(flux)

        # Check for convergence
        change = np.linalg.norm((flux - flux_old) / flux / cells_x)
        print(f"Count: {count:>2}\tKeff: {keff:.8f}", end="\r")
        converged = (change < change_kk) or (count >= count_kk)
        count += 1

        # Update old flux and tracked flux
        flux_old = flux.copy()
        tracked_flux[count - 2] = flux.copy()

    print(f"\nConvergence: {change:2.6e}")
    np.save(filepath + "flux_fission_model", tracked_flux[: count - 1])

    # Save relevant information to file
    np.save(filepath + "fission_cross_sections", xs_fission)
    np.save(filepath + "scatter_cross_sections", xs_scatter)
    np.save(filepath + "medium_map", medium_map)

    return flux, keff


def ml_power_iteration(
    flux_old,
    xs_total,
    xs_scatter,
    xs_fission,
    medium_map,
    delta_x,
    angle_x,
    angle_w,
    bc_x,
    chi=None,
    geometry=1,
    fission_models=[],
    scatter_models=[],
    fission_labels=None,
    scatter_labels=None,
):
    """Power iteration with optional DJINN surrogate model acceleration.

    Performs k-eigenvalue power iteration that can optionally use trained DJINN
    (Deep Joint Informed Neural Network) models to predict fission and/or
    scattering sources, significantly accelerating convergence. When no models
    are provided, falls back to standard power iteration.

    Parameters
    ----------
    flux_old : numpy.ndarray
        Initial scalar flux guess indexed by [cell, group].
    xs_total : numpy.ndarray
        Total cross sections indexed by [material, group].
    xs_scatter : numpy.ndarray
        Scattering cross sections indexed by [material, group, group].
    xs_fission : numpy.ndarray
        Fission cross sections indexed by [material, group].
    medium_map : array_like
        Spatial material index mapping of length I (number of cells).
    delta_x : array_like
        Cell widths for spatial discretization.
    angle_x : array_like
        Discrete ordinates (angular quadrature points).
    angle_w : array_like
        Angular quadrature weights.
    bc_x : list-like
        Boundary condition indicators [left, right] (0=vacuum, 1=reflective).
    chi : numpy.ndarray, optional
        Fission Neutron Distribution. Must be included if xs_fission is nusigf.
    geometry : int, optional
        Geometry type (1=slab, 2=sphere). Default is 1.
    fission_models : list, optional
        Trained DJINN models for fission source prediction. Empty list uses
        traditional calculation. Default is [].
    scatter_models : list, optional
        Trained DJINN models for scattering source prediction. Empty list uses
        traditional calculation. Default is [].
    fission_labels : array_like, optional
        Material labels for fission model predictions. Default is None.
    scatter_labels : array_like, optional
        Material labels for scatter model predictions. Default is None.

    Returns
    -------
    flux : numpy.ndarray
        Converged scalar flux distribution indexed by [cell, group].
    keff : float
        Converged effective multiplication factor.

    Notes
    -----
    - Convergence criteria differ based on model usage: stricter for standard
      iteration (change_kk), more relaxed for DJINN-accelerated (change_pp).
    - Early termination occurs if flux change increases after 5 iterations,
      preventing divergence.
    - Normalization strategy changes when DJINN models are used to maintain
      numerical stability.
    """

    # Set boundary source
    boundary = np.zeros((2, 1, 1))

    # Initialize keff
    cells_x = medium_map.shape[0]
    keff_old = 1.0

    # Initialize power source
    fission_source = np.zeros((cells_x, 1, xs_total.shape[1]))

    converged = False
    count = 0
    change_old = 100.0

    while not (converged):
        # Update power source term
        # No Fission DJINN predictions
        if len(fission_models) == 0 and chi is None:
            tools.fission_mat_prod(
                flux_old, xs_fission, fission_source, medium_map, keff_old
            )
        elif len(fission_models) == 0:
            tools.fission_vec_prod(
                flux_old, chi, xs_fission, fission_source, medium_map, keff_old
            )
        # Fission DJINN predictions
        else:
            tools.fission_prod_predict(
                flux_old,
                xs_fission,
                fission_source,
                medium_map,
                1.0,
                fission_models,
                fission_labels,
            )

        # Solve for scalar flux
        # No Scatter DJINN predictions
        if len(scatter_models) == 0:
            flux = mg.source_iteration(
                flux_old,
                xs_total,
                xs_scatter,
                fission_source,
                boundary,
                medium_map,
                delta_x,
                angle_x,
                angle_w,
                bc_x,
                geometry,
            )
        # Scatter DJINN predictions
        else:
            flux = mg.ml_source_iteration(
                flux_old,
                xs_total,
                xs_scatter,
                fission_source,
                boundary,
                medium_map,
                delta_x,
                angle_x,
                angle_w,
                bc_x,
                geometry,
                scatter_models,
                scatter_labels,
            )

        # Update keffective and normalize flux
        if len(fission_models) == 0:
            keff = tools._update_keffective(
                flux, flux_old, xs_fission, medium_map, keff_old
            )
            flux /= np.linalg.norm(flux)

        else:
            keff = np.linalg.norm(flux)
            flux /= keff

        # Check for convergence
        change = np.linalg.norm((flux - flux_old) / flux / cells_x)
        # Early exit
        if (change > change_old) and (count > 5):
            print(f"\nConvergence: {change_old:2.6e}")
            return flux_old, keff_old

        print(f"Count: {count:>2}\tKeff: {keff:.8f}\tChange: {change:.2e}", end="\r")
        if (len(fission_models) == 0) and (len(scatter_models) == 0):
            converged = (change < change_kk) or (count >= count_kk)
        else:
            converged = (change < change_pp) or (count >= count_pp)
        count += 1

        flux_old = flux.copy()
        change_old = change
        keff_old = keff

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
    """Run hybrid coarse-fine mesh power iteration for k-eigenvalue problems.

    Implements a two-level hybrid solver that separates the solution into
    uncollided (fine resolution) and collided (coarse resolution) components.
    The fine mesh captures detailed physics while the coarse mesh handles
    computationally expensive collided contributions, significantly improving
    efficiency for problems with many energy groups and angular directions.

    Parameters
    ----------
    xs_total_u : numpy.ndarray
        Fine (uncollided) total cross sections indexed by [material, group].
    xs_scatter_u : numpy.ndarray
        Fine (uncollided) scattering cross sections indexed by
        [material, group, group].
    xs_fission_u : numpy.ndarray
        Fine (uncollided) fission cross sections indexed by [material, group].
    medium_map : array_like
        Spatial material index mapping of length I (number of cells).
    delta_x : array_like
        Cell widths for spatial discretization.
    angle_xu : array_like
        Fine mesh discrete ordinates (angular quadrature points).
    angle_wu : array_like
        Fine mesh angular quadrature weights.
    bc_x : list-like
        Boundary condition indicators [left, right] (0=vacuum, 1=reflective).
    angles_c : int
        Number of angular directions for coarse mesh discretization.
    groups_c : int
        Number of energy groups for coarse mesh discretization.
    energy_grid : tuple
        Energy grid specification (edges_g, edges_gidx_u, edges_gidx_c)
        defining fine-to-coarse group mapping.
    geometry : int, optional
        Geometry type (1=slab, 2=sphere). Default is 1.

    Returns
    -------
    flux_u : numpy.ndarray
        Converged fine mesh scalar flux distribution indexed by [cell, group].
    keff : float
        Converged effective multiplication factor.

    Notes
    -----
    The hybrid method works by:
    1. Computing coarse mesh collided flux with reduced angular/energy resolution
    2. Projecting fission source from fine to coarse mesh
    3. Combining coarse collided contribution with fine uncollided solution
    4. Iterating until convergence on the fine mesh

    This approach is particularly effective for deep penetration problems or
    systems with large spatial domains where full fine-mesh transport sweeps
    would be prohibitively expensive.
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
