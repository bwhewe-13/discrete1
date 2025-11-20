"""Multigroup solvers and DJINN integration utilities.

This module provides multigroup-level driver routines that orchestrate
outer iterations over energy groups and call the spatial sweep
implementations (discrete ordinates) for each group. It includes
standard source-iteration, variable/coarsened-group iteration,
Dynamic Mode Decomposition (DMD) accelerated routines, and helpers for
generating training data or using DJINN-predicted scatter/fission
sources.

Public routines include:
- source_iteration: standard multigroup source iteration
- variable_source_iteration: coarse/fine-group-aware iteration
- dynamic_mode_decomp: DMD-accelerated solver
- known_source_angular / known_source_scalar: wrappers for known-source problems
- source_iteration_djinn: use DJINN models to predict scatter sources

The multigroup drivers expect arrays shaped as used by the rest of
the package (cell-major ordering, angular and group axes as described
in function docstrings).
"""

import numpy as np

from discrete1 import tools
from discrete1.spatial_sweep import _known_scatter, _known_source, discrete_ordinates

count_gg = 100
change_gg = 1e-08

count_pp = 50
change_pp = 1e-08


def source_iteration(
    flux_old,
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
):
    """Multigroup source iteration solver.

    Performs outer iterations over energy groups using inner discrete
    ordinates sweeps for each group. Off-diagonal scatter (up- and
    down-scatter) contributions are handled via an "off_scatter"
    accumulator that uses the previous and current flux iterates.

    Parameters
    ----------
    flux_old : numpy.ndarray
        Initial scalar flux guess with shape (n_cells, n_groups).
    xs_total : array_like
        Total cross sections (n_materials or n_cells, n_groups).
    xs_scatter : array_like
        Scatter cross sections (indexed by material, GxG).
    external : numpy.ndarray
        External source array (n_cells, n_angles, n_groups) or with
        singleton dimensions where appropriate.
    boundary : numpy.ndarray
        Boundary conditions array (2, n_angles, n_groups) or with
        singleton dimensions where appropriate.
    medium_map : array_like
        Material index per spatial cell (n_cells,).
    delta_x : array_like
        Cell widths (n_cells,).
    angle_x : array_like
        Angular ordinates (n_angles,).
    angle_w : array_like
        Quadrature weights (n_angles,).
    bc_x : sequence
        Boundary condition flags for left/right boundaries.
    geometry : int
        Geometry selector (1=slab, 2=sphere).

    Returns
    -------
    numpy.ndarray
        Converged scalar flux array with shape (n_cells, n_groups).
    """

    cells_x, groups = flux_old.shape
    flux = np.zeros((cells_x, groups))
    off_scatter = np.zeros((cells_x,))

    converged = False
    count = 1
    change = 0.0

    while not (converged):
        flux *= 0.0

        for gg in range(groups):
            # Check for sizes
            qq = 0 if external.shape[2] == 1 else gg
            bc = 0 if boundary.shape[2] == 1 else gg

            # Update off scatter source
            tools._off_scatter(flux, flux_old, medium_map, xs_scatter, off_scatter, gg)

            # Run discrete ordinates for one group
            flux[:, gg] = discrete_ordinates(
                flux_old[:, gg],
                xs_total[:, gg],
                xs_scatter[:, gg, gg],
                off_scatter,
                external[:, :, qq],
                boundary[:, :, bc],
                medium_map,
                delta_x,
                angle_x,
                angle_w,
                bc_x,
                geometry,
            )

        # Check for convergence
        try:
            change = np.linalg.norm((flux - flux_old) / flux / cells_x)
        except RuntimeWarning:
            change = 0.0
        converged = (change < change_gg) or (count >= count_gg)
        count += 1

        flux_old = flux.copy()

    return flux


def variable_source_iteration(
    flux_old,
    xs_total,
    star_coef_c,
    xs_scatter,
    external,
    boundary,
    medium_map,
    delta_x,
    angle_x,
    angle_w,
    bc_x,
    delta_coarse,
    delta_fine,
    edges_gidx_c,
    geometry,
):
    """Source iteration for problems with variable coarse/fine groups.

    This variant performs multigroup source iteration when a coarse
    energy grid is constructed from a fine grid. It computes coarse
    group cross-sections and off-scatter terms appropriately and then
    calls the discrete-ordinates solver per coarse group.

    Parameters
    ----------
    flux_old : numpy.ndarray
        Initial scalar flux guess on the fine grid (n_cells, n_groups_fine).
    xs_total : array_like
        Total cross sections on the fine grid.
    star_coef_c : array_like
        Additional coarse-group correction terms added to xs_total_c.
    xs_scatter : array_like
        Scatter cross sections on the fine grid.
    external, boundary, medium_map, delta_x, angle_x, angle_w, bc_x :
        See :func:`source_iteration` for descriptions.
    delta_coarse : array_like
        Coarse-group widths (n_groups_coarse,).
    delta_fine : array_like
        Fine-group widths (n_groups_fine,).
    edges_gidx_c : array_like
        Indices mapping coarse groups to fine-group indices.
    geometry : int
        Geometry selector (1=slab, 2=sphere).

    Returns
    -------
    numpy.ndarray
        Converged scalar flux on the fine grid (n_cells, n_groups_fine).
    """

    cells_x, groups = flux_old.shape
    flux = np.zeros((cells_x, groups))
    off_scatter = np.zeros((cells_x,), dtype=np.float64)

    converged = False
    count = 1
    change = 0.0

    while not (converged):
        flux *= 0.0

        for gg in range(groups):
            # Check for sizes
            qq = 0 if external.shape[2] == 1 else gg
            bc = 0 if boundary.shape[2] == 1 else gg

            idx1 = edges_gidx_c[gg]
            idx2 = edges_gidx_c[gg + 1]

            xs_scatter_c = (
                np.sum(
                    xs_scatter[:, idx1:idx2, idx1:idx2] * delta_fine[idx1:idx2],
                    axis=(1, 2),
                )
                / delta_coarse[gg]
            )

            xs_total_c = (
                np.sum(xs_total[:, idx1:idx2] * delta_fine[idx1:idx2], axis=1)
                / delta_coarse[gg]
            )
            xs_total_c += star_coef_c[gg]

            # Update off scatter source
            tools._variable_off_scatter(
                flux / delta_coarse,
                flux_old / delta_coarse,
                medium_map,
                xs_scatter[:, idx1:idx2] * delta_fine,
                off_scatter,
                gg,
                edges_gidx_c,
            )

            # Run discrete ordinates for one group
            flux[:, gg] = discrete_ordinates(
                flux_old[:, gg],
                xs_total_c,
                xs_scatter_c,
                off_scatter,
                external[:, :, qq],
                boundary[:, :, bc],
                medium_map,
                delta_x,
                angle_x,
                angle_w,
                bc_x,
                geometry,
            )

        # Check for convergence
        try:
            change = np.linalg.norm((flux - flux_old) / flux / cells_x)
        except RuntimeWarning:
            change = 0.0
        converged = (change < change_gg) or (count >= count_gg)
        count += 1

        flux_old = flux.copy()

    return flux


def dynamic_mode_decomp(
    flux_old,
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
    R,
    K,
):
    """Perform Dynamic Mode Decomposition (DMD) accelerated iteration.

    Runs R+K source-iteration steps collecting snapshot differences
    then performs a DMD extrapolation to estimate the converged flux.

    Parameters
    ----------
    flux_old : numpy.ndarray
        Initial scalar flux guess with shape (n_cells, n_groups).
    xs_total, xs_scatter, external, boundary, medium_map, delta_x, angle_x, \
        angle_w, bc_x, geometry :
        See :func:`source_iteration` for descriptions.
    R : int
        Number of iterations before collecting DMD snapshots.
    K : int
        Number of DMD snapshots to collect and use for extrapolation.

    Returns
    -------
    numpy.ndarray
        Estimated flux after DMD extrapolation (n_cells, n_groups).
    """

    cells_x, groups = flux_old.shape
    flux = np.zeros((cells_x, groups))
    off_scatter = np.zeros((cells_x,))

    # Initialize Y_plus and Y_minus
    y_plus = np.zeros((cells_x, groups, K - 1))
    y_minus = np.zeros((cells_x, groups, K - 1))

    converged = False
    change = 0.0

    for rk in range(R + K):

        # Return flux if there is convergence
        if converged:
            return flux

        flux *= 0.0

        for gg in range(groups):
            # Check for sizes
            qq = 0 if external.shape[2] == 1 else gg
            bc = 0 if boundary.shape[2] == 1 else gg

            # Update off scatter source
            tools._off_scatter(flux, flux_old, medium_map, xs_scatter, off_scatter, gg)

            # Run discrete ordinates for one group
            flux[:, gg] = discrete_ordinates(
                flux_old[:, gg],
                xs_total[:, gg],
                xs_scatter[:, gg, gg],
                off_scatter,
                external[:, :, qq],
                boundary[:, :, bc],
                medium_map,
                delta_x,
                angle_x,
                angle_w,
                bc_x,
                geometry,
            )

        # Check for convergence
        change = np.linalg.norm((flux - flux_old) / flux / cells_x)
        converged = change < change_gg

        # Collect difference for DMD on K iterations
        if rk >= R:
            kk = rk - R
            if kk < (K - 1):
                y_minus[:, :, kk] = flux - flux_old
            if kk > 0:
                y_plus[:, :, kk - 1] = flux - flux_old

        flux_old = flux.copy()

    # Perform DMD
    flux = tools.dmd(flux, y_minus, y_plus, K)

    return flux


########################################################################
# Multigroup Known Source Problems
########################################################################


def known_source_angular(
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
):
    """Compute angular flux for a prescribed multigroup source.

    Solves the transport equation for a supplied angular source term and
    returns the angular flux field (including any edge terms indicated
    by `edges`). This is a convenience wrapper around the
    spatial_sweep._known_source kernel invoked per group.

    Parameters
    ----------
    xs_total : array_like
        Total cross sections (per material or per cell) indexed by group.
    source : numpy.ndarray
        Prescribed source with shape (n_cells, n_angles, n_groups).
    boundary, medium_map, delta_x, angle_x, angle_w, bc_x, geometry:
        See :func:`source_iteration` for descriptions.
    edges : int
        Number of edge cells included in the angular flux array.

    Returns
    -------
    numpy.ndarray
        Angular flux array with shape (n_cells + edges, n_angles, n_groups).
    """

    cells_x, angles, groups = source.shape

    # Initialize scalar flux
    angular_flux = np.zeros((cells_x + edges, angles, groups))

    # Initialize dummy variable
    zero = np.zeros((cells_x + edges))

    for gg in range(groups):

        qq = 0 if source.shape[2] == 1 else gg
        bc = 0 if boundary.shape[2] == 1 else gg

        _known_source(
            angular_flux[:, :, gg],
            xs_total[:, gg],
            zero,
            source[:, :, qq],
            boundary[:, :, bc],
            medium_map,
            delta_x,
            angle_x,
            angle_w,
            bc_x,
            geometry,
            edges,
        )

    return angular_flux


def known_source_scalar(
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
):
    """Compute scalar (group-integrated) flux for a prescribed source.

    Wrapper that solves for scalar flux using a prescribed source and
    the internal _known_source kernel. Returns a (n_cells+edges, n_groups)
    scalar flux array suitable for post-processing.

    Parameters
    ----------
    xs_total, source, boundary, medium_map, delta_x, angle_x, angle_w, bc_x, \
        geometry, edges :
        See :func:`known_source_angular` for descriptions.

    Returns
    -------
    numpy.ndarray
        Scalar flux array with shape (n_cells + edges, n_groups).
    """

    cells_x, angles, groups = source.shape

    # Initialize scalar flux
    scalar_flux = np.zeros((cells_x + edges, groups, 1))

    # Initialize dummy variable
    # zero = 1e-15 * np.ones((cells_x + edges))
    zero = np.zeros((cells_x + edges))

    for gg in range(groups):

        qq = 0 if source.shape[2] == 1 else gg
        bc = 0 if boundary.shape[2] == 1 else gg

        _known_source(
            scalar_flux[:, gg],
            xs_total[:, gg],
            zero,
            source[:, :, qq],
            boundary[:, :, bc],
            medium_map,
            delta_x,
            angle_x,
            angle_w,
            bc_x,
            geometry,
            edges,
        )

    return scalar_flux[:, :, 0]


########################################################################
# Multigroup DJINN Problems
########################################################################


def source_iteration_collect(
    flux_old,
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
    iteration,
    filepath,
):
    """Source iteration that collects intermediate flux snapshots.

    Runs standard multigroup source iteration while recording per-outer-
    iteration flux snapshots into an array saved to disk. Useful for
    generating training data for DJINN scatter models.

    Parameters
    ----------
    flux_old : numpy.ndarray
        Initial scalar flux guess with shape (n_cells, n_groups).
    xs_total, xs_scatter, external, boundary, medium_map, delta_x, angle_x, \
        angle_w, bc_x, geometry :
        See :func:`source_iteration` for descriptions.
    iteration : int
        Iteration index used to name the saved snapshot file.
    filepath : str
        Directory/filepath prefix where snapshots are saved.

    Returns
    -------
    numpy.ndarray
        Final converged flux (n_cells, n_groups).
    """

    cells_x, groups = flux_old.shape
    flux = np.zeros((cells_x, groups))
    off_scatter = np.zeros((cells_x,))
    tracked_flux = np.zeros((count_gg, cells_x, groups))

    converged = False
    count = 1
    change = 0.0

    while not (converged):
        flux *= 0.0

        for gg in range(groups):
            # Check for sizes
            qq = 0 if external.shape[2] == 1 else gg
            bc = 0 if boundary.shape[2] == 1 else gg

            # Update off scatter source
            tools._off_scatter(flux, flux_old, medium_map, xs_scatter, off_scatter, gg)

            # Run discrete ordinates for one group
            flux[:, gg] = discrete_ordinates(
                flux_old[:, gg],
                xs_total[:, gg],
                xs_scatter[:, gg, gg],
                off_scatter,
                external[:, :, qq],
                boundary[:, :, bc],
                medium_map,
                delta_x,
                angle_x,
                angle_w,
                bc_x,
                geometry,
            )

        # Check for convergence
        change = np.linalg.norm((flux - flux_old) / flux / cells_x)
        converged = (change < change_gg) or (count >= count_gg)
        count += 1

        # Update old flux and tracked flux
        flux_old = flux.copy()
        tracked_flux[count - 2] = flux.copy()

    fiteration = str(iteration).zfill(3)
    np.save(filepath + f"flux_scatter_model_{fiteration}", tracked_flux[: count - 1])

    return flux


def source_iteration_djinn(
    flux_old,
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
    scatter_models=[],
    scatter_labels=None,
):
    """Source iteration using DJINN-predicted scatter source.

    Uses pre-trained DJINN models to predict the scatter source each
    outer iteration and then solves each group with the provided
    discrete-ordinates known-scatter kernel. This is intended to
    accelerate or replace explicit scattering computations.

    Parameters
    ----------
    flux_old : numpy.ndarray
        Initial scalar flux guess with shape (n_cells, n_groups).
    xs_total, xs_scatter, external, boundary, medium_map, delta_x, angle_x, \
        angle_w, bc_x, geometry :
        See :func:`source_iteration` for descriptions.
    scatter_models : list, optional
        Sequence of loaded DJINN models (or placeholders) indexed by
        material id; models produce group-wise scatter predictions.
    scatter_labels : numpy.ndarray, optional
        Optional label array used as additional model inputs (n_cells,).

    Returns
    -------
    numpy.ndarray
        Converged scalar flux (n_cells, n_groups).
    """

    cells_x, groups = flux_old.shape
    flux = np.zeros((cells_x, groups))
    scatter_source = np.zeros((cells_x, groups))

    converged = False
    count = 1
    change = 0.0

    while not (converged):
        flux *= 0.0

        tools._djinn_scatter_predict(
            flux_old,
            xs_scatter,
            scatter_source,
            medium_map,
            scatter_models,
            scatter_labels,
        )

        for gg in range(groups):
            # Check for sizes
            qq = 0 if external.shape[2] == 1 else gg
            bc = 0 if boundary.shape[2] == 1 else gg

            # Run discrete ordinates for one group
            flux[:, gg] = _known_scatter(
                xs_total[:, gg],
                scatter_source[:, gg],
                external[:, :, qq],
                boundary[:, :, bc],
                medium_map,
                delta_x,
                angle_x,
                angle_w,
                bc_x,
                geometry,
            )

        # Check for convergence
        change = np.linalg.norm((flux - flux_old) / flux / cells_x)
        converged = (change < change_pp) or (count >= count_pp)
        count += 1

        # Update old flux and tracked flux
        flux_old = flux.copy()

    return flux
