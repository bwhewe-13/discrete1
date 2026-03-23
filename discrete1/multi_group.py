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
- ml_source_iteration: use DJINN models to predict scatter sources

The multigroup drivers expect arrays shaped as used by the rest of
the package (cell-major ordering, angular and group axes as described
in function docstrings).
"""

import numpy as np

from discrete1 import tools
from discrete1.spatial_sweep import (
    known_source_sn,
    scatter_source_sn,
    slab_anisotropic_sn,
    slab_isotropic_sn,
    sphere_anisotropic_sn,
    sphere_isotropic_sn,
)

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
    P=None,
    P_weights=None,
):
    """Multigroup source iteration solver - dispatches to iso or aniso variant.

    Performs outer iterations over energy groups using inner discrete
    ordinates sweeps for each group. Off-diagonal scatter (up- and
    down-scatter) contributions are handled via an "off_scatter"
    accumulator that uses the previous and current flux iterates.

    Parameters
    ----------
    flux_old : numpy.ndarray, shape (n_cells, n_groups)
        Initial scalar flux guess.
    xs_total : numpy.ndarray, shape (n_materials, n_groups)
        Total cross sections.
    xs_scatter : numpy.ndarray
        Scatter cross sections.
        - Isotropic: shape (n_materials, n_groups, n_groups).
        - Anisotropic: shape (n_materials, n_groups, n_groups, L+1).
    external : numpy.ndarray, shape (n_cells, n_angles or 1, n_groups or 1)
        External source array.
    boundary : numpy.ndarray, shape (2, n_angles or 1, n_groups or 1)
        Boundary conditions.
    medium_map : numpy.ndarray, shape (n_cells,)
        Material index per cell.
    delta_x : numpy.ndarray, shape (n_cells,)
        Cell widths.
    angle_x : numpy.ndarray, shape (angles,)
        Angular quadrature points.
    angle_w : numpy.ndarray, shape (angles,)
        Angular quadrature weights.
    bc_x : sequence
        Boundary condition flags for left/right boundaries.
    geometry : int
        1 = slab, 2 = sphere.
    P : numpy.ndarray, shape (L+1, angles), optional
        Precomputed Legendre polynomials. Required for anisotropic.
    P_weights : numpy.ndarray, shape (L+1, angles), optional
        Precomputed w_n * P_l(mu_n). Required for anisotropic.

    Returns
    -------
    numpy.ndarray, shape (n_cells, n_groups)
        Converged scalar flux.
    """
    if xs_scatter.ndim == 3:
        return source_iteration_iso(
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
        )
    else:
        if P is None:
            n_moments = xs_scatter.shape[3]
            P = tools.legendre_polynomials(n_moments, angle_x)
            P_weights = angle_w[np.newaxis, :] * P
        return source_iteration_aniso(
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
            P,
            P_weights,
        )


def _one_sweep_iso(
    flux,
    flux_old,
    xs_total,
    xs_scatter,
    off_scatter,
    external,
    boundary,
    medium_map,
    delta_x,
    angle_x,
    angle_w,
    bc_x,
    geometry,
):
    groups = flux_old.shape[1]
    flux *= 0.0

    for gg in range(groups):
        qq = 0 if external.shape[2] == 1 else gg
        bc = 0 if boundary.shape[2] == 1 else gg

        # Update off scatter source
        tools._off_scatter(flux, flux_old, medium_map, xs_scatter, off_scatter, gg)

        if geometry == 1:
            flux[:, gg] = slab_isotropic_sn(
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
                edges=0,
            )
        elif geometry == 2:
            flux[:, gg] = sphere_isotropic_sn(
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
                edges=0,
            )
    return flux


def _one_sweep_aniso(
    flux,
    flux_old,
    xs_total,
    xs_scatter,
    off_scatter,
    external,
    boundary,
    medium_map,
    delta_x,
    angle_x,
    angle_w,
    bc_x,
    geometry,
    legendre_weights,
    P,
    P_weights,
):
    groups = flux_old.shape[1]
    flux *= 0.0

    for gg in range(groups):
        qq = 0 if external.shape[2] == 1 else gg
        bc = 0 if boundary.shape[2] == 1 else gg

        # Off-diagonal scatter uses L=0 moment only
        tools._off_scatter(
            flux, flux_old, medium_map, xs_scatter[..., 0], off_scatter, gg
        )

        # Within-group Legendre moments mapped to cells -> (cells_x, L+1)
        xs_self_scatter = xs_scatter[:, gg, gg, :][medium_map]

        # Precompute (2l+1) * xs_l(x) — fixed for the inner sweep loop
        xs_scatter_w = xs_self_scatter * legendre_weights

        if geometry == 1:
            flux[:, gg] = slab_anisotropic_sn(
                flux_old[:, gg],
                xs_total[:, gg],
                xs_self_scatter,
                xs_scatter_w,
                off_scatter,
                external[:, :, qq],
                boundary[:, :, bc],
                medium_map,
                delta_x,
                angle_x,
                angle_w,
                bc_x,
                0,
                P,
                P_weights,
            )
        elif geometry == 2:
            flux[:, gg] = sphere_anisotropic_sn(
                flux_old[:, gg],
                xs_total[:, gg],
                xs_self_scatter,
                xs_scatter_w,
                off_scatter,
                external[:, :, qq],
                boundary[:, :, bc],
                medium_map,
                delta_x,
                angle_x,
                angle_w,
                bc_x,
                0,
                P,
                P_weights,
            )
    return flux


def source_iteration_iso(
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
    """Multigroup source iteration for isotropic scattering.

    Performs outer iterations over energy groups using inner discrete
    ordinates sweeps for each group. Convergence is measured with the
    relative L2 norm of the flux change across all cells and groups.

    Parameters
    ----------
    flux_old : numpy.ndarray, shape (n_cells, n_groups)
        Initial scalar flux guess.
    xs_total : numpy.ndarray, shape (n_materials, n_groups)
        Total cross sections.
    xs_scatter : numpy.ndarray, shape (n_materials, n_groups, n_groups)
        Isotropic scatter cross sections. ``xs_scatter[m, og, ig]`` is
        the cross section for scattering from group ``ig`` to group
        ``og`` for material ``m``.
    external : numpy.ndarray, shape (n_cells, n_angles or 1, n_groups or 1)
        External source array.
    boundary : numpy.ndarray, shape (2, n_angles or 1, n_groups or 1)
        Boundary conditions.
    medium_map : numpy.ndarray, shape (n_cells,)
        Material index per cell.
    delta_x : numpy.ndarray, shape (n_cells,)
        Cell widths.
    angle_x : numpy.ndarray, shape (n_angles,)
        Angular quadrature points.
    angle_w : numpy.ndarray, shape (n_angles,)
        Angular quadrature weights.
    bc_x : sequence of int
        Boundary condition flags ``[left, right]``. 0 = vacuum,
        1 = reflect.
    geometry : int
        Geometry type. 1 for slab, 2 for sphere.

    Returns
    -------
    numpy.ndarray, shape (n_cells, n_groups)
        Converged scalar flux.
    """
    cells_x, groups = flux_old.shape
    flux = np.zeros((cells_x, groups))
    off_scatter = np.zeros((cells_x,))

    converged = False
    count = 1
    change = 0.0

    while not (converged):
        flux = _one_sweep_iso(
            flux,
            flux_old,
            xs_total,
            xs_scatter,
            off_scatter,
            external,
            boundary,
            medium_map,
            delta_x,
            angle_x,
            angle_w,
            bc_x,
            geometry,
        )
        try:
            change = np.linalg.norm((flux - flux_old) / flux / cells_x)
        except RuntimeWarning:
            change = 0.0
        converged = (change < change_gg) or (count >= count_gg)
        count += 1
        flux_old = flux.copy()

    return flux


def source_iteration_aniso(
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
    P,
    P_weights,
):
    """Multigroup source iteration for anisotropic scattering.

    Identical workflow to :func:`source_iteration_iso` but each group
    sweep uses the full Legendre expansion of the scatter kernel. The
    within-group anisotropic correction is applied inside the spatial
    sweep; only the L=0 moment drives the off-diagonal scatter term.

    Parameters
    ----------
    flux_old : numpy.ndarray, shape (n_cells, n_groups)
        Initial scalar flux guess.
    xs_total : numpy.ndarray, shape (n_materials, n_groups)
        Total cross sections.
    xs_scatter : numpy.ndarray, shape (n_materials, n_groups, n_groups, L+1)
        Legendre moments of scatter cross sections. The last axis indexes
        the moment order (0 to L).
    external : numpy.ndarray, shape (n_cells, n_angles or 1, n_groups or 1)
        External source array.
    boundary : numpy.ndarray, shape (2, n_angles or 1, n_groups or 1)
        Boundary conditions.
    medium_map : numpy.ndarray, shape (n_cells,)
        Material index per cell.
    delta_x : numpy.ndarray, shape (n_cells,)
        Cell widths.
    angle_x : numpy.ndarray, shape (n_angles,)
        Angular quadrature points.
    angle_w : numpy.ndarray, shape (n_angles,)
        Angular quadrature weights.
    bc_x : sequence of int
        Boundary condition flags ``[left, right]``. 0 = vacuum,
        1 = reflect.
    geometry : int
        Geometry type. 1 for slab, 2 for sphere.
    P : numpy.ndarray, shape (L+1, n_angles)
        Precomputed Legendre polynomials ``P[l, n]`` = P_l(mu_n).
    P_weights : numpy.ndarray, shape (L+1, n_angles)
        Precomputed ``w_n * P_l(mu_n)`` used for flux moment accumulation.

    Returns
    -------
    numpy.ndarray, shape (n_cells, n_groups)
        Converged scalar flux.
    """
    cells_x, groups = flux_old.shape
    flux = np.zeros((cells_x, groups))
    off_scatter = np.zeros((cells_x,))
    legendre_weights = (2 * np.arange(xs_scatter.shape[3]) + 1)[None, :]

    converged = False
    count = 1
    change = 0.0

    while not converged:
        flux = _one_sweep_aniso(
            flux,
            flux_old,
            xs_total,
            xs_scatter,
            off_scatter,
            external,
            boundary,
            medium_map,
            delta_x,
            angle_x,
            angle_w,
            bc_x,
            geometry,
            legendre_weights,
            P,
            P_weights,
        )

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
    P=None,
    P_weights=None,
):
    """Source iteration for problems with variable coarse/fine groups.

    Dispatches to the isotropic or anisotropic variant based on
    ``xs_scatter.ndim``. Coarse-group cross sections and off-scatter
    terms are collapsed from the fine-group data before each group
    sweep, allowing efficient treatment of problems where a coarse
    energy grid is derived from a fine one.

    Parameters
    ----------
    flux_old : numpy.ndarray, shape (n_cells, n_groups_coarse)
        Initial scalar flux guess on the coarse group structure.
    xs_total : numpy.ndarray, shape (n_materials, n_groups_fine)
        Fine-group total cross sections.
    star_coef_c : numpy.ndarray, shape (n_groups_coarse,)
        Coarse-group streaming/removal coefficient added to the
        collapsed total cross section.
    xs_scatter : numpy.ndarray
        Fine-group scatter cross sections.

        - Isotropic: shape ``(n_materials, n_groups_fine, n_groups_fine)``.
        - Anisotropic: shape ``(n_materials, n_groups_fine, n_groups_fine, L+1)``.
    external : numpy.ndarray, shape (n_cells, n_angles or 1, n_groups_coarse or 1)
        External source array.
    boundary : numpy.ndarray, shape (2, n_angles or 1, n_groups_coarse or 1)
        Boundary conditions.
    medium_map : numpy.ndarray, shape (n_cells,)
        Material index per cell.
    delta_x : numpy.ndarray, shape (n_cells,)
        Cell widths.
    angle_x : numpy.ndarray, shape (n_angles,)
        Angular quadrature points.
    angle_w : numpy.ndarray, shape (n_angles,)
        Angular quadrature weights.
    bc_x : sequence of int
        Boundary condition flags ``[left, right]``. 0 = vacuum,
        1 = reflect.
    delta_coarse : numpy.ndarray, shape (n_groups_coarse,)
        Coarse-group energy widths.
    delta_fine : numpy.ndarray, shape (n_groups_fine,)
        Fine-group energy widths.
    edges_gidx_c : numpy.ndarray, shape (n_groups_coarse + 1,)
        Fine-group index boundaries for each coarse group.
    geometry : int
        Geometry type. 1 for slab, 2 for sphere.
    P : numpy.ndarray, shape (L+1, n_angles), optional
        Precomputed Legendre polynomials. Computed automatically for
        anisotropic problems when not provided.
    P_weights : numpy.ndarray, shape (L+1, n_angles), optional
        Precomputed ``w_n * P_l(mu_n)``. Computed automatically when
        not provided.

    Returns
    -------
    numpy.ndarray, shape (n_cells, n_groups_coarse)
        Converged scalar flux on the coarse group structure.
    """
    if xs_scatter.ndim == 3:
        return _variable_source_iteration_iso(
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
        )
    else:
        if P is None:
            n_moments = xs_scatter.shape[3]
            P = tools.legendre_polynomials(n_moments, angle_x)
            P_weights = angle_w[np.newaxis, :] * P
        return _variable_source_iteration_aniso(
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
            P,
            P_weights,
        )


def _variable_source_iteration_iso(
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
    """Isotropic variable coarse/fine group source iteration."""
    cells_x, groups = flux_old.shape
    flux = np.zeros((cells_x, groups))
    off_scatter = np.zeros((cells_x,), dtype=np.float64)

    converged = False
    count = 1
    change = 0.0

    while not (converged):
        flux *= 0.0

        for gg in range(groups):
            qq = 0 if external.shape[2] == 1 else gg
            bc = 0 if boundary.shape[2] == 1 else gg

            idx1 = edges_gidx_c[gg]
            idx2 = edges_gidx_c[gg + 1]

            # Collapse within-group scatter to coarse group — scalar
            xs_scatter_c = (
                np.sum(
                    xs_scatter[:, idx1:idx2, idx1:idx2] * delta_fine[idx1:idx2],
                    axis=(1, 2),
                )
                / delta_coarse[gg]
            )  # shape (n_materials,)

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

            if geometry == 1:
                flux[:, gg] = slab_isotropic_sn(
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
                    edges=0,
                )
            elif geometry == 2:
                flux[:, gg] = sphere_isotropic_sn(
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
                    edges=0,
                )

        try:
            change = np.linalg.norm((flux - flux_old) / flux / cells_x)
        except RuntimeWarning:
            change = 0.0
        converged = (change < change_gg) or (count >= count_gg)
        count += 1
        flux_old = flux.copy()

    return flux


def _variable_source_iteration_aniso(
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
    P,
    P_weights,
):
    """Anisotropic variable coarse/fine group source iteration."""
    cells_x, groups = flux_old.shape
    flux = np.zeros((cells_x, groups))
    off_scatter = np.zeros((cells_x,), dtype=np.float64)

    converged = False
    count = 1
    change = 0.0

    while not converged:
        flux *= 0.0

        for gg in range(groups):
            qq = 0 if external.shape[2] == 1 else gg
            bc = 0 if boundary.shape[2] == 1 else gg

            idx1 = edges_gidx_c[gg]
            idx2 = edges_gidx_c[gg + 1]

            # Collapse within-group scatter moments to coarse group.
            # Weight by incoming fine-group width (axis 2) correct for
            # energy-weighted group collapsing of a transfer matrix.
            xs_scatter_c = (
                np.sum(
                    xs_scatter[:, idx1:idx2, idx1:idx2, :]
                    * delta_fine[None, None, idx1:idx2, None],
                    axis=(1, 2),
                )
                / delta_coarse[gg]
            )  # shape (n_materials, L+1)

            xs_total_c = (
                np.sum(xs_total[:, idx1:idx2] * delta_fine[idx1:idx2], axis=1)
                / delta_coarse[gg]
            )
            xs_total_c += star_coef_c[gg]

            # Off-diagonal scatter uses L=0 moment only
            tools._variable_off_scatter(
                flux / delta_coarse,
                flux_old / delta_coarse,
                medium_map,
                xs_scatter[:, idx1:idx2, :, 0] * delta_fine[None, None, :],
                off_scatter,
                gg,
                edges_gidx_c,
            )

            xs_self_scatter = xs_scatter_c[medium_map]  # (cells_x, L+1)
            xs_scatter_w = (
                xs_self_scatter * (2 * np.arange(xs_scatter.shape[3]) + 1)[None, :]
            )

            if geometry == 1:
                flux[:, gg] = slab_anisotropic_sn(
                    flux_old[:, gg],
                    xs_total_c,
                    xs_self_scatter,
                    xs_scatter_w,
                    off_scatter,
                    external[:, :, qq],
                    boundary[:, :, bc],
                    medium_map,
                    delta_x,
                    angle_x,
                    angle_w,
                    bc_x,
                    0,
                    P,
                    P_weights,
                )
            elif geometry == 2:
                flux[:, gg] = sphere_anisotropic_sn(
                    flux_old[:, gg],
                    xs_total_c,
                    xs_self_scatter,
                    xs_scatter_w,
                    off_scatter,
                    external[:, :, qq],
                    boundary[:, :, bc],
                    medium_map,
                    delta_x,
                    angle_x,
                    angle_w,
                    bc_x,
                    0,
                    P,
                    P_weights,
                )

        try:
            change = np.linalg.norm((flux - flux_old) / flux / cells_x)
        except RuntimeWarning:
            change = 0.0
        converged = (change < change_gg) or (count >= count_gg)
        count += 1
        flux_old = flux.copy()

    return flux


########################################################################
# DMD-accelerated source iteration
########################################################################
def _dmd_warmup(
    flux,
    flux_old,
    xs_total,
    xs_scatter,
    off_scatter,
    external,
    boundary,
    medium_map,
    delta_x,
    angle_x,
    angle_w,
    bc_x,
    geometry,
    R,
    anisotropic,
    legendre_weights,
    P=None,
    P_weights=None,
):
    """R warm-up sweeps before DMD snapshot collection.

    Returns the current flux and flux_old after R iterations, or returns
    early with (flux, None) if convergence is reached.
    """
    for _ in range(R):
        if anisotropic:
            _one_sweep_aniso(
                flux,
                flux_old,
                xs_total,
                xs_scatter,
                off_scatter,
                external,
                boundary,
                medium_map,
                delta_x,
                angle_x,
                angle_w,
                bc_x,
                geometry,
                legendre_weights,
                P,
                P_weights,
            )
        else:
            _one_sweep_iso(
                flux,
                flux_old,
                xs_total,
                xs_scatter,
                off_scatter,
                external,
                boundary,
                medium_map,
                delta_x,
                angle_x,
                angle_w,
                bc_x,
                geometry,
            )

        change = np.linalg.norm((flux - flux_old) / flux / flux_old.shape[0])
        if change < change_gg:
            return flux, None  # None signals early convergence

        flux, flux_old = flux_old, flux

    return flux, flux_old


def _dmd_collect(
    flux,
    flux_old,
    xs_total,
    xs_scatter,
    off_scatter,
    external,
    boundary,
    medium_map,
    delta_x,
    angle_x,
    angle_w,
    bc_x,
    geometry,
    K,
    n_state,
    anisotropic,
    legendre_weights=None,
    P=None,
    P_weights=None,
):
    """K snapshot collection sweeps for DMD extrapolation.

    Returns populated y_minus, y_plus matrices and the last flux_old,
    or returns early with (flux, None, None) if convergence is reached.
    """
    y_minus = np.zeros((n_state, K - 1))
    y_plus = np.zeros((n_state, K - 1))

    for kk in range(K):
        if anisotropic:
            _one_sweep_aniso(
                flux,
                flux_old,
                xs_total,
                xs_scatter,
                off_scatter,
                external,
                boundary,
                medium_map,
                delta_x,
                angle_x,
                angle_w,
                bc_x,
                geometry,
                legendre_weights,
                P,
                P_weights,
            )
        else:
            _one_sweep_iso(
                flux,
                flux_old,
                xs_total,
                xs_scatter,
                off_scatter,
                external,
                boundary,
                medium_map,
                delta_x,
                angle_x,
                angle_w,
                bc_x,
                geometry,
            )

        change = np.linalg.norm((flux - flux_old) / flux / flux_old.shape[0])
        if change < change_gg:
            return flux, None, None  # None signals early convergence

        diff = (flux - flux_old).ravel()
        if kk < K - 1:
            y_minus[:, kk] = diff
        if kk > 0:
            y_plus[:, kk - 1] = diff

        flux, flux_old = flux_old, flux

    return flux_old, y_minus, y_plus


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
    P=None,
    P_weights=None,
):
    """DMD-accelerated multigroup source iteration.

    Runs ``R`` warm-up sweeps followed by ``K`` snapshot-collection
    sweeps, then performs a Dynamic Mode Decomposition (DMD)
    extrapolation to estimate the converged flux from the collected
    difference snapshots. Dispatches to isotropic or anisotropic inner
    sweeps based on ``xs_scatter.ndim``.

    Parameters
    ----------
    flux_old : numpy.ndarray, shape (n_cells, n_groups)
        Initial scalar flux guess.
    xs_total : numpy.ndarray, shape (n_materials, n_groups)
        Total cross sections.
    xs_scatter : numpy.ndarray
        Scatter cross sections.

        - Isotropic: shape ``(n_materials, n_groups, n_groups)``.
        - Anisotropic: shape ``(n_materials, n_groups, n_groups, L+1)``.
    external : numpy.ndarray, shape (n_cells, n_angles or 1, n_groups or 1)
        External source array.
    boundary : numpy.ndarray, shape (2, n_angles or 1, n_groups or 1)
        Boundary conditions.
    medium_map : numpy.ndarray, shape (n_cells,)
        Material index per cell.
    delta_x : numpy.ndarray, shape (n_cells,)
        Cell widths.
    angle_x : numpy.ndarray, shape (n_angles,)
        Angular quadrature points.
    angle_w : numpy.ndarray, shape (n_angles,)
        Angular quadrature weights.
    bc_x : sequence of int
        Boundary condition flags ``[left, right]``. 0 = vacuum,
        1 = reflect.
    geometry : int
        Geometry type. 1 for slab, 2 for sphere.
    R : int
        Number of warm-up iterations performed before snapshot
        collection begins.
    K : int
        Number of DMD snapshot iterations used for extrapolation.
    P : numpy.ndarray, shape (L+1, n_angles), optional
        Precomputed Legendre polynomials. Computed automatically for
        anisotropic problems when not provided.
    P_weights : numpy.ndarray, shape (L+1, n_angles), optional
        Precomputed ``w_n * P_l(mu_n)``. Computed automatically when
        not provided.

    Returns
    -------
    numpy.ndarray, shape (n_cells, n_groups)
        Estimated converged flux after DMD extrapolation.
    """
    anisotropic = xs_scatter.ndim == 4
    legendre_weights = None
    if anisotropic:
        if P is None:
            n_moments = xs_scatter.shape[3]
            P = tools.legendre_polynomials(n_moments, angle_x)
            P_weights = angle_w[np.newaxis, :] * P
        legendre_weights = 2 * np.arange(xs_scatter.shape[3]) + 1

    cells_x, groups = flux_old.shape
    n_state = cells_x * groups
    off_scatter = np.zeros((cells_x,))
    flux = np.zeros((cells_x, groups))
    flux_old = flux_old.copy()

    # Warm-up phase
    flux, flux_old = _dmd_warmup(
        flux,
        flux_old,
        xs_total,
        xs_scatter,
        off_scatter,
        external,
        boundary,
        medium_map,
        delta_x,
        angle_x,
        angle_w,
        bc_x,
        geometry,
        R,
        anisotropic,
        legendre_weights,
        P,
        P_weights,
    )
    if flux_old is None:
        return flux

    # Snapshot collection phase
    flux_old, y_minus, y_plus = _dmd_collect(
        flux,
        flux_old,
        xs_total,
        xs_scatter,
        off_scatter,
        external,
        boundary,
        medium_map,
        delta_x,
        angle_x,
        angle_w,
        bc_x,
        geometry,
        K,
        n_state,
        anisotropic,
        legendre_weights,
        P,
        P_weights,
    )
    if y_minus is None:
        return flux_old

    return tools.dmd(flux_old, y_minus, y_plus, K)


########################################################################
# Known Source Problems
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
    by ``edges``). This is a convenience wrapper around the
    :func:`~discrete1.spatial_sweep.known_source_sn` kernel invoked
    per group.

    Parameters
    ----------
    xs_total : numpy.ndarray, shape (n_materials, n_groups)
        Total cross sections.
    source : numpy.ndarray, shape (n_cells, n_angles, n_groups)
        Prescribed angular source.
    boundary : numpy.ndarray, shape (2, n_angles or 1, n_groups or 1)
        Boundary conditions.
    medium_map : numpy.ndarray, shape (n_cells,)
        Material index per cell.
    delta_x : numpy.ndarray, shape (n_cells,)
        Cell widths.
    angle_x : numpy.ndarray, shape (n_angles,)
        Angular quadrature points.
    angle_w : numpy.ndarray, shape (n_angles,)
        Angular quadrature weights.
    bc_x : sequence of int
        Boundary condition flags ``[left, right]``. 0 = vacuum,
        1 = reflect.
    geometry : int
        Geometry type. 1 for slab, 2 for sphere.
    edges : int
        If 1, include an extra cell for the right-edge flux;
        0 otherwise.

    Returns
    -------
    numpy.ndarray, shape (n_cells + edges, n_angles, n_groups)
        Angular flux.
    """

    cells_x, angles, groups = source.shape

    # Initialize scalar flux
    angular_flux = np.zeros((cells_x + edges, angles, groups))

    # Initialize dummy variable
    zero = np.zeros((cells_x + edges))

    for gg in range(groups):

        qq = 0 if source.shape[2] == 1 else gg
        bc = 0 if boundary.shape[2] == 1 else gg

        known_source_sn(
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

    Solves for the scalar (angle-integrated) flux using a prescribed
    source and the
    :func:`~discrete1.spatial_sweep.known_source_sn` kernel. Returns
    a scalar flux array suitable for post-processing.

    Parameters
    ----------
    xs_total : numpy.ndarray, shape (n_materials, n_groups)
        Total cross sections.
    source : numpy.ndarray, shape (n_cells, n_angles, n_groups)
        Prescribed angular source.
    boundary : numpy.ndarray, shape (2, n_angles or 1, n_groups or 1)
        Boundary conditions.
    medium_map : numpy.ndarray, shape (n_cells,)
        Material index per cell.
    delta_x : numpy.ndarray, shape (n_cells,)
        Cell widths.
    angle_x : numpy.ndarray, shape (n_angles,)
        Angular quadrature points.
    angle_w : numpy.ndarray, shape (n_angles,)
        Angular quadrature weights.
    bc_x : sequence of int
        Boundary condition flags ``[left, right]``. 0 = vacuum,
        1 = reflect.
    geometry : int
        Geometry type. 1 for slab, 2 for sphere.
    edges : int
        If 1, include an extra cell for the right-edge flux;
        0 otherwise.

    Returns
    -------
    numpy.ndarray, shape (n_cells + edges, n_groups)
        Scalar flux.
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

        known_source_sn(
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

    Runs standard isotropic multigroup source iteration while recording
    per-outer-iteration flux snapshots into an array which is saved to
    disk. Useful for generating training data for DJINN scatter models.

    Parameters
    ----------
    flux_old : numpy.ndarray, shape (n_cells, n_groups)
        Initial scalar flux guess.
    xs_total : numpy.ndarray, shape (n_materials, n_groups)
        Total cross sections.
    xs_scatter : numpy.ndarray, shape (n_materials, n_groups, n_groups)
        Isotropic scatter cross sections.
    external : numpy.ndarray, shape (n_cells, n_angles or 1, n_groups or 1)
        External source array.
    boundary : numpy.ndarray, shape (2, n_angles or 1, n_groups or 1)
        Boundary conditions.
    medium_map : numpy.ndarray, shape (n_cells,)
        Material index per cell.
    delta_x : numpy.ndarray, shape (n_cells,)
        Cell widths.
    angle_x : numpy.ndarray, shape (n_angles,)
        Angular quadrature points.
    angle_w : numpy.ndarray, shape (n_angles,)
        Angular quadrature weights.
    bc_x : sequence of int
        Boundary condition flags ``[left, right]``. 0 = vacuum,
        1 = reflect.
    geometry : int
        Geometry type. 1 for slab, 2 for sphere.
    iteration : int
        Iteration index appended to the saved snapshot filename.
    filepath : str
        Directory or filepath prefix for the saved snapshot file.

    Returns
    -------
    numpy.ndarray, shape (n_cells, n_groups)
        Final converged flux.
    """
    cells_x, groups = flux_old.shape
    flux = np.zeros((cells_x, groups))
    off_scatter = np.zeros((cells_x,))
    tracked_flux = np.zeros((count_gg, cells_x, groups))

    converged = False
    count = 1
    change = 0.0

    while not (converged):
        flux = _one_sweep_iso(
            flux,
            flux_old,
            xs_total,
            xs_scatter,
            off_scatter,
            external,
            boundary,
            medium_map,
            delta_x,
            angle_x,
            angle_w,
            bc_x,
            geometry,
        )

        change = np.linalg.norm((flux - flux_old) / flux / cells_x)
        converged = (change < change_gg) or (count >= count_gg)
        count += 1
        flux_old = flux.copy()
        tracked_flux[count - 2] = flux.copy()

    fiteration = str(iteration).zfill(3)
    np.save(filepath + f"flux_scatter_model_{fiteration}", tracked_flux[: count - 1])

    return flux


def ml_source_iteration(
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
    outer iteration and then solves each group with the
    :func:`~discrete1.spatial_sweep.scatter_source_sn` kernel. Falls
    back to the standard physics-based scatter calculation for materials
    without a trained model.

    Parameters
    ----------
    flux_old : numpy.ndarray, shape (n_cells, n_groups)
        Initial scalar flux guess.
    xs_total : numpy.ndarray, shape (n_materials, n_groups)
        Total cross sections.
    xs_scatter : numpy.ndarray, shape (n_materials, n_groups, n_groups)
        Isotropic scatter cross sections used as fallback for materials
        without a trained model.
    external : numpy.ndarray, shape (n_cells, n_angles or 1, n_groups or 1)
        External source array.
    boundary : numpy.ndarray, shape (2, n_angles or 1, n_groups or 1)
        Boundary conditions.
    medium_map : numpy.ndarray, shape (n_cells,)
        Material index per cell.
    delta_x : numpy.ndarray, shape (n_cells,)
        Cell widths.
    angle_x : numpy.ndarray, shape (n_angles,)
        Angular quadrature points.
    angle_w : numpy.ndarray, shape (n_angles,)
        Angular quadrature weights.
    bc_x : sequence of int
        Boundary condition flags ``[left, right]``. 0 = vacuum,
        1 = reflect.
    geometry : int
        Geometry type. 1 for slab, 2 for sphere.
    scatter_models : list
        Trained DJINN models indexed by material id. An integer
        placeholder (typically ``0``) triggers the physics fallback.
    scatter_labels : numpy.ndarray, optional
        Label array passed to ``model.predict()`` for parametric models.
        Default is ``None``.

    Returns
    -------
    numpy.ndarray, shape (n_cells, n_groups)
        Converged scalar flux.
    """
    cells_x, groups = flux_old.shape
    flux = np.zeros((cells_x, groups))
    scatter_source = np.zeros((cells_x, groups))

    converged = False
    count = 1
    change = 0.0

    while not (converged):
        flux *= 0.0

        tools.scatter_prod_predict(
            flux_old,
            xs_scatter,
            scatter_source,
            medium_map,
            scatter_models,
            scatter_labels,
        )

        for gg in range(groups):
            qq = 0 if external.shape[2] == 1 else gg
            bc = 0 if boundary.shape[2] == 1 else gg

            flux[:, gg] = scatter_source_sn(
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

        change = np.linalg.norm((flux - flux_old) / flux / cells_x)
        converged = (change < change_pp) or (count >= count_pp)
        count += 1
        flux_old = flux.copy()

    return flux
