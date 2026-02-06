"""Utility routines and performance-critical kernels for neutron transport.

This module provides computational tools for multigroup neutron transport
calculations, including source term calculations, scattering operations,
fission source handling, and hybrid machine learning / physics methods.
Functions are optimized using Numba JIT compilation where appropriate
for performance-critical operations.

The module is organized into several functional categories:

**Multigroup Transport Functions**
    - ``scatter_prod``: Compute scattering source terms
    - ``fission_mat_prod``: Compute fission source terms for eigenvalue problems
    - ``fission_vec_prod``: Compute fission source terms for eigenvalue problems
    - ``reaction_rates``: Calculate per-cell reaction rates

**Hybrid ML/Physics Functions**
    - ``scatter_prod_predict``: Scatter source with ML model predictions
    - ``fission_prod_predict``: Fission source with ML model predictions

**Geometry and Boundary Utilities**
    - ``reflector_corrector``: Manage reflective boundary conditions

**Dynamic Mode Decomposition**
    - ``dmd``: Accelerate convergence using DMD extrapolation

**Spatially Independent (0D) Variants**
    Functions prefixed with ``_0d`` provide zero-dimensional versions
    of transport operations for homogeneous problems.

**Internal Kernels**
    Functions prefixed with ``_`` are Numba-decorated internal kernels that
    implement heavy numerical work. These are used by higher-level solvers
    and are generally not part of the public API.

Notes
-----
Many functions use Numba's JIT compilation with ``nopython=True`` and
``cache=True`` for optimal performance. Type signatures are explicitly
specified for compiled functions to ensure type stability.

The hybrid ML/physics functions allow material-specific predictions using
trained machine learning models while falling back to traditional physics
calculations when models are unavailable.

Examples
--------
Compute scattering source for a multigroup problem:

>>> source = np.zeros((n_cells, n_groups))
>>> scatter_prod(flux, xs_scatter, source, medium_map)

Use hybrid ML approach for fission source:

>>> fission_prod_predict(flux, xs_fission, source, medium_map, keff, models)
"""

import numba
import numpy as np


################################################################################
# Multigroup functions
################################################################################
def transfer_matrix(xs_scatter, xs_fission, chi=None):
    """Combine scatter and fission matrices.

    Allows for combining the scattering and fission matrices for fixed source
    and time dependent problems. If ``chi`` is not None, ``xs_fission``
    represents the nusigf vector.

    Parameters
    ----------
    xs_scatter : numpy.ndarray
        Scattering cross sections indexed by [material, group, group].
    xs_fission : numpy.ndarray
        Fission cross sections indexed by [material, group, group] or
        [material, group] if chi is not None.
    chi : numpy.ndarray, optional
        Fission Neutron Distribution indexed by [material, group]. Must be
        included if xs_fission is nusigf. Default is None.

    Returns
    -------
    flux : numpy.ndarray
        Converged scalar flux distribution indexed by [cell, group].
    """
    if chi is None:
        return xs_scatter + xs_fission
    xs_matrix = np.zeros(xs_scatter.shape)
    _transfer_matrix(xs_matrix, xs_scatter, chi, xs_fission)
    return xs_matrix


@numba.jit("void(f8[:,:,:], f8[:,:,:], f8[:,:], f8[:,:])", nopython=True, cache=True)
def _transfer_matrix(xs_matrix, xs_scatter, chi, nusigf):
    # Get parameters
    materials, groups, _ = xs_matrix.shape
    mat = numba.int32
    og = numba.int32
    ig = numba.int32
    # Iterate over cells and groups
    for mat in range(materials):
        for og in range(groups):
            for ig in range(groups):
                xs_matrix[mat, og, ig] = (
                    xs_scatter[mat, og, ig] + chi[mat, og] + nusigf[mat, ig]
                )


@numba.jit(
    "void(f8[:,:], f8[:,:], i4[:], f8[:,:,:], f8[:], i4)", nopython=True, cache=True
)
def _off_scatter(flux, flux_old, medium_map, xs_scatter, off_scatter, gg):
    # Get parameters
    cells_x, groups = flux.shape
    ii = numba.int32
    mat = numba.int32
    og = numba.int32
    # Zero out previous off scatter term
    off_scatter *= 0.0
    # Iterate over cells and groups
    for ii in range(cells_x):
        mat = medium_map[ii]
        for og in range(0, gg):
            off_scatter[ii] += xs_scatter[mat, gg, og] * flux[ii, og]
        for og in range(gg + 1, groups):
            off_scatter[ii] += xs_scatter[mat, gg, og] * flux_old[ii, og]


@numba.jit(
    "void(f8[:,:], f8[:,:], i4[:], f8[:,:,:], f8[:], i4, i4[:])",
    nopython=True,
    cache=True,
)
def _variable_off_scatter(
    flux,
    flux_old,
    medium_map,
    xs_scatter,
    off_scatter,
    gg,
    edges_gidx_c,
):
    # Get parameters
    cells_x, groups_c = flux.shape
    ii = numba.int32
    mat = numba.int32
    og = numba.int32

    # Zero out previous off scatter term
    off_scatter *= 0.0

    # Iterate over cells and groups
    for og in range(groups_c):

        idx1 = edges_gidx_c[og]
        idx2 = edges_gidx_c[og + 1]

        if og < gg:
            for ii in range(cells_x):
                mat = medium_map[ii]
                off_scatter[ii] += np.sum(xs_scatter[mat, :, idx1:idx2]) * flux[ii, og]

        elif og > gg:
            for ii in range(cells_x):
                mat = medium_map[ii]
                off_scatter[ii] += (
                    np.sum(xs_scatter[mat, :, idx1:idx2]) * flux_old[ii, og]
                )

        elif og == gg:
            continue


@numba.jit("void(f8[:,:], f8[:,:], i4[:])", nopython=True, cache=True)
def _coarsen_flux(fine_flux, coarse_flux, edges_gidx_c):
    # Get parameters
    _, groups_c = coarse_flux.shape
    gg = numba.int32

    # Iterate over collided groups
    for gg in range(groups_c):
        coarse_flux[:, gg] = np.sum(
            fine_flux[:, edges_gidx_c[gg] : edges_gidx_c[gg + 1]], axis=1
        )


def reflector_corrector(reflector, angle_x, edge, nn, bc_x):
    """Apply reflector boundary correction for a paired angle.

    When reflective boundary conditions are active, the outgoing edge
    flux for one discrete ordinate is used to set the corresponding
    reflected ordinate's incoming reflector buffer. This helper writes
    the edge value into the reflector array at the symmetric angle
    index when reflection flags are set in ``bc_x``.

    Parameters
    ----------
    reflector : numpy.ndarray
        Array storing reflected edge values for each angle (n_angles,).
    angle_x : array_like
        Angular ordinates (n_angles,).
    edge : float
        Computed outgoing edge flux for the current angle.
    nn : int
        Current angle index.
    bc_x : sequence
        Boundary condition flags for left/right reflections.
    """

    # Get opposite angle
    reflected_idx = numba.int32(angle_x.shape[0] - nn - 1)
    if (angle_x[nn] > 0.0 and bc_x[1] == 1) or (angle_x[nn] < 0.0 and bc_x[0] == 1):
        reflector[reflected_idx] = edge


@numba.jit(
    "void(f8[:,:,:], f8[:,:], f8[:,:,:], i4[:], f8[:,:,:])", nopython=True, cache=True
)
def _source_total(source, flux, xs_scatter, medium_map, external):
    # Get parameters
    cells_x, angles, groups = source.shape
    ii = numba.int32
    nn = numba.int32
    nn_q = numba.int32
    mat = numba.int32
    og = numba.int32
    og_q = numba.int32
    ig = numba.int32
    one_group = numba.float64
    # Zero out previous source term
    source *= 0.0
    # Iterate over cells, angles, groups
    for ii in range(cells_x):
        mat = medium_map[ii]

        # Iterate over groups
        for og in range(groups):
            og_q = 0 if external.shape[2] == 1 else og
            one_group = 0.0
            for ig in range(groups):
                one_group += flux[ii, ig] * xs_scatter[mat, og, ig]

            # Iterate over angles
            for nn in range(angles):
                nn_q = 0 if external.shape[1] == 1 else nn
                source[ii, nn, og] = one_group + external[ii, nn_q, og_q]


@numba.jit("void(f8[:,:,:], f8[:,:], f8[:,:,:], i4[:])", nopython=True, cache=True)
def _time_right_side(q_star, flux, xs_scatter, medium_map):
    # Create (sigma_s + sigma_f) * phi + external + 1/(v*dt) * psi function
    # Get parameters
    cells_x, angles, groups = q_star.shape
    ii = numba.int32
    mat = numba.int32
    og = numba.int32
    ig = numba.int32
    one_group = numba.float64
    # Iterate over cells and groups
    for ii in range(cells_x):
        mat = medium_map[ii]
        # Iterate over groups
        for og in range(groups):
            one_group = 0.0
            for ig in range(groups):
                one_group += flux[ii, ig] * xs_scatter[mat, og, ig]
            q_star[ii, :, og] += one_group


################################################################################
# Anisotropic Scattering
################################################################################

# def legendre_polynomial(moment, x):
#     # Legendre polynomial using polynomial degree (moment) and
#     # evaluation point (mu)
#     if moment == 0:
#         return 1
#     elif moment == 1:
#         return mu
#     else:
#         return ((2 * moment - 1) * mu * legendre_polynomial(moment - 1, mu) \
#                 - (moment - 1) * legendre_polynomial(moment - 2, mu)) / moment


# @numba.jit("f8(f8, f8[:], f8, f8)", nopython=True, cache=True)
# def anisotropic_scattering(angular_flux, xs_scatter, angle_x, angle_w):
#     moment = numba.int32
#     source = numba.float64(0.0)
#     for moment in range(xs_scatter.shape[0]):
#         source += (2 * moment + 1) * xs_scatter[moment] * angle_w \
#                     * legendre_polynomial(moment, angle_x) * angular_flux
#     return source


################################################################################
# Multigroup functions - DMD
################################################################################


def _svd_dmd(A, K):
    residual = 1e-09

    # Compute SVD
    U, S, V = np.linalg.svd(A, full_matrices=False)

    # Find the non-zero singular values
    if S[(1 - np.cumsum(S) / np.sum(S)) > residual].size >= 1:
        spos = S[(1 - np.cumsum(S) / np.sum(S)) > residual].copy()
    else:
        spos = S[S > 0].copy()

    # Create diagonal matrix
    mat_size = np.min([K, len(spos)])
    S = np.zeros((mat_size, mat_size))

    # Select the u and v that correspond with the nonzero singular values
    U = U[:, :mat_size].copy()
    V = V[:mat_size, :].copy()

    # S will be the inverse of the singular value diagonal matrix
    S[np.diag_indices(mat_size)] = 1 / spos

    return U, S, V


# @numba.jit("f8[:,:](f8[:,:], f8[:,:,:], f8[:,:,:], i4)", nopython=True, cache=True)
def dmd(flux_old, y_minus, y_plus, K):
    """Estimate flux via Dynamic Mode Decomposition (DMD) extrapolation.

    Given snapshot differences (y_minus, y_plus) collected over K-1
    timesteps/iterations, compute a low-rank DMD model and use it to
    extrapolate the flux from the last known iterate.

    Parameters
    ----------
    flux_old : numpy.ndarray
        Last known flux (n_cells, n_groups).
    y_minus : numpy.ndarray
        Snapshot difference array with shape (n_cells, n_groups, K-1)
        containing previous (backward) differences.
    y_plus : numpy.ndarray
        Snapshot difference array with shape (n_cells, n_groups, K-1)
        containing forward differences.
    K : int
        Number of snapshots (including the base) used for DMD.

    Returns
    -------
    numpy.ndarray
        Extrapolated flux array with shape (n_cells, n_groups).
    """

    # Collect dimensions
    cells_x, groups = flux_old.shape

    # Flatten y_minus, y_plus
    y_minus = y_minus.reshape(cells_x * groups, K - 1)
    y_plus = y_plus.reshape(cells_x * groups, K - 1)

    # Call SVD
    U, S, V = _svd_dmd(y_minus, K)

    # Calculate Atilde
    Atilde = U.T @ y_plus @ V.T @ S.T

    # Calculate delta_y
    eye = np.identity(Atilde.shape[0])
    delta_y = np.linalg.solve(eye - Atilde, (U.T @ y_plus[:, -1]).T)

    # Estimate new flux
    flux = (flux_old.flatten() - y_plus[:, K - 2]) + (U @ delta_y).T
    flux = flux.reshape(cells_x, groups)

    return flux


################################################################################
# K-eigenvalue functions
################################################################################


@numba.jit(
    "f8[:,:,:](f8[:,:], f8[:,:,:], f8[:,:,:], i4[:], f8)", nopython=True, cache=True
)
def fission_mat_prod(flux, xs_fission, source, medium_map, keff):
    r"""Calculate fission source term for criticality problems.

    Computes the fission source for each spatial cell and energy group by
    integrating the product of flux and fission cross sections, scaled by
    the effective multiplication factor (keff). This is used in eigenvalue
    (criticality) calculations.

    Parameters
    ----------
    flux : numpy.ndarray
        Scalar flux array of shape (cells_x, groups). Contains the neutron
        flux at each spatial cell and energy group.
    xs_fission : numpy.ndarray
        Fission production cross section array of shape (materials, groups, groups).
        xs_fission[m, og, ig] is the cross section for neutrons in group ig
        causing fission that produces neutrons in group og for material m.
    source : numpy.ndarray
        Fission source array of shape (cells_x, 1, groups) to be updated.
        Zeroed out and overwritten with computed fission source.
    medium_map : numpy.ndarray
        Material index map of shape (cells_x,). Maps each spatial cell to
        its material index.
    keff : float
        Effective multiplication factor. Used to normalize the fission source.

    Returns
    -------
    numpy.ndarray
        Updated fission source array of shape (cells_x, 1, groups).

    Notes
    -----
    This function is JIT-compiled with Numba for performance. The fission
    source for cell ii and output group og is:

    .. math::
        S_{f,ii,og} = \\frac{1}{k_{eff}} \\sum_{ig} \\phi_{ii,ig} \\sigma_{f,mat,og,ig}

    where mat = medium_map[ii].
    """
    # Get parameters
    cells_x, groups = flux.shape
    ii = numba.int32
    mat = numba.int32
    og = numba.int32
    ig = numba.int32
    one_group = numba.float64
    # Zero out previous source
    source *= 0.0
    # Iterate over cells and groups
    for ii in range(cells_x):
        mat = medium_map[ii]
        if xs_fission[mat].sum() == 0.0:
            continue
        for og in range(groups):
            # source[ii,0,og] = np.sum(flux[ii] @ xs_fission[mat,og]) / keff
            one_group = 0.0
            for ig in range(groups):
                one_group += flux[ii, ig] * xs_fission[mat, og, ig]
            source[ii, 0, og] = one_group / keff
    # Return matrix vector product
    return source


@numba.jit("f8(f8[:,:], f8[:,:], f8[:,:,:], i4[:], f8)", nopython=True, cache=True)
def _update_keff_mat(flux, flux_old, xs_fission, medium_map, keff):
    # Get iterables
    cells_x, groups = flux.shape
    ii = numba.int32
    mat = numba.int32
    og = numba.int32
    ig = numba.int32
    # Initialize fission rates
    rate_new = 0.0
    rate_old = 0.0
    # Iterate over cells and groups
    for ii in range(cells_x):
        mat = medium_map[ii]
        if xs_fission[mat].sum() == 0.0:
            continue
        for og in range(groups):
            for ig in range(groups):
                rate_new += flux[ii, ig] * xs_fission[mat, og, ig]
                rate_old += flux_old[ii, ig] * xs_fission[mat, og, ig]
    return (rate_new * keff) / rate_old


@numba.jit(
    "f8[:,:,:](f8[:,:], f8[:,:], f8[:,:], f8[:,:,:], i4[:], f8)",
    nopython=True,
    cache=True,
)
def fission_vec_prod(flux, chi, nusigf, source, medium_map, keff):
    r"""Calculate fission source term for criticality problems.

    Computes the fission source for each spatial cell and energy group by
    integrating the product of flux and fission cross sections, scaled by
    the effective multiplication factor (keff). This is used in eigenvalue
    (criticality) calculations.

    Parameters
    ----------
    flux : numpy.ndarray
        Scalar flux array of shape (cells_x, groups). Contains the neutron
        flux at each spatial cell and energy group.
    chi : numpy.ndarray
        Fission neutron distribution of shape (materials, groups). chi[m, og] is
        the fission probability output of group og for material m.
    nusigf : numpy.ndarray
        Fission production cross section array of shape (materials, groups).
        nusigf[m, ig] is the cross section for neutrons in group ig
        causing fission that produces neutrons for material m.
    source : numpy.ndarray
        Fission source array of shape (cells_x, 1, groups) to be updated.
        Zeroed out and overwritten with computed fission source.
    medium_map : numpy.ndarray
        Material index map of shape (cells_x,). Maps each spatial cell to
        its material index.
    keff : float
        Effective multiplication factor. Used to normalize the fission source.

    Returns
    -------
    numpy.ndarray
        Updated fission source array of shape (cells_x, 1, groups).

    Notes
    -----
    This function is JIT-compiled with Numba for performance. The fission
    source for cell ii and output group og is:

    .. math::
        S_{f,ii,og} = \\frac{1}{k_{eff}} \\sum_{ig} \\phi_{ii,ig} \\sigma_{f,mat,og,ig}

    where mat = medium_map[ii].
    """
    # Get parameters
    cells_x, groups = flux.shape
    ii = numba.int32
    mat = numba.int32
    og = numba.int32
    ig = numba.int32
    one_group = numba.float64
    # Zero out previous source
    source *= 0.0
    # Iterate over cells and groups
    for ii in range(cells_x):
        mat = medium_map[ii]
        if chi[mat].sum() == 0.0:
            continue
        for og in range(groups):
            # source[ii,0,og] = np.sum(flux[ii] @ xs_fission[mat,og]) / keff
            one_group = 0.0
            for ig in range(groups):
                one_group += flux[ii, ig] * chi[mat, og] * nusigf[mat, ig]
            source[ii, 0, og] = one_group / keff
    # Return matrix vector product
    return source


@numba.jit(
    "f8(f8[:,:], f8[:,:], f8[:,:], f8[:,:], i4[:], f8)", nopython=True, cache=True
)
def _update_keff_vec(flux, flux_old, chi, nusigf, medium_map, keff):
    # Get iterables
    cells_x, groups = flux.shape
    ii = numba.int32
    mat = numba.int32
    og = numba.int32
    ig = numba.int32
    # Initialize fission rates
    rate_new = 0.0
    rate_old = 0.0
    # Iterate over cells and groups
    for ii in range(cells_x):
        mat = medium_map[ii]
        if chi[mat].sum() == 0.0:
            continue
        for og in range(groups):
            for ig in range(groups):
                rate_new += flux[ii, ig] * chi[mat, og] * nusigf[mat, ig]
                rate_old += flux_old[ii, ig] * chi[mat, og] * nusigf[mat, ig]
    return (rate_new * keff) / rate_old


@numba.jit(
    "f8[:,:,:](f8[:,:], f8[:,:,:], f8[:,:,:], i4[:], f8, i4[:])",
    nopython=True,
    cache=True,
)
def _hybrid_fission_source(flux_u, xs_fission, source_c, medium_map, keff, coarse_idx):
    # Get parameters
    cells_x, groups = flux_u.shape
    ii = numba.int32
    mat = numba.int32
    og = numba.int32
    ig = numba.int32
    one_group = numba.float64
    # Zero out previous source
    source_c *= 0.0
    # Iterate over cells and groups
    for ii in range(cells_x):
        mat = medium_map[ii]
        if xs_fission[mat].sum() == 0.0:
            continue
        for og in range(groups):
            # source[ii,0,og] = np.sum(flux[ii] @ xs_fission[mat,og]) / keff
            one_group = 0.0
            for ig in range(groups):
                one_group += flux_u[ii, ig] * xs_fission[mat, og, ig]
            source_c[ii, 0, coarse_idx[og]] += one_group
    # Include k-effective term
    source_c /= keff
    # Return matrix vector product
    return source_c


@numba.jit("void(f8[:,:], f8[:,:], i4[:], f8[:])", nopython=True, cache=True)
def _hybrid_combine_fluxes(flux_u, flux_c, coarse_idx, factor):
    # Get parameters
    cells_x, groups = flux_u.shape
    ii = numba.int32
    gg = numba.int32
    # Combine the uncollided and collided fluxes
    for ii in range(cells_x):
        for gg in range(groups):
            flux_u[ii, gg] = flux_c[ii, coarse_idx[gg]] * factor[gg]


################################################################################
# DJINN functions
################################################################################
# Fission source is of size (I x 1 x G) - for source iteration dimensions
# Scatter source is of size (I x G) - for discrete ordinates dimensions


def fission_prod_predict(
    flux, xs_fission, source, medium_map, keff, models, label=None
):
    r"""Calculate fission source using hybrid ML/physics approach.

    Computes the fission source using a combination of machine learning models
    and traditional physics calculations. For each material, if a trained ML
    model is available, it predicts the fission source; otherwise, the standard
    physics-based calculation is used.

    Parameters
    ----------
    flux : numpy.ndarray
        Scalar flux array of shape (cells_x, groups). Contains the neutron
        flux at each spatial cell and energy group.
    xs_fission : numpy.ndarray
        Fission production cross section array of shape (materials, groups, groups).
        Used for scaling predictions and fallback calculations.
    source : numpy.ndarray
        Fission source array of shape (cells_x, 1, groups) to be updated.
        Zeroed out and overwritten with computed fission source.
    medium_map : numpy.ndarray
        Material index map of shape (cells_x,). Maps each spatial cell to
        its material index. Also used to index into the models list.
    keff : float
        Effective multiplication factor. Used to normalize the fission source.
    models : list
        List of trained ML models (or integers as placeholders). models[nn]
        is used for material nn. If models[nn] is an integer (typically 0),
        falls back to physics-based calculation.
    label : numpy.ndarray, optional
        Parameter/label array for parametric ML predictions. Passed to
        model.predict() when making predictions. Default is None.

    Returns
    -------
    None
        The source array is modified in-place.

    Notes
    -----
    The ML models predict the energy distribution of fission neutrons, which
    is then scaled to conserve the total fission rate. The scaling factor is
    computed as:

    .. math::
        scale_i = \\sum_g \\phi_{i,g} \\sigma_{f,mat,g,0}

    The predicted source is then normalized by keff and rescaled to match
    the physics-based total fission rate.
    """
    # Zero out previous source
    source *= 0.0

    # Iterate over models
    for nn, model in enumerate(models):
        # Find which cells model is predicting
        idx = np.argwhere(medium_map == nn).flatten()

        # Check if model available
        if isinstance(model, int):
            # Calculate standard source
            source[idx] = fission_mat_prod(
                flux[idx], xs_fission, source[idx], medium_map[idx], keff
            )
            continue

        # Separate predicting flux
        mat_flux = flux[idx].copy()
        # Check for zero values
        if np.sum(mat_flux) == 0:
            continue
        # Get scaling factor
        scale = np.sum(mat_flux * xs_fission[nn, :, 0], axis=1)
        # Check for labels and predict
        mat_flux = model.predict(mat_flux, label)
        # Scale back and add to source
        source[idx, 0] = mat_flux / keff * (scale / np.sum(mat_flux, axis=1))[:, None]


def scatter_prod_predict(flux, xs_scatter, source, medium_map, models, label=None):
    r"""Calculate scatter source using hybrid ML/physics approach.

    Computes the scatter source using a combination of machine learning models
    and traditional physics calculations. For each material, if a trained ML
    model is available, it predicts the scatter source; otherwise, the standard
    physics-based calculation is used.

    Parameters
    ----------
    flux : numpy.ndarray
        Scalar flux array of shape (cells_x, groups). Contains the neutron
        flux at each spatial cell and energy group.
    xs_scatter : numpy.ndarray
        Scattering cross section array of shape (materials, groups, groups).
        xs_scatter[m, og, ig] is the cross section for scattering from
        group ig to group og for material m.
    source : numpy.ndarray
        Scatter source array of shape (cells_x, groups) to be updated.
        Zeroed out and overwritten with computed scatter source.
    medium_map : numpy.ndarray
        Material index map of shape (cells_x,). Maps each spatial cell to
        its material index. Also used to index into the models list.
    models : list
        List of trained ML models (or integers as placeholders). models[nn]
        is used for material nn. If models[nn] is an integer (typically 0),
        falls back to physics-based calculation.
    label : numpy.ndarray, optional
        Parameter/label array for parametric ML predictions. Passed to
        model.predict() when making predictions. Default is None.

    Returns
    -------
    None
        The source array is modified in-place.

    Notes
    -----
    The ML models predict the energy distribution of scattered neutrons, which
    is then scaled to conserve the total scatter rate. The scaling factor is
    computed as:

    .. math::
        scale_i = \\sum_g \\phi_{i,g} \\sigma_{s,mat,g,0}

    The predicted source is rescaled to match the physics-based total scatter rate:

    .. math::
        S_{s,predicted} = S_{ML} \\cdot \\frac{scale}{\\sum_g S_{ML,g}}
    """
    # Zero out previous source
    source *= 0.0

    # Iterate over models
    for nn, model in enumerate(models):
        # Find which cells model is predicting
        idx = np.argwhere(medium_map == nn).flatten()

        # Check if model available
        if isinstance(model, int):
            # Calculate standard source
            source[idx] = scatter_prod(
                flux[idx], xs_scatter, source[idx], medium_map[idx]
            )
            continue

        # Separate predicting flux
        mat_flux = flux[idx].copy()
        # Check for zero values
        if np.sum(mat_flux) == 0:
            continue
        # Get scaling factor
        scale = np.sum(mat_flux * xs_scatter[nn, :, 0], axis=1)
        # Check for labels and predict
        mat_flux = model.predict(mat_flux, label=label)
        # Scale back and add to source
        source[idx] = mat_flux * (scale / np.sum(mat_flux, axis=1))[:, None]


@numba.jit("f8[:,:](f8[:,:], f8[:,:,:], f8[:,:], i4[:])", nopython=True, cache=True)
def scatter_prod(flux, xs_scatter, source, medium_map):
    r"""Calculate scatter source term for multigroup transport.

    Computes the scattering source for each spatial cell and energy group by
    integrating the product of flux and scattering cross sections. This
    represents neutrons scattering from all energy groups into each output
    energy group.

    Parameters
    ----------
    flux : numpy.ndarray
        Scalar flux array of shape (cells_x, groups). Contains the neutron
        flux at each spatial cell and energy group.
    xs_scatter : numpy.ndarray
        Scattering cross section array of shape (materials, groups, groups).
        xs_scatter[m, og, ig] is the cross section for scattering from
        group ig to group og for material m.
    source : numpy.ndarray
        Scatter source array of shape (cells_x, groups) to be updated.
        Overwritten with computed scatter source.
    medium_map : numpy.ndarray
        Material index map of shape (cells_x,). Maps each spatial cell to
        its material index.

    Returns
    -------
    numpy.ndarray
        Updated scatter source array of shape (cells_x, groups).

    Notes
    -----
    This function is JIT-compiled with Numba for performance. The scatter
    source for cell ii and output group og is:

    .. math::
        S_{s,ii,og} = \\sum_{ig} \\phi_{ii,ig} \\sigma_{s,mat,og,ig}

    where mat = medium_map[ii].
    """
    # Get parameters
    cells_x, groups = flux.shape
    ii = numba.int32
    mat = numba.int32
    og = numba.int32
    ig = numba.int32
    one_group = numba.float64
    # Iterate over cells and groups
    for ii in range(cells_x):
        # Calculate material
        mat = medium_map[ii]
        if xs_scatter[mat].sum() == 0.0:
            continue
        # Iterate over groups
        for og in range(groups):
            one_group = 0.0
            for ig in range(groups):
                one_group += flux[ii, ig] * xs_scatter[mat, og, ig]
            source[ii, og] = one_group
    # Return matrix vector product
    return source


################################################################################
# Hybrid functions
################################################################################


@numba.jit(
    "void(f8[:,:], f8[:,:,:], f8[:,:,:], i4[:], i4[:])", nopython=True, cache=True
)
def _hybrid_source_collided(flux_u, xs_scatter, source_c, medium_map, coarse_idx):
    # Get parameters
    cells_x, groups = flux_u.shape
    ii = numba.int32
    mat = numba.int32
    og = numba.int32
    ig = numba.int32
    # Zero out previous source
    source_c *= 0.0
    # Iterate over all spatial cells
    for ii in range(cells_x):
        mat = medium_map[ii]
        for og in range(groups):
            for ig in range(groups):
                source_c[ii, 0, coarse_idx[og]] += (
                    flux_u[ii, ig] * xs_scatter[mat, og, ig]
                )


@numba.jit(
    "void(f8[:,:], f8[:,:], f8[:,:,:], f8[:,:,:], i4[:], i4[:], f8[:])",
    nopython=True,
    cache=True,
)
def _hybrid_source_total(
    flux_u, flux_c, xs_scatter_u, q_star, medium_map, coarse_idx, factor
):
    # Get parameters
    cells_x, angles, groups = q_star.shape
    ii = numba.int32
    mat = numba.int32
    nn = numba.int32
    og = numba.int32
    ig = numba.int32
    one_group = numba.float64
    # Assume that source is already (Qu + 1 / (v * dt) * psi^{\ell-1})
    for ii in range(cells_x):
        mat = medium_map[ii]
        for og in range(groups):
            flux_u[ii, og] = flux_u[ii, og] + flux_c[ii, coarse_idx[og]] * factor[og]
        for og in range(groups):
            one_group = 0.0
            for ig in range(groups):
                one_group += flux_u[ii, ig] * xs_scatter_u[mat, og, ig]
            for nn in range(angles):
                q_star[ii, nn, og] += one_group


################################################################################
# Display Options
################################################################################


def reaction_rates(flux, xs_matrix, medium_map):
    """Compute per-cell reaction rates from flux and cross sections.

    Parameters
    ----------
    flux : numpy.ndarray
        Scalar or angular-integrated flux with shape (n_cells, n_groups).
    xs_matrix : numpy.ndarray
        Cross-section matrix indexed by material (n_materials, n_groups, ...)
        where the last axes align with group dimensions.
    medium_map : array_like
        Material index per spatial cell (n_cells,).

    Returns
    -------
    numpy.ndarray
        Reaction rate per cell and per group with shape (n_cells, n_groups).
    """

    # Flux parameters
    cells_x, groups = flux.shape
    # Initialize reaction rate data
    rate = np.zeros((cells_x, groups))
    # Iterate over spatial cells
    for ii, mat in enumerate(medium_map):
        rate[ii] = flux[ii] @ xs_matrix[mat].T
    # return reaction rate
    return rate


################################################################################
# Spatially Independent Functions
################################################################################


@numba.jit("f8(f8[:], f8[:], f8[:,:], i4)", nopython=True, cache=True)
def _off_scatter_0d(flux, flux_old, xs_scatter, gg):
    # Get parameters
    groups = flux.shape[0]
    off_scatter = numba.float64
    off_scatter = 0.0
    og = numba.int32
    # Iterate over groups
    for og in range(0, gg):
        off_scatter += xs_scatter[gg, og] * flux[og]
    for og in range(gg + 1, groups):
        off_scatter += xs_scatter[gg, og] * flux_old[og]
    return off_scatter


@numba.jit("f8[:,:](f8[:], f8[:,:], f8[:,:], f8)", nopython=True, cache=True)
def _fission_mat_source_0d(flux, xs_fission, source, keff):
    # Get parameters
    groups = flux.shape[0]
    og = numba.int32
    ig = numba.int32
    one_group = numba.float64
    # Zero out previous source
    source *= 0.0
    # Iterate over groups
    for og in range(groups):
        one_group = 0.0
        for ig in range(groups):
            one_group += flux[ig] * xs_fission[og, ig]
        source[0, og] = one_group / keff
    # Return matrix vector product
    return source


@numba.jit("f8[:,:](f8[:], f8[:], f8[:], f8[:,:], f8)", nopython=True, cache=True)
def _fission_vec_source_0d(flux, chi, nusigf, source, keff):
    # Get parameters
    groups = flux.shape[0]
    og = numba.int32
    ig = numba.int32
    one_group = numba.float64
    # Zero out previous source
    source *= 0.0
    # Iterate over groups
    for og in range(groups):
        one_group = 0.0
        for ig in range(groups):
            one_group += flux[ig] * chi[og] * nusigf[ig]
        source[0, og] = one_group / keff
    # Return matrix vector product
    return source


@numba.jit("f8(f8[:], f8[:], f8[:,:], f8)", nopython=True, cache=True)
def _update_keff_mat_0d(flux, flux_old, xs_fission, keff):
    # Get iterables
    groups = flux.shape[0]
    og = numba.int32
    ig = numba.int32
    # Initialize fission rates
    rate_new = 0.0
    rate_old = 0.0
    # Iterate over groups
    for og in range(groups):
        for ig in range(groups):
            rate_new += flux[ig] * xs_fission[og, ig]
            rate_old += flux_old[ig] * xs_fission[og, ig]
    return (rate_new * keff) / rate_old


@numba.jit("f8(f8[:], f8[:], f8[:], f8[:], f8)", nopython=True, cache=True)
def _update_keff_vec_0d(flux, flux_old, chi, nusigf, keff):
    # Get iterables
    groups = flux.shape[0]
    og = numba.int32
    ig = numba.int32
    # Initialize fission rates
    rate_new = 0.0
    rate_old = 0.0
    # Iterate over groups
    for og in range(groups):
        for ig in range(groups):
            rate_new += flux[ig] * chi[og] * nusigf[ig]
            rate_old += flux_old[ig] * chi[og] * nusigf[ig]
    return (rate_new * keff) / rate_old
