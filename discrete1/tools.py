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

# from scipy.linalg import svd


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
    numpy.ndarray
        Combined transfer matrix indexed by [material, group, group, moment].
    """
    xs_matrix = xs_scatter.copy()

    # Anisotropic scattering with fission matrix
    if (chi is None) and (xs_scatter.ndim == 4):
        xs_matrix[:, :, :, 0] += xs_fission
    # Anisotropic scattering with nusigf vector and chi
    elif (chi is not None) and (xs_scatter.ndim == 4):
        _transfer_matrix(xs_matrix[:, :, :, 0], xs_scatter[:, :, :, 0], chi, xs_fission)
    # Isotropic scattering with fission matrix
    elif (chi is None) and (xs_scatter.ndim == 3):
        xs_matrix += xs_fission
    # Isotropic scattering with nusigf vector and chi
    else:
        _transfer_matrix(xs_matrix, xs_scatter, chi, xs_fission)

    return xs_matrix


@numba.jit("void(f8[:,:,:], f8[:,:,:], f8[:,:], f8[:,:])", nopython=True, cache=True)
def _transfer_matrix(xs_matrix, xs_scatter, chi, nusigf):
    """Assemble the isotropic transfer matrix from scatter and fission data.

    Parameters
    ----------
    xs_matrix : numpy.ndarray
        Output transfer matrix of shape ``(materials, groups, groups)``.
    xs_scatter : numpy.ndarray
        Scattering matrix of shape ``(materials, groups, groups)``.
    chi : numpy.ndarray
        Fission spectrum of shape ``(materials, groups)``.
    nusigf : numpy.ndarray
        Fission production vector of shape ``(materials, groups)``.

    Returns
    -------
    None
        The combined matrix is written into ``xs_matrix`` in-place.
    """
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
                    xs_scatter[mat, og, ig] + chi[mat, og] * nusigf[mat, ig]
                )


@numba.jit(
    "void(f8[:,:], f8[:,:], i4[:], f8[:,:,:], f8[:], i4)", nopython=True, cache=True
)
def _off_scatter(flux, flux_old, medium_map, xs_scatter, off_scatter, gg):
    """Accumulate off-diagonal scatter contributions for one group sweep.

    Parameters
    ----------
    flux : numpy.ndarray
        Current scalar flux iterate with shape ``(cells_x, groups)``.
    flux_old : numpy.ndarray
        Previous scalar flux iterate with shape ``(cells_x, groups)``.
    medium_map : numpy.ndarray
        Material index per cell with shape ``(cells_x,)``.
    xs_scatter : numpy.ndarray
        Scattering matrix of shape ``(materials, groups, groups)``.
    off_scatter : numpy.ndarray
        Output array of shape ``(cells_x,)`` filled in-place.
    gg : int
        Energy group currently being solved.

    Returns
    -------
    None
        The off-scatter source is written into ``off_scatter``.
    """
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
    flux, flux_old, medium_map, xs_scatter, off_scatter, gg, edges_gidx_c
):
    """Compute off-diagonal scatter source for variable coarse/fine groups.

    Uses only the L=0 Legendre moment for group-to-group transfer,
    consistent with the treatment in _off_scatter. The anisotropic
    correction for within-group scattering is handled separately inside
    the sweep via flux_moments accumulation.

    Parameters
    ----------
    flux : numpy.ndarray, shape (cells_x, groups_c)
        Current scalar flux iterate.
    flux_old : numpy.ndarray, shape (cells_x, groups_c)
        Previous scalar flux iterate.
    medium_map : numpy.ndarray, shape (cells_x,)
        Material index per cell.
    xs_scatter : numpy.ndarray, shape (n_materials, groups, groups, L+1)
        Legendre moments of scatter cross sections.
    off_scatter : numpy.ndarray, shape (cells_x,)
        Off-diagonal scatter accumulator, zeroed and filled in-place.
    gg : int
        Current group index being solved.
    edges_gidx_c : numpy.ndarray, shape (groups_c + 1,)
        Coarse-to-fine group boundary indices.

    Returns
    -------
    None
        The off-diagonal scatter source is written into ``off_scatter``.
    """
    cells_x, groups_c = flux.shape
    ii = numba.int32
    mat = numba.int32
    og = numba.int32

    # Zero out previous off scatter term
    off_scatter *= 0.0

    for og in range(groups_c):
        idx1 = edges_gidx_c[og]
        idx2 = edges_gidx_c[og + 1]

        if og < gg:
            for ii in range(cells_x):
                mat = medium_map[ii]
                # L=0 moment only for group-to-group transfer
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
    """Collapse fine-group flux into a coarse-group representation.

    Parameters
    ----------
    fine_flux : numpy.ndarray
        Fine-group flux of shape ``(cells_x, groups)``.
    coarse_flux : numpy.ndarray
        Output coarse-group flux of shape ``(cells_x, groups_c)``.
    edges_gidx_c : numpy.ndarray
        Fine-group edge indices for each coarse group with shape
        ``(groups_c + 1,)``.

    Returns
    -------
    None
        The coarsened flux is written into ``coarse_flux`` in-place.
    """
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

    Returns
    -------
    None
        The reflected edge value is written into ``reflector`` when needed.
    """

    # Get opposite angle
    reflected_idx = numba.int32(angle_x.shape[0] - nn - 1)
    if (angle_x[nn] > 0.0 and bc_x[1] == 1) or (angle_x[nn] < 0.0 and bc_x[0] == 1):
        reflector[reflected_idx] = edge


@numba.jit(
    "void(f8[:,:,:], f8[:,:], f8[:,:,:], i4[:], f8[:,:,:])", nopython=True, cache=True
)
def _source_total(source, flux, xs_scatter, medium_map, external):
    """Build the total isotropic source from scatter and external terms.

    Parameters
    ----------
    source : numpy.ndarray
        Output source array of shape ``(cells_x, angles, groups)``.
    flux : numpy.ndarray
        Scalar flux array of shape ``(cells_x, groups)``.
    xs_scatter : numpy.ndarray
        Scattering matrix of shape ``(materials, groups, groups)``.
    medium_map : numpy.ndarray
        Material index per cell with shape ``(cells_x,)``.
    external : numpy.ndarray
        External source of shape ``(cells_x, angles or 1, groups or 1)``.

    Returns
    -------
    None
        The total source is written into ``source`` in-place.
    """
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


@numba.jit(
    "void(f8[:,:,:], f8[:,:], f8[:,:,:,:], i4[:], f8[:,:,:], f8[:,:])",
    nopython=True,
    cache=True,
)
def _source_total_aniso(source, flux, xs_scatter, medium_map, external, P):
    """Build total source array including anisotropic scatter and external.

    For each cell, group, and angle the source is:

        S(x, mu_n, g) = sum_l (2l+1) * [sum_ig xs_l(x,g,ig) * phi_l(x,ig)]
                        * P_l(mu_n) + external(x, mu_n, g)

    Since only the scalar flux is available here, phi_0(x, g) = flux(x, g)
    and all higher moments are zero. The full anisotropic correction for
    within-group scattering is handled inside the sweep via flux_moments.

    Parameters
    ----------
    source : numpy.ndarray, shape (cells_x, angles, groups)
        Output source array, zeroed and filled in-place.
    flux : numpy.ndarray, shape (cells_x, groups)
        Scalar flux (L=0 moment).
    xs_scatter : numpy.ndarray, shape (n_materials, groups, groups, L+1)
        Legendre moments of scatter cross sections.
    medium_map : numpy.ndarray, shape (cells_x,)
        Material index per cell.
    external : numpy.ndarray, shape (cells_x, angles or 1, groups or 1)
        External source.
    P : numpy.ndarray, shape (L+1, angles)
        Precomputed Legendre polynomials P[l, n] = P_l(mu_n).

    Returns
    -------
    None
        The total source is written into ``source`` in-place.
    """
    # Get parameters
    cells_x, angles, groups = source.shape
    n_moments = numba.int32(xs_scatter.shape[3])
    ii = numba.int32
    nn = numba.int32
    nn_q = numba.int32
    mat = numba.int32
    og = numba.int32
    og_q = numba.int32
    ig = numba.int32
    ll = numba.int32
    group_sum = numba.float64
    scatter = numba.float64

    # Zero out previous source term
    source *= 0.0

    # Iterate over cells
    for ii in range(cells_x):
        mat = medium_map[ii]

        # Iterate over outgoing groups
        for og in range(groups):
            og_q = 0 if external.shape[2] == 1 else og

            # Iterate over angles
            for nn in range(angles):
                nn_q = 0 if external.shape[1] == 1 else nn
                scatter = 0.0

                # Sum over Legendre moments
                for ll in range(n_moments):
                    # Only L=0 contributes since higher flux moments are
                    # unavailable here (no angular flux). Loop is kept
                    # general so it works correctly if higher moments are
                    # ever passed in via an extended flux array.
                    group_sum = 0.0
                    for ig in range(groups):
                        # phi_l(x, ig) = flux[ii, ig] for l=0, 0 otherwise
                        if ll == 0:
                            group_sum += xs_scatter[mat, og, ig, ll] * flux[ii, ig]
                    scatter += (2 * ll + 1) * group_sum * P[ll, nn]

                source[ii, nn, og] = scatter + external[ii, nn_q, og_q]


@numba.jit("void(f8[:,:,:], f8[:,:], f8[:,:,:], i4[:])", nopython=True, cache=True)
def _time_right_side(q_star, flux, xs_scatter, medium_map):
    """Add isotropic scattering contributions to a transient right-hand side.

    Parameters
    ----------
    q_star : numpy.ndarray
        Right-hand side accumulator of shape ``(cells_x, angles, groups)``.
    flux : numpy.ndarray
        Scalar flux array of shape ``(cells_x, groups)``.
    xs_scatter : numpy.ndarray
        Scattering matrix of shape ``(materials, groups, groups)``.
    medium_map : numpy.ndarray
        Material index per cell with shape ``(cells_x,)``.

    Returns
    -------
    None
        ``q_star`` is updated in-place.
    """
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


@numba.jit(
    "void(f8[:,:,:], f8[:,:], f8[:,:,:,:], i4[:], f8[:,:])",
    nopython=True,
    cache=True,
)
def _time_right_side_aniso(q_star, flux, xs_scatter, medium_map, P):
    """Update q_star with anisotropic scatter contribution.

    Adds (sigma_s * phi) to q_star for each cell, angle, and group,
    using the Legendre expansion of the scatter cross section. Since
    only the scalar flux is available here, only the L=0 moment
    contributes; higher moments are zero and their terms vanish.

    Parameters
    ----------
    q_star : numpy.ndarray, shape (cells_x, angles, groups)
        Right-hand side accumulator, updated in-place.
    flux : numpy.ndarray, shape (cells_x, groups)
        Scalar flux (L=0 moment).
    xs_scatter : numpy.ndarray, shape (n_materials, groups, groups, L+1)
        Legendre moments of scatter cross sections.
    medium_map : numpy.ndarray, shape (cells_x,)
        Material index per cell.
    P : numpy.ndarray, shape (L+1, angles)
        Precomputed Legendre polynomials P[l, n] = P_l(mu_n).

    Returns
    -------
    None
        ``q_star`` is updated in-place.
    """
    # Get parameters
    cells_x, angles, groups = q_star.shape
    n_moments = numba.int32(xs_scatter.shape[3])
    ii = numba.int32
    nn = numba.int32
    mat = numba.int32
    og = numba.int32
    ig = numba.int32
    ll = numba.int32
    group_sum = numba.float64
    scatter = numba.float64

    # Iterate over cells
    for ii in range(cells_x):
        mat = medium_map[ii]

        # Iterate over outgoing groups
        for og in range(groups):

            # Iterate over angles — scatter source is now direction-dependent
            for nn in range(angles):
                scatter = 0.0

                # Sum over Legendre moments
                for ll in range(n_moments):
                    # Only L=0 contributes since higher flux moments
                    # are unavailable here (scalar flux only)
                    if ll == 0:
                        group_sum = 0.0
                        for ig in range(groups):
                            group_sum += flux[ii, ig] * xs_scatter[mat, og, ig, ll]
                        scatter += (2 * ll + 1) * group_sum * P[ll, nn]

                q_star[ii, nn, og] += scatter


################################################################################
# Anisotropic Scattering
################################################################################


def legendre_polynomials(n_moments, angle_x):
    """Evaluate all Legendre polynomials P_0..P_{L} at every quadrature point.

    Uses the three-term recurrence relation, avoiding the overhead of the
    recursive ``legendre_polynomial`` function.  Called once per solve.

    Parameters
    ----------
    n_moments : int
        Number of Legendre moments (L+1).
    angle_x : numpy.ndarray, shape (angles,)
        Angular quadrature points on [-1, 1].

    Returns
    -------
    P : numpy.ndarray, shape (n_moments, angles)
        P[l, n] = P_l(mu_n).
    """
    angles = angle_x.shape[0]
    P = np.zeros((n_moments, angles))
    P[0] = 1.0
    if n_moments == 1:
        return P
    P[1] = angle_x
    for ii in range(2, n_moments):
        P[ii] = ((2 * ii - 1) * angle_x * P[ii - 1] - (ii - 1) * P[ii - 2]) / ii
    return P


################################################################################
# Multigroup functions - DMD
################################################################################


def _svd_dmd(A, K):
    """Compute the truncated SVD factors used by the DMD extrapolation.

    Parameters
    ----------
    A : numpy.ndarray
        Snapshot matrix with shape ``(n_state, n_snapshots)``.
    K : int
        Maximum rank retained for the decomposition.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        Left singular vectors, inverse singular-value diagonal matrix, and
        right singular vectors truncated to the retained rank.
    """
    residual = 1e-09

    # Compute SVD
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    # U, S, Vt = svd(A, full_matrices=False, check_finite=False)

    # Find the non-zero singular values
    energy = np.cumsum(S**2) / np.sum(S**2)
    mask = (1 - energy) > residual
    spos = S[mask] if mask.any() else S[S > 0]

    # Create diagonal matrix
    mat_size = min(K, len(spos))

    # Select the u and v that correspond with the nonzero singular values
    U = U[:, :mat_size]
    Vt = Vt[:mat_size, :]

    # S will be the inverse of the singular value diagonal matrix
    S_inv = np.diag(1 / spos[:mat_size])

    return U, S_inv, Vt


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
        Snapshot difference array with shape (n_cells * n_groups, K-1)
        containing previous (backward) differences.
    y_plus : numpy.ndarray
        Snapshot difference array with shape (n_cells * n_groups, K-1)
        containing forward differences.
    K : int
        Number of snapshots (including the base) used for DMD.

    Returns
    -------
    numpy.ndarray
        Extrapolated flux array with shape (n_cells, n_groups).
    """
    # Call SVD
    U, S_inv, Vt = _svd_dmd(y_minus, K)

    # Calculate Atilde
    Atilde = U.T @ y_plus @ Vt.T @ S_inv

    # Calculate delta_y
    rhs = U.T @ y_plus[:, -1]
    delta_y = np.linalg.solve(np.eye(Atilde.shape[0]) - Atilde, rhs)

    # Estimate new flux
    flux = (flux_old.flatten() - y_plus[:, K - 2]) + (U @ delta_y)

    return flux.reshape(flux_old.shape)


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
    """Update $k_{eff}$ using matrix-form fission production data.

    Parameters
    ----------
    flux : numpy.ndarray
        Current flux iterate with shape ``(cells_x, groups)``.
    flux_old : numpy.ndarray
        Previous flux iterate with shape ``(cells_x, groups)``.
    xs_fission : numpy.ndarray
        Fission production matrix of shape ``(materials, groups, groups)``.
    medium_map : numpy.ndarray
        Material index per cell with shape ``(cells_x,)``.
    keff : float
        Previous estimate of the effective multiplication factor.

    Returns
    -------
    float
        Updated effective multiplication factor.
    """
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
    """Update $k_{eff}$ using vector-form fission data.

    Parameters
    ----------
    flux : numpy.ndarray
        Current flux iterate with shape ``(cells_x, groups)``.
    flux_old : numpy.ndarray
        Previous flux iterate with shape ``(cells_x, groups)``.
    chi : numpy.ndarray
        Fission spectrum of shape ``(materials, groups)``.
    nusigf : numpy.ndarray
        Production vector of shape ``(materials, groups)``.
    medium_map : numpy.ndarray
        Material index per cell with shape ``(cells_x,)``.
    keff : float
        Previous estimate of the effective multiplication factor.

    Returns
    -------
    float
        Updated effective multiplication factor.
    """
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
    """Project fine-group fission production into coarse-group source bins.

    Parameters
    ----------
    flux_u : numpy.ndarray
        Fine-group uncollided flux of shape ``(cells_x, groups)``.
    xs_fission : numpy.ndarray
        Fine-group fission matrix of shape ``(materials, groups, groups)``.
    source_c : numpy.ndarray
        Coarse-group source accumulator of shape ``(cells_x, 1, coarse_groups)``.
    medium_map : numpy.ndarray
        Material index per cell with shape ``(cells_x,)``.
    keff : float
        Effective multiplication factor.
    coarse_idx : numpy.ndarray
        Mapping from fine group to coarse group with shape ``(groups,)``.

    Returns
    -------
    numpy.ndarray
        Updated coarse-group fission source array.
    """
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
    """Reconstruct fine-group flux from coarse-group corrections.

    Parameters
    ----------
    flux_u : numpy.ndarray
        Fine-group flux updated in-place with shape ``(cells_x, groups)``.
    flux_c : numpy.ndarray
        Coarse-group flux of shape ``(cells_x, coarse_groups)``.
    coarse_idx : numpy.ndarray
        Mapping from fine group to coarse group with shape ``(groups,)``.
    factor : numpy.ndarray
        Per-group scaling factors with shape ``(groups,)``.

    Returns
    -------
    None
        ``flux_u`` is overwritten with the combined fine-group flux.
    """
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
    label : list, optional
        Parameter/label array for parametric ML predictions. Passed to
        model.predict() when making predictions. Can be a list (for one model
        or all models having the same label) or a list of lists (label for
        each model). Default is None.

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

    one_label = True if isinstance(label[0], float) else False

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
        if one_label:
            mat_flux = model.predict(mat_flux, label=label)
        else:
            mat_flux = model.predict(mat_flux, label=label[nn])
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
    """Build the collided coarse-group source from isotropic scatter.

    Parameters
    ----------
    flux_u : numpy.ndarray
        Uncollided fine-group flux of shape ``(cells_x, groups)``.
    xs_scatter : numpy.ndarray
        Fine-group scattering matrix of shape ``(materials, groups, groups)``.
    source_c : numpy.ndarray
        Coarse-group source accumulator of shape ``(cells_x, 1, coarse_groups)``.
    medium_map : numpy.ndarray
        Material index per cell with shape ``(cells_x,)``.
    coarse_idx : numpy.ndarray
        Mapping from fine group to coarse group with shape ``(groups,)``.

    Returns
    -------
    None
        The collided coarse-group source is written into ``source_c``.
    """
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
    "void(f8[:,:], f8[:,:,:,:], f8[:,:,:], i4[:], i4[:])",
    nopython=True,
    cache=True,
)
def _hybrid_source_collided_aniso(flux_u, xs_scatter, source_c, medium_map, coarse_idx):
    """Build collided source with anisotropic scatter.

    Accumulates the scatter source from the uncollided flux into the
    coarse-group collided source. Since source_c has a singleton angle
    dimension (shape cells_x, 1, coarse_groups), the scatter source is
    direction-integrated (L=0 only), consistent with the original.

    Parameters
    ----------
    flux_u : numpy.ndarray, shape (cells_x, groups)
        Uncollided scalar flux.
    xs_scatter : numpy.ndarray, shape (n_materials, groups, groups, L+1)
        Legendre moments of scatter cross sections.
    source_c : numpy.ndarray, shape (cells_x, 1, coarse_groups)
        Collided source accumulator, zeroed and filled in-place.
    medium_map : numpy.ndarray, shape (cells_x,)
        Material index per cell.
    coarse_idx : numpy.ndarray, shape (groups,)
        Maps fine group og to coarse group index.

    Returns
    -------
    None
        The collided coarse-group source is written into ``source_c``.
    """
    cells_x, groups = flux_u.shape
    ii = numba.int32
    mat = numba.int32
    og = numba.int32
    ig = numba.int32

    # Zero out previous source
    source_c *= 0.0

    for ii in range(cells_x):
        mat = medium_map[ii]
        for og in range(groups):
            for ig in range(groups):
                # L=0 moment only - source_c has singleton angle dimension
                # so the scatter source is direction-integrated here.
                # The (2*0+1)=1 prefactor is implicit.
                source_c[ii, 0, coarse_idx[og]] += (
                    flux_u[ii, ig] * xs_scatter[mat, og, ig, 0]
                )


@numba.jit(
    "void(f8[:,:], f8[:,:], f8[:,:,:], f8[:,:,:], i4[:], i4[:], f8[:])",
    nopython=True,
    cache=True,
)
def _hybrid_source_total(
    flux_u, flux_c, xs_scatter_u, q_star, medium_map, coarse_idx, factor
):
    """Add hybrid scatter source contributions to the transport right-hand side.

    Parameters
    ----------
    flux_u : numpy.ndarray
        Fine-group uncollided flux of shape ``(cells_x, groups)``.
    flux_c : numpy.ndarray
        Coarse-group collided flux of shape ``(cells_x, coarse_groups)``.
    xs_scatter_u : numpy.ndarray
        Fine-group scattering matrix of shape ``(materials, groups, groups)``.
    q_star : numpy.ndarray
        Right-hand side accumulator of shape ``(cells_x, angles, groups)``.
    medium_map : numpy.ndarray
        Material index per cell with shape ``(cells_x,)``.
    coarse_idx : numpy.ndarray
        Mapping from fine group to coarse group with shape ``(groups,)``.
    factor : numpy.ndarray
        Per-group scaling factors with shape ``(groups,)``.

    Returns
    -------
    None
        ``flux_u`` and ``q_star`` are updated in-place.
    """
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


@numba.jit(
    "void(f8[:,:], f8[:,:], f8[:,:,:,:], f8[:,:,:], i4[:], i4[:], f8[:], f8[:,:])",
    nopython=True,
    cache=True,
)
def _hybrid_source_total_aniso(
    flux_u, flux_c, xs_scatter_u, q_star, medium_map, coarse_idx, factor, P
):
    """Build total hybrid source with anisotropic scatter.

    Updates flux_u by adding the coarse-group correction, then accumulates
    the anisotropic scatter source into q_star for each cell, angle, and
    group. Since only the scalar flux is available here, only the L=0
    moment contributes; the angle dependence enters via P[ll, nn].

    Parameters
    ----------
    flux_u : numpy.ndarray, shape (cells_x, groups)
        Uncollided scalar flux, updated in-place with coarse correction.
    flux_c : numpy.ndarray, shape (cells_x, coarse_groups)
        Collided scalar flux on coarse group structure.
    xs_scatter_u : numpy.ndarray, shape (n_materials, groups, groups, L+1)
        Legendre moments of fine-group scatter cross sections.
    q_star : numpy.ndarray, shape (cells_x, angles, groups)
        Right-hand side accumulator, updated in-place.
    medium_map : numpy.ndarray, shape (cells_x,)
        Material index per cell.
    coarse_idx : numpy.ndarray, shape (groups,)
        Maps fine group og to coarse group index.
    factor : numpy.ndarray, shape (groups,)
        Scaling factor per fine group for coarse-to-fine mapping.
    P : numpy.ndarray, shape (L+1, angles)
        Precomputed Legendre polynomials P[l, n] = P_l(mu_n).

    Returns
    -------
    None
        ``flux_u`` and ``q_star`` are updated in-place.
    """
    cells_x, angles, groups = q_star.shape
    n_moments = numba.int32(xs_scatter_u.shape[3])
    ii = numba.int32
    nn = numba.int32
    mat = numba.int32
    og = numba.int32
    ig = numba.int32
    ll = numba.int32
    group_sum = numba.float64
    scatter = numba.float64

    for ii in range(cells_x):
        mat = medium_map[ii]

        # Update uncollided flux with coarse-group correction
        for og in range(groups):
            flux_u[ii, og] = flux_u[ii, og] + flux_c[ii, coarse_idx[og]] * factor[og]

        # Build anisotropic scatter source and accumulate into q_star
        for og in range(groups):
            for nn in range(angles):
                scatter = 0.0
                for ll in range(n_moments):
                    # Only L=0 contributes since only scalar flux is
                    # available here; higher moment terms are zero
                    if ll == 0:
                        group_sum = 0.0
                        for ig in range(groups):
                            group_sum += flux_u[ii, ig] * xs_scatter_u[mat, og, ig, ll]
                        scatter += (2 * ll + 1) * group_sum * P[ll, nn]
                q_star[ii, nn, og] += scatter


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
    """Compute off-diagonal scatter for a zero-dimensional group solve.

    Parameters
    ----------
    flux : numpy.ndarray
        Current scalar flux iterate with shape ``(groups,)``.
    flux_old : numpy.ndarray
        Previous scalar flux iterate with shape ``(groups,)``.
    xs_scatter : numpy.ndarray
        Scattering matrix of shape ``(groups, groups)``.
    gg : int
        Energy group currently being solved.

    Returns
    -------
    float
        Off-diagonal scatter source for group ``gg``.
    """
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
    """Build the zero-dimensional fission source from a full fission matrix.

    Parameters
    ----------
    flux : numpy.ndarray
        Scalar flux vector with shape ``(groups,)``.
    xs_fission : numpy.ndarray
        Fission matrix of shape ``(groups, groups)``.
    source : numpy.ndarray
        Output source array of shape ``(1, groups)``.
    keff : float
        Effective multiplication factor.

    Returns
    -------
    numpy.ndarray
        Updated zero-dimensional fission source.
    """
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
    """Build the zero-dimensional fission source from vector-form data.

    Parameters
    ----------
    flux : numpy.ndarray
        Scalar flux vector with shape ``(groups,)``.
    chi : numpy.ndarray
        Fission spectrum with shape ``(groups,)``.
    nusigf : numpy.ndarray
        Fission production vector with shape ``(groups,)``.
    source : numpy.ndarray
        Output source array of shape ``(1, groups)``.
    keff : float
        Effective multiplication factor.

    Returns
    -------
    numpy.ndarray
        Updated zero-dimensional fission source.
    """
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
    """Update $k_{eff}$ for a zero-dimensional matrix fission model.

    Parameters
    ----------
    flux : numpy.ndarray
        Current flux iterate with shape ``(groups,)``.
    flux_old : numpy.ndarray
        Previous flux iterate with shape ``(groups,)``.
    xs_fission : numpy.ndarray
        Fission matrix of shape ``(groups, groups)``.
    keff : float
        Previous estimate of the effective multiplication factor.

    Returns
    -------
    float
        Updated effective multiplication factor.
    """
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
    """Update $k_{eff}$ for a zero-dimensional vector fission model.

    Parameters
    ----------
    flux : numpy.ndarray
        Current flux iterate with shape ``(groups,)``.
    flux_old : numpy.ndarray
        Previous flux iterate with shape ``(groups,)``.
    chi : numpy.ndarray
        Fission spectrum with shape ``(groups,)``.
    nusigf : numpy.ndarray
        Fission production vector with shape ``(groups,)``.
    keff : float
        Previous estimate of the effective multiplication factor.

    Returns
    -------
    float
        Updated effective multiplication factor.
    """
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
    return (rate_new * keff) / rate_old
    return (rate_new * keff) / rate_old
