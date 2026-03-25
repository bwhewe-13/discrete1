"""Infinite medium multigroup solver.

This module provides multigroup-level driver routines that orchestrate
outer iterations over energy groups and angles (discrete ordinates).

Public routines include:
- source_iteration: standard multigroup source iteration
- discrete_ordinates: discrete ordinates for one group

The multigroup drivers expect arrays shaped as used by the rest of
the package (angular and group axes as described in function docstrings).
"""

import numpy as np

from discrete1 import tools

count_nn = 100
change_nn = 1e-12

count_gg = 100
change_gg = 1e-08


def source_iteration(
    flux_old, xs_total, xs_scatter, external, angle_x, angle_w, P=None, P_weights=None
):
    """Multigroup source iteration solver - dispatches to iso or aniso variant.

    Performs outer iterations over energy groups using inner discrete
    ordinates sweeps for each group. Off-diagonal scatter (up- and
    down-scatter) contributions are handled via an "off_scatter"
    accumulator that uses the previous and current flux iterates.

    Parameters
    ----------
    flux_old : numpy.ndarray
        Initial scalar flux guess with shape (n_groups,).
    xs_total : array_like
        Total cross sections (n_groups,).
    xs_scatter : array_like
        Scatter cross sections (n_groups, n_groups).
    external : numpy.ndarray
        External source array (n_angles, n_groups) or with
        singleton dimensions where appropriate.
    angle_x : array_like
        Angular ordinates (n_angles,).
    angle_w : array_like
        Quadrature weights (n_angles,).
    P : numpy.ndarray, shape (L+1, angles), optional
        Precomputed Legendre polynomials. Required for anisotropic.
    P_weights : numpy.ndarray, shape (L+1, angles), optional
        Precomputed w_n * P_l(mu_n). Required for anisotropic.

    Returns
    -------
    numpy.ndarray, shape (n_groups,)
        Converged scalar flux.
    """
    if xs_scatter.ndim == 2:
        return source_iteration_iso(
            flux_old,
            xs_total,
            xs_scatter,
            external,
            angle_x,
            angle_w,
        )
    else:
        if P is None:
            n_moments = xs_scatter.shape[2]
            P = tools.legendre_polynomials(n_moments, angle_x)
            P_weights = angle_w[np.newaxis, :] * P
        return source_iteration_aniso(
            flux_old,
            xs_total,
            xs_scatter,
            external,
            angle_x,
            angle_w,
            P,
            P_weights,
        )


def source_iteration_iso(flux_old, xs_total, xs_scatter, external, angle_x, angle_w):
    """Multigroup source iteration solver for isotropic scattering.

    Performs outer iterations over energy groups using inner discrete
    ordinates sweeps for each group. Off-diagonal scatter (up- and
    down-scatter) contributions are handled via an "off_scatter"
    accumulator that uses the previous and current flux iterates.

    Parameters
    ----------
    flux_old : numpy.ndarray
        Initial scalar flux guess with shape (n_groups,).
    xs_total : array_like
        Total cross sections (n_groups,).
    xs_scatter : array_like
        Scatter cross sections (n_groups, n_groups).
    external : numpy.ndarray
        External source array (n_angles, n_groups) or with
        singleton dimensions where appropriate.
    angle_x : array_like
        Angular ordinates (n_angles,).
    angle_w : array_like
        Quadrature weights (n_angles,).

    Returns
    -------
    numpy.ndarray
        Converged scalar flux array with shape (n_groups,).
    """
    groups = flux_old.shape[0]
    flux = np.zeros((groups,))
    off_scatter = 0.0

    converged = False
    count = 1
    change = 0.0

    while not (converged):
        flux *= 0.0

        for gg in range(groups):
            # Check for sizes
            qq = 0 if external.shape[1] == 1 else gg

            # Update off scatter source
            off_scatter = tools._off_scatter_0d(flux, flux_old, xs_scatter, gg)

            # Run discrete ordinates for one group
            flux[gg] = discrete_ordinates_iso(
                flux_old[gg],
                xs_total[gg],
                xs_scatter[gg, gg],
                off_scatter,
                external[:, qq],
                angle_x,
                angle_w,
            )

        # Check for convergence
        try:
            change = np.linalg.norm((flux - flux_old) / flux)
        except RuntimeWarning:
            change = 0.0
        converged = (change < change_gg) or (count >= count_gg)
        count += 1

        flux_old = flux.copy()

    return flux


def discrete_ordinates_iso(
    flux_old, xs_total, xs_scatter, off_scatter, external, angle_x, angle_w
):
    """Angular sweep for isotropic scattering.

    This function returns the updated scalar flux array evaluated
    at a specific energy group.

    Parameters
    ----------
    flux_old : float
        Previous iterate of the scalar flux.
    xs_total : float
        Total macroscopic cross sections per material.
    xs_scatter : float
        Scatter cross section used in source evaluation.
    off_scatter : float
        Off-diagonal scatter correction/source term (shape: n_cells,).
    external : numpy.ndarray
        External source array (n_angles, n_groups).
    angle_x : numpy.ndarray
        Angular ordinates (n_angles,).
    angle_w : numpy.ndarray
        Quadrature weights (n_angles,).

    Returns
    -------
    float
        Updated scalar flux for energy group.
    """
    flux = 0.0
    angles = angle_x.shape[0]

    converged = False
    count = 1
    change = 0.0

    while not (converged):

        flux = 0.0

        for nn in range(angles):

            qq = 0 if external.shape[0] == 1 else nn
            flux += (
                (xs_scatter * flux_old + external[qq] + off_scatter)
                * angle_w[nn]
                / xs_total
            )

        # Check for convergence
        try:
            change = np.linalg.norm((flux - flux_old) / flux)
        except RuntimeWarning:
            change = 0.0
        converged = (change < change_nn) or (count >= count_nn)
        count += 1

        flux_old = flux

    return flux


def source_iteration_aniso(
    flux_old, xs_total, xs_scatter, external, angle_x, angle_w, P, P_weights
):
    """Multigroup source iteration for anisotropic scattering.

    Identical workflow to :func:`source_iteration_iso` but each group
    sweep uses the full Legendre expansion of the scatter kernel. The
    within-group anisotropic correction is applied inside the angular
    sweep; only the L=0 moment drives the off-diagonal scatter term.

    Parameters
    ----------
    flux_old : numpy.ndarray
        Initial scalar flux guess with shape (n_groups,).
    xs_total : array_like
        Total cross sections (n_groups,).
    xs_scatter : array_like
        Scatter cross sections (n_groups, n_groups).
    external : numpy.ndarray
        External source array (n_angles, n_groups) or with
        singleton dimensions where appropriate.
    angle_x : array_like
        Angular ordinates (n_angles,).
    angle_w : array_like
        Quadrature weights (n_angles,).
    P : numpy.ndarray, shape (L+1, n_angles)
        Precomputed Legendre polynomials ``P[l, n]`` = P_l(mu_n).
    P_weights : numpy.ndarray, shape (L+1, n_angles)
        Precomputed ``w_n * P_l(mu_n)`` used for flux moment accumulation.

    Returns
    -------
    numpy.ndarray
        Converged scalar flux array with shape (n_groups,).
    """
    groups = flux_old.shape[0]
    flux = np.zeros((groups,))
    off_scatter = 0.0
    legendre_weights = (2 * np.arange(xs_scatter.shape[2]) + 1)[None, :]

    converged = False
    count = 1
    change = 0.0

    while not (converged):
        flux *= 0.0

        for gg in range(groups):
            # Check for sizes
            qq = 0 if external.shape[1] == 1 else gg

            # Update off scatter source
            off_scatter = tools._off_scatter_0d(flux, flux_old, xs_scatter[..., 0], gg)

            # Within-group Legendre moments mapped to cells -> (cells_x, L+1)
            xs_self_scatter = xs_scatter[gg, gg, :]

            # Precompute (2l+1) * xs_l(x) — fixed for the inner sweep loop
            xs_scatter_w = xs_self_scatter * legendre_weights

            # Run discrete ordinates for one group
            flux[gg] = discrete_ordinates_aniso(
                flux_old[gg],
                xs_total[gg],
                xs_self_scatter,
                xs_scatter_w,
                off_scatter,
                external[:, qq],
                angle_x,
                angle_w,
                P,
                P_weights,
            )

        # Check for convergence
        try:
            change = np.linalg.norm((flux - flux_old) / flux)
        except RuntimeWarning:
            change = 0.0
        converged = (change < change_gg) or (count >= count_gg)
        count += 1

        flux_old = flux.copy()

    return flux


def discrete_ordinates_aniso(
    flux_old,
    xs_total,
    xs_self_scatter,
    xs_scatter_w,
    off_scatter,
    external,
    angle_x,
    angle_w,
    P,
    P_weights,
):
    """Angular sweep for anisotropic scattering.

    This function returns the updated scalar flux array evaluated
    at a specific energy group.

    Parameters
    ----------
    flux_old : float
        Previous iterate of the scalar flux.
    xs_total : float
        Total macroscopic cross sections per material.
    xs_self_scatter : float
        Within-group Legendre scatter moments
    xs_scatter_w : numpy.ndarray, shape (L+1,)
        Precomputed (2l+1) * xs_self_scatter.
    off_scatter : float
        Off-diagonal scatter correction/source term (shape: n_cells,).
    external : numpy.ndarray
        External source array (n_angles, n_groups).
    angle_x : numpy.ndarray
        Angular ordinates (n_angles,).
    angle_w : numpy.ndarray
        Quadrature weights (n_angles,).
    P : numpy.ndarray, shape (L+1, angles)
        Precomputed P[l, n] = P_l(mu_n).
    P_weights : numpy.ndarray, shape (L+1, angles)
        Precomputed w_n * P_l(mu_n).

    Returns
    -------
    float
        Updated scalar flux for energy group.
    """
    flux = 0.0
    angles = angle_x.shape[0]
    n_moments = xs_self_scatter.shape[0]

    # Seed L=0 flux moment from initial guess; higher moments start zero.
    flux_moments = np.zeros((1, n_moments))
    flux_moments[:, 0] = flux_old
    anisotropic = np.zeros((1, angles))

    converged = False
    count = 1
    change = 0.0

    while not (converged):
        flux = 0.0

        # Build anisotropic scatter source from moments of previous sweep.
        anisotropic = np.einsum("xl,xl,ln->xn", xs_scatter_w, flux_moments, P)

        # Reset moment accumulator for this sweep.
        flux_moments *= 0.0

        for nn in range(angles):
            qq = 0 if external.shape[0] == 1 else nn
            angular = (anisotropic[0, nn] + external[qq] + off_scatter) / xs_total
            flux += angular * angle_w[nn]

            # Accumulate flux moments: one rank-1 outer-product per ordinate.
            flux_moments += np.outer(angular, P_weights[:, nn])

        # Check for convergence
        try:
            change = np.linalg.norm((flux - flux_old) / flux)
        except RuntimeWarning:
            change = 0.0
        converged = (change < change_nn) or (count >= count_nn)
        count += 1

        flux_old = flux

    return flux
