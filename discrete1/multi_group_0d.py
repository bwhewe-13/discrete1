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


def source_iteration(flux_old, xs_total, xs_scatter, external, angle_x, angle_w):
    """Multigroup source iteration solver.

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
            flux[gg] = discrete_ordinates(
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


def discrete_ordinates(
    flux_old, xs_total, xs_scatter, off_scatter, external, angle_x, angle_w
):
    """Dispatch to the appropriate angular sweep.

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
