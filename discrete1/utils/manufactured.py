"""Functions for manufactured solutions.

Provides solutions for given steady state and time-dependent problems.
Includes functions to calculate the order of accuracy.
"""

import numpy as np


def solution_ss_01(x, angle_x):
    """One material, single direction."""
    flux = np.zeros((len(x), len(angle_x)))
    for n, mu in enumerate(angle_x):
        if mu > 0:
            flux[:, n] = 1.0
        else:
            flux[:, n] = 1 - np.exp((1 - x) / mu)
    return flux


def solution_ss_02(x, angle_x):
    """One material, angular dependent."""
    flux = np.zeros((len(x), len(angle_x)))
    for n, mu in enumerate(angle_x):
        if mu > 0:
            flux[:, n] = 0.5 + 0.5 * np.exp(-x / mu)
        else:
            flux[:, n] = 0.5 - 0.5 * np.exp((1 - x) / mu)
    return flux


def solution_ss_03(x, angle_x):
    """One material, angular dependent, with source."""
    flux = np.zeros((len(x), len(angle_x)))
    for n, mu in enumerate(angle_x):
        flux[:, n] = 0.5 + 0.25 * x**2 * np.exp(mu)
    return flux


def solution_ss_04(x, angle_x):
    """Two materials, angular independent."""
    length_x = 2
    flux = np.zeros((len(x), len(angle_x)))
    for n, mu in enumerate(angle_x):
        flux[x <= 1, n] = -2 * x[x <= 1] ** 2 + 2 * length_x * x[x <= 1]
        flux[x > 1, n] = 0.25 * x[x > 1] - 0.125 * length_x + 0.5 * length_x**2
    return flux


def solution_ss_05(x, angle_x):
    """Two materials, angular dependent."""
    length_x = 2
    flux = np.zeros((len(x), len(angle_x)))
    for n, mu in enumerate(angle_x):
        flux[x <= 1, n] = -2 * np.exp(mu) * x[x <= 1] ** 2 + 2 * length_x**2 * x[x <= 1]
        flux[x > 1, n] = length_x * np.exp(mu) * x[x > 1] + length_x**2 * (
            length_x - np.exp(mu)
        )
    return flux


def solution_td_01(x, angle_x, edges_t):
    """One material, time dependent, angular independent."""
    flux = np.zeros((edges_t.shape[0], x.shape[0], angle_x.shape[0], 1))
    for cc, tt in enumerate(edges_t):
        for nn, mu in enumerate(angle_x):
            flux[cc, :, nn, 0] = (-x) * (x - 2) * np.sin(x - 0.1 * tt) + 2
    return flux


def solution_td_02(x, angle_x, edges_t):
    """One material, time dependent, angular dependent."""
    flux = np.zeros((edges_t.shape[0], x.shape[0], angle_x.shape[0], 1))
    for cc, tt in enumerate(edges_t):
        for nn, mu in enumerate(angle_x):
            flux[cc, :, nn, 0] = 1 + np.sin(x - 0.5 * tt) + np.cos(mu)
    return flux


########################################################################
# Manufactured Solutions and Accuracy
########################################################################
def spatial_error(approx, reference, ndims=1):
    """Calculate the spatial error between approximate and reference solutions.

    Parameters
    ----------
    approx : numpy.ndarray
        Approximate flux.
    reference : numpy.ndarray
        Reference flux.
    ndims : int
        Number of spatial dimensions for flux (1 or 2).

    Returns
    -------
    float
        L2 error normalized for the number of spatial cells.
    """
    assert approx.shape == reference.shape, "Not the same array shape"
    if ndims == 1:
        normalized = (approx.shape[0]) ** (-0.5)
    elif ndims == 2:
        normalized = (approx.shape[0] * approx.shape[1]) ** (-0.5)
    return normalized * np.linalg.norm(approx - reference)


def order_accuracy(error1, error2, ratio):
    """Calculate order of accuracy.

    Finding the order of accuracy between errors on different grids
    where error2 is the refined grid.

    Parameters
    ----------
    error1 : float
        Error between an approximate solution and the reference solution on the
        same grid.
    error2 : float
        Error between an approximate solution and the reference solution on a
        more refined grid.
    ratio : float
        Ratio between the spatial cell width of the error1 grid and the error2
        grid (delta x1 / delta x2).

    Returns
    -------
    float
        Order of accuracy.
    """
    return np.log(error1 / error2) / np.log(ratio)


def wynn_epsilon(lst, rank):
    """Perform Wynn Epsilon Convergence Algorithm.

    Parameters
    ----------
    lst : list
        List of float type.
        Values for convergence.
    rank : int
        Rank of system.

    Returns
    -------
    numpy.ndarray
        2D array with float type of shape (2 * rank + 2, 2 * rank + 2).
        Diagonal is convergence.
    """
    N = 2 * rank + 1
    error = np.zeros((N + 1, N + 1))
    for ii in range(1, N + 1):
        error[ii, 1] = lst[ii - 1]
    for ii in range(3, N + 2):
        for jj in range(3, ii + 1):
            if (error[ii - 1, jj - 2] - error[ii - 2, jj - 2]) == 0.0:
                error[ii - 1, jj - 1] = error[ii - 2, jj - 3]
            else:
                error[ii - 1, jj - 1] = error[ii - 2, jj - 3] + 1 / (
                    error[ii - 1, jj - 2] - error[ii - 2, jj - 2]
                )
    return abs(error[-1, -1])
