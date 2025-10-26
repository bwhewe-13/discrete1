"""Built-in boundary condition factories for discrete1.

This module provides a set of reusable boundary condition constructors
used across tests and examples. Each function returns arrays shaped to
match the solver interfaces (spatial boundaries, angles, energy groups,
and optional time axes).

Public functions
- manufactured_ss_03, manufactured_ss_04, manufactured_ss_05
- manufactured_td_02
- deuterium_deuterium, deuterium_tritium
- time_dependence_constant, time_dependence_decay_01, time_dependence_decay_02

All functions return NumPy arrays suitable as boundary source inputs to
the solver.
"""

import math

import numpy as np


def manufactured_ss_03(angle_x):
    """Manufactured steady-state boundary (case 03).

    One-group, angle-dependent boundary used for manufactured solution tests.

    Parameters
    ----------
    angle_x : array_like
        1D array of angular ordinates (mu values) with shape (n_angles,).

    Returns
    -------
    numpy.ndarray
        Boundary array with shape (2, n_angles, 1). Axis 0 indexes the
        two spatial boundaries (left, right).
    """
    # One group, angle dependent boundary
    boundary_x = np.zeros((2, angle_x.shape[0], 1))
    boundary_x[0, :, 0] = 0.5
    boundary_x[1, :, 0] = 0.5 + 0.25 * np.exp(angle_x)
    return boundary_x


def manufactured_ss_04():
    """Manufactured steady-state boundary (case 04).

    One-group, angle-independent boundary used for manufactured solution tests.

    Returns
    -------
    numpy.ndarray
        Boundary array with shape (2, 1, 1).
    """
    # One group, angle independent boundary
    length_x = 2.0
    boundary_x = np.zeros((2, 1, 1))
    boundary_x[1] = 0.5 * length_x**2 + 0.125 * length_x
    return boundary_x


def manufactured_ss_05():
    """Manufactured steady-state boundary (case 05).

    One-group, angle-independent boundary (polynomial) used for tests.

    Returns
    -------
    numpy.ndarray
        Boundary array with shape (2, 1, 1).
    """
    # One group, angle independent boundary
    length_x = 2.0
    boundary_x = np.zeros((2, 1, 1))
    boundary_x[1] = length_x**3
    return boundary_x


def manufactured_td_02(angle_x, edges_t):
    """Manufactured time-dependent boundary (case TD_02).

    Builds a time- and angle-dependent boundary array for manufactured
    time-dependent tests.

    Parameters
    ----------
    angle_x : array_like
        1D array of angular ordinates (mu values).
    edges_t : array_like
        1D array of time edge values. The returned array has one entry per
        time edge.

    Returns
    -------
    numpy.ndarray
        Boundary array with shape (n_time_edges, 2, n_angles, 1).
        The two entries along axis=1 correspond to the two spatial boundaries.
    """
    # Time dependent, one group, angle dependent boundary
    length_x = np.pi
    boundary_x = np.zeros((edges_t.shape[0], 2, angle_x.shape[0], 1))
    for cc, tt in enumerate(edges_t):
        for nn, mu in enumerate(angle_x):
            boundary_x[cc, 0, nn, 0] = 1 + np.sin(0.0 - 0.5 * tt) + np.cos(mu)
            boundary_x[cc, 1, nn, 0] = 1 + np.sin(length_x - 0.5 * tt) + np.cos(mu)
    return boundary_x


def deuterium_deuterium(location, edges_g):
    """Create a monoenergetic boundary source for D-D fusion (2.45 MeV).

    Parameters
    ----------
    location : int
        Spatial boundary index (0 for left, 1 for right) where the source is
        applied.
    edges_g : array_like
        Energy group edge values. Used to determine the group index nearest
        to 2.45 MeV.

    Returns
    -------
    numpy.ndarray
        Boundary array with shape (2, 1, n_groups) with a unit source in the
        selected group at the specified location.
    """
    # Source entering from 2.45 MeV
    group = np.argmin(abs(edges_g - 2.45e6))
    boundary_x = np.zeros((2, 1, edges_g.shape[0] - 1))
    boundary_x[(location, ..., group)] = 1.0
    return boundary_x


def deuterium_tritium(location, edges_g):
    """Create a monoenergetic boundary source for D-T fusion (14.1 MeV).

    Parameters
    ----------
    location : int
        Spatial boundary index (0 for left, 1 for right) where the source is
        applied.
    edges_g : array_like
        Energy group edge values. Used to determine the group index nearest
        to 14.1 MeV.

    Returns
    -------
    numpy.ndarray
        Boundary array with shape (2, 1, n_groups) with a unit source in the
        selected group at the specified location.
    """
    # Source entering from 14.1 MeV
    group = np.argmin(abs(edges_g - 14.1e6))
    boundary_x = np.zeros((2, 1, edges_g.shape[0] - 1))
    boundary_x[(location, ..., group)] = 1.0
    return boundary_x


def time_dependence_constant(boundary_x):
    """Wrap a boundary array to add a time axis (constant in time).

    Parameters
    ----------
    boundary_x : numpy.ndarray
        A boundary array without time dimension. The function will add a
        leading time axis of length 1.

    Returns
    -------
    numpy.ndarray
        Boundary array with a leading time axis (shape: 1, ...).
    """
    return boundary_x[None, ...]


def time_dependence_decay_01(boundary_x, edges_t, off_time):
    """Turn off a previously time-independent boundary after a given time.

    The function repeats the provided boundary over all time steps defined by
    ``edges_t`` and zeros it out for all steps where the time exceeds
    ``off_time``.

    Parameters
    ----------
    boundary_x : numpy.ndarray
        Boundary array without a time axis.
    edges_t : array_like
        Time edge array. Number of time steps is ``edges_t.shape[0] - 1``.
    off_time : float
        Time threshold (same units as ``edges_t``) after which the boundary
        is turned off.

    Returns
    -------
    numpy.ndarray
        Time-dependent boundary array with shape (n_steps, ...).
    """
    # Turn off boundary at specific step
    steps = edges_t.shape[0] - 1
    boundary_x = np.repeat(boundary_x[None, ...], steps, axis=0)
    loc = np.argwhere(edges_t[1:] > off_time).flatten()
    boundary_x[loc, ...] *= 0.0
    return boundary_x


def time_dependence_decay_02(boundary_x, edges_t):
    """Apply a smooth decay (erfc-based) to a boundary over time steps.

    This function identifies non-zero entries in ``boundary_x`` and applies a
    complementary error-function-based decay schedule over the provided
    time edges. Times are converted to microseconds internally for the decay
    logic.

    Parameters
    ----------
    boundary_x : numpy.ndarray
        Boundary array without time axis. Non-zero entries will be decayed.
    edges_t : array_like
        Time edge array. Number of time steps is ``edges_t.shape[0] - 1``.

    Returns
    -------
    numpy.ndarray
        Time-dependent boundary array with shape (n_steps, ...).
    """  # noqa: D202

    # Complementary Error Function
    def erfc(x):
        return 1 - math.erf(x)

    # Turn off boundary by decay
    steps = edges_t.shape[0] - 1
    # Find where boundary != 0
    idx = tuple(np.argwhere(boundary_x != 0.0).flatten())
    # Repeat over all groups
    boundary_x = np.repeat(boundary_x[None, ...], edges_t.shape[0] - 1, axis=0)
    for tt in range(steps):
        # dt = edges_t[tt + 1] - edges_t[tt]
        # Convert to microseconds
        t_us = np.round(edges_t[tt + 1] * 1e6, 12)
        if t_us >= 0.2:
            k = np.ceil((t_us - 0.2) / 0.1)
            err_arg = (t_us - 0.1 * (1 + k)) / (0.01)
            boundary_x[tt][idx] = 0.5**k * (1 + 2 * erfc(err_arg))
    return boundary_x
