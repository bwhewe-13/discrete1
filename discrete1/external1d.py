"""Built-in external source factories used by tests and examples.

This module contains a set of ready-made external source constructors
used across examples, tests, and solver inputs. Functions return NumPy
arrays shaped to match solver expectations (spatial cells, angles,
groups, and optional time axes).

Public functions
- manufactured_ss_03, manufactured_ss_04, manufactured_ss_05
- manufactured_td_01, manufactured_td_02
- reeds, ambe
- time_dependence_constant

Data files for some sources are loaded from the package `sources/`
directory via pkg_resources.
"""

import numpy as np
import pkg_resources

DATA_PATH = pkg_resources.resource_filename("discrete1", "sources/")


def manufactured_ss_03(x, angle_x):
    """Manufactured steady-state source (case 03).

    Constructs a one-group, angle-dependent external source used for
    manufactured-solution tests. The returned array has shape
    (n_x, n_angles, 1).

    Parameters
    ----------
    x : array_like
        Spatial cell centers or coordinates (length n_x).
    angle_x : array_like
        Angular ordinates (length n_angles).

    Returns
    -------
    numpy.ndarray
        External source array with shape (n_x, n_angles, 1).
    """
    # One group, angle dependent source
    external = np.zeros((x.shape[0], angle_x.shape[0], 1))
    const1 = 0.5
    const2 = 0.25

    def func(mu):
        return (
            const2 * mu * np.exp(mu) * 2 * x
            + const1
            + const2 * x**2 * np.exp(mu)
            - 0.9 / 2 * (2 * const1 + const2 * x**2 * (np.exp(1) - np.exp(-1)))
        )

    for nn, mu in enumerate(angle_x):
        external[:, nn, 0] = func(mu)

    return external


def manufactured_ss_04(x, angle_x):
    """Manufactured steady-state source (case 04).

    One-group source with two material regions (quasi and scatter forms).

    Parameters
    ----------
    x : array_like
        Spatial coordinates (n_x,).
    angle_x : array_like
        Angular ordinates (n_angles,).

    Returns
    -------
    numpy.ndarray
        External source array with shape (n_x, n_angles, 1).
    """
    # One group, two material source
    external = np.zeros((x.shape[0], angle_x.shape[0], 1))
    length_x = 2.0

    def _quasi(x, mu):
        c = 0.3
        return (
            2 * length_x * mu
            - 4 * x * mu
            - 2 * x**2
            + 2 * length_x * x
            - c * (-2 * x**2 + 2 * length_x * x)
        )

    def _scatter(x, mu):
        c = 0.9
        const = -0.125 * length_x + 0.5 * length_x**2
        return 0.25 * (mu + x) + const - c * ((0.25 * x + const))

    for nn, mu in enumerate(angle_x):
        idx = x < (0.5 * length_x)
        external[idx, nn, 0] = _quasi(x[idx], mu)
        idx = x > (0.5 * length_x)
        external[idx, nn, 0] = _scatter(x[idx], mu)

    return external


def manufactured_ss_05(x, angle_x):
    """Manufactured steady-state source (case 05).

    One-group manufactured source with region-dependent expressions.

    Parameters
    ----------
    x : array_like
        Spatial coordinates (n_x,).
    angle_x : array_like
        Angular ordinates (n_angles,).

    Returns
    -------
    numpy.ndarray
        External source array with shape (n_x, n_angles, 1).
    """
    external = np.zeros((x.shape[0], angle_x.shape[0], 1))
    length_x = 2.0

    def _quasi(x, mu):
        c = 0.3
        return (
            mu * (2 * length_x**2 - 4 * np.exp(mu) * x)
            - 2 * np.exp(mu) * x**2
            + 2 * length_x**2 * x
            - c / 2 * (-2 * x**2 * (np.exp(1) - np.exp(-1)) + 4 * length_x**2 * x)
        )

    def _scatter(x, mu):
        c = 0.9
        const = length_x**3 - length_x**2 * np.exp(mu)
        return (
            length_x * mu * np.exp(mu)
            + length_x * x * np.exp(mu)
            + const
            - c
            / 2
            * (
                2 * length_x**3
                + (np.exp(1) - np.exp(-1)) * (x * length_x - length_x**2)
            )
        )

    for nn, mu in enumerate(angle_x):
        idx = x < (0.5 * length_x)
        external[idx, nn, 0] = _quasi(x[idx], mu)
        idx = x > (0.5 * length_x)
        external[idx, nn, 0] = _scatter(x[idx], mu)

    return external


def manufactured_td_01(x, angle_x, edges_t):
    """Manufactured time-dependent source (case TD_01).

    Parameters
    ----------
    x : array_like
        Spatial coordinates (n_x,).
    angle_x : array_like
        Angular ordinates (n_angles,).
    edges_t : array_like
        Time edge values (n_time_edges,).

    Returns
    -------
    numpy.ndarray
        External source array with shape (n_time_edges, n_x, n_angles, 1).
    """
    external = np.zeros((edges_t.shape[0], x.shape[0], angle_x.shape[0], 1))
    for cc, tt in enumerate(edges_t):
        for nn, mu in enumerate(angle_x):
            external[cc, :, nn, 0] = (
                2
                + x * (-0.2 - x * (-0.1 + mu) + 2 * mu) * np.cos(0.1 * tt - x)
                + ((-2 + x) * x + 2 * (x - 1) * mu) * np.sin(0.1 * tt - x)
            )
    return external


def manufactured_td_02(x, angle_x, edges_t):
    """Manufactured time-dependent source (case TD_02).

    Parameters
    ----------
    x : array_like
        Spatial coordinates (n_x,).
    angle_x : array_like
        Angular ordinates (n_angles,).
    edges_t : array_like
        Time edge values (n_time_edges,).

    Returns
    -------
    numpy.ndarray
        External source array with shape (n_time_edges, n_x, n_angles, 1).
    """
    external = np.zeros((edges_t.shape[0], x.shape[0], angle_x.shape[0], 1))
    for cc, tt in enumerate(edges_t):
        for nn, mu in enumerate(angle_x):
            external[cc, :, nn, 0] = (
                1
                + (mu - 0.5) * np.cos(0.5 * tt - x)
                + np.cos(mu)
                - np.sin(0.5 * tt - x)
            )
    return external


def reeds(edges_x, bc):
    """Reeds problem spatial-dependent source.

    Constructs the piecewise spatial source used by the Reeds test problem.

    Parameters
    ----------
    edges_x : array_like
        Spatial edge coordinates.
    bc : list-like
        Boundary condition indicator; used to select half-domain source
        patterns when bc indicates only one side is active.

    Returns
    -------
    numpy.ndarray
        Spatial source array with shape (n_cells, 1, 1).
    """
    # Spatial dependent source for Reeds Problem
    external = np.zeros((edges_x.shape[0] - 1, 1, 1))
    source_values = np.array([0.0, 1.0, 0.0, 50.0, 50.0, 0.0, 1.0, 0.0])
    lhs = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0])
    rhs = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0])

    if np.sum(bc) > 0.0:
        idx = slice(0, 4) if bc == [0, 1] else slice(4, 8)
        corrector = 0.0 if bc == [0, 1] else 8.0
        source_values = source_values[idx].copy()
        lhs = lhs[idx].copy() - corrector
        rhs = rhs[idx].copy() - corrector

    def loc(x):
        return int(np.argwhere(edges_x == x))

    bounds = [slice(loc(ii), loc(jj)) for ii, jj in zip(lhs, rhs)]

    for ii in range(len(bounds)):
        external[bounds[ii], 0, 0] = source_values[ii]

    return external


def ambe(x, loc_x, edges_g):
    """Load an AmBe external neutron source and map it into groups.

    Parameters
    ----------
    x : array_like
        Spatial coordinates (n_x,).
    loc_x : int
        Index in ``x`` where the AmBe source is located.
    edges_g : array_like
        Energy group edges. If values appear large (>20) they are assumed to
        be in eV and will be converted to MeV.

    Returns
    -------
    numpy.ndarray
        External source array with shape (n_x, 1, n_groups).
    """
    # AmBe source in middle of material
    external = np.zeros((x.shape[0], 1, edges_g.shape[0] - 1))

    data = np.load(DATA_PATH + "external/AmBe_source_050G.npz")
    # Convert to MeV
    if np.max(edges_g) > 20.0:
        edges_g *= 1e-6

    centers_g = 0.5 * (edges_g[1:] + edges_g[:-1])

    def loc_g(x1, x2):
        return np.argwhere((centers_g > x1) & (centers_g <= x2)).flatten()

    for ii in range(len(data["magnitude"])):
        idx = loc_g(data["edges"][ii], data["edges"][ii + 1])
        external[loc_x, 0, idx] = data["magnitude"][ii]

    return external


def time_dependence_constant(boundary_x):
    """Wrap an external source to add a time axis (constant in time).

    Parameters
    ----------
    boundary_x : numpy.ndarray
        External source array without a time axis.

    Returns
    -------
    numpy.ndarray
        Time-dependent external source with a leading time axis (shape: 1, ...).
    """
    return boundary_x[None, ...]
