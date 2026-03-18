"""Spatial sweep routines for slab and spherical geometries.

This module implements discrete-ordinates spatial sweeps used by the
solvers. It provides high-level driver functions that perform slab or
sphere sweeps for steady-state and time-dependent problems, plus
specialized variants for known external sources or scatter-only
operators used by DJINN integrations.

The performance-critical kernels are decorated with ``numba.jit`` and
operate on NumPy arrays. Public API (typical callers):

- ``discrete_ordinates``: generic dispatcher to slab/sphere handlers
- ``slab_ordinates`` / ``sphere_ordinates``: full sweep drivers
- ``slab_known_source_sn`` / ``sphere_known_source_sn``: known-source
- ``slab_scatter_source_sn`` / ``sphere_scatter_source_sn``: DJINN scatter

Docstrings on the individual functions describe parameter shapes and
return values expected by the rest of the package.
"""

import numba
import numpy as np

from discrete1 import tools

count_nn = 100
change_nn = 1e-12

########################################################################
# Function Calls
########################################################################


def discrete_ordinates(
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
    P=None,
    P_weights=None,
):
    """Dispatch to the appropriate spatial sweep for the geometry.

    This function chooses the slab or sphere sweeping routine based on
    the ``geometry`` argument and returns the updated scalar flux array
    evaluated at cell centers.

    Parameters
    ----------
    flux_old : numpy.ndarray, shape (n_cells,)
        Previous iterate of the scalar flux.
    xs_total : numpy.ndarray, shape (n_materials,)
        Total macroscopic cross sections per material for this group.
    xs_scatter : numpy.ndarray
        Within-group scatter cross sections mapped to cells.
        - Isotropic: shape (n_cells,) — scalar xs_s0 per cell.
        - Anisotropic: shape (n_cells, L+1) — Legendre moments per cell.
    off_scatter : numpy.ndarray, shape (n_cells,)
        Off-diagonal scatter source.
    external : numpy.ndarray, shape (n_cells, n_angles or 1)
        External source for this group.
    boundary : numpy.ndarray, shape (2, n_angles or 1)
        Boundary conditions for this group.
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
    numpy.ndarray
        Updated scalar flux at cell centers (shape: n_cells,).
    """
    edges = 0
    isotropic = xs_scatter.ndim == 1  # (cells_x, L+1) vs (cells_x,)

    # Isotropic slab geometry
    if geometry == 1 and isotropic:

        return slab_isotropic_sn(
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
            edges,
        )
    # Anisotropic slab geometry
    elif geometry == 1 and not isotropic:
        xs_scatter_w = xs_scatter * (2 * np.arange(xs_scatter.shape[1]) + 1)[None, :]
        return slab_anisotropic_sn(
            flux_old,
            xs_total,
            xs_scatter,
            xs_scatter_w,
            off_scatter,
            external,
            boundary,
            medium_map,
            delta_x,
            angle_x,
            angle_w,
            P,
            P_weights,
            bc_x,
            edges,
        )

    # Isotropic sphere geometry
    elif geometry == 2 and isotropic:
        return sphere_isotropic_sn(
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
            edges,
        )
    # Anisotropic sphere geometry
    elif geometry == 2 and not isotropic:
        xs_scatter_w = xs_scatter * (2 * np.arange(xs_scatter.shape[1]) + 1)[None, :]
        return sphere_anisotropic_sn(
            flux_old,
            xs_total,
            xs_scatter,
            xs_scatter_w,
            off_scatter,
            external,
            boundary,
            medium_map,
            delta_x,
            angle_x,
            angle_w,
            bc_x,
            edges,
            P,
            P_weights,
        )


def known_source_sn(
    flux,
    xs_total,
    zero,
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
    """Sweep with a fully prescribed source (no scatter iteration)."""
    # Slab geometry
    if geometry == 1:
        return slab_known_source_sn(
            flux,
            xs_total,
            zero,
            source,
            boundary,
            medium_map,
            delta_x,
            angle_x,
            angle_w,
            bc_x,
            edges,
        )

    # Sphere geometry
    elif geometry == 2:
        return sphere_known_source_sn(
            flux,
            xs_total,
            zero,
            source,
            boundary,
            medium_map,
            delta_x,
            angle_x,
            angle_w,
            bc_x,
            edges,
        )


def scatter_source_sn(
    xs_total,
    scatter_source,
    external,
    boundary,
    medium_map,
    delta_x,
    angle_x,
    angle_w,
    bc_x,
    geometry,
):
    """Sweep with a prescribed scatter source (DJINN variant)."""
    # Slab geometry
    if geometry == 1:
        return slab_scatter_source_sn(
            xs_total,
            scatter_source,
            external,
            boundary,
            medium_map,
            delta_x,
            angle_x,
            angle_w,
            bc_x,
        )

    # Sphere geometry
    elif geometry == 2:
        return sphere_scatter_source_sn(
            xs_total,
            scatter_source,
            external,
            boundary,
            medium_map,
            delta_x,
            angle_x,
            angle_w,
            bc_x,
        )


########################################################################
# Slab Sweeps - Isotropic
########################################################################


def slab_isotropic_sn(
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
    edges,
):
    """Isotropic slab-geometry discrete-ordinates spatial sweep.

    The solver iterates over angular ordinates and performs forward and
    backward diamond-difference passes until the scalar flux converges
    (or a maximum iteration count is reached). This routine returns the
    converged cell-centered scalar flux.

    Parameters
    ----------
    flux_old : numpy.ndarray
        Initial flux guess (n_cells,).
    xs_total : numpy.ndarray, shape (n_materials,)
        Total cross sections per material.
    xs_scatter : numpy.ndarray, shape (n_cells,)
        Within-group isotropic scatter cross section mapped to cells.
    off_scatter : numpy.ndarray, shape (n_cells,)
        Off-diagonal scatter source.
    external : numpy.ndarray, shape (n_cells, n_angles or 1)
        External source.
    boundary : numpy.ndarray, shape (2, n_angles or 1)
        Boundary conditions.
    medium_map : numpy.ndarray, shape (n_cells,)
        Material index per cell.
    delta_x : numpy.ndarray, shape (n_cells,)
        Cell widths.
    angle_x, angle_w : numpy.ndarray, shape (angles,)
        Angular quadrature points and weights.
    bc_x : sequence
        Boundary condition flags.
    edges : int
        If 1, accumulate edge fluxes; else cell-center fluxes.

    Returns
    -------
    numpy.ndarray, shape (n_cells,)
        Converged scalar flux at cell centers.
    """

    cells_x = flux_old.shape[0]
    angles = angle_x.shape[0]

    flux = np.zeros((cells_x,))
    reflector = np.zeros((angles,))
    edge1 = 0.0

    converged = False
    count = 1
    change = 0.0

    while not (converged):
        flux *= 0.0
        reflector *= 0.0

        for nn in range(angles):
            qq = 0 if external.shape[1] == 1 else nn
            bc = 0 if boundary.shape[1] == 1 else nn

            if angle_x[nn] > 0.0:
                edge1 = reflector[nn] + boundary[0, bc]
                edge1 = slab_forward_iso(
                    flux,
                    flux_old,
                    xs_total,
                    xs_scatter,
                    off_scatter,
                    external[:, qq],
                    edge1,
                    medium_map,
                    delta_x,
                    angle_x[nn],
                    angle_w[nn],
                    edges,
                )

            elif angle_x[nn] < 0.0:
                edge1 = reflector[nn] + boundary[1, bc]
                edge1 = slab_backward_iso(
                    flux,
                    flux_old,
                    xs_total,
                    xs_scatter,
                    off_scatter,
                    external[:, qq],
                    edge1,
                    medium_map,
                    delta_x,
                    angle_x[nn],
                    angle_w[nn],
                    edges,
                )
            else:
                raise Exception("Discontinuity at 0")

            tools.reflector_corrector(reflector, angle_x, edge1, nn, bc_x)

        # Check for convergence
        try:
            change = np.linalg.norm((flux - flux_old) / flux / cells_x)
        except RuntimeWarning:
            change = 0.0
        converged = (change < change_nn) or (count >= count_nn)
        count += 1

        flux_old = flux.copy()

    return flux


@numba.jit(
    "f8(f8[:], f8[:], f8[:], f8[:], f8[:], f8[:], f8, i4[:], \
            f8[:], f8, f8, i4)",
    nopython=True,
    cache=True,
)
def slab_forward_iso(
    flux,
    flux_old,
    xs_total,
    xs_scatter,
    off_scatter,
    external,
    edge1,
    medium_map,
    delta_x,
    angle_x,
    angle_w,
    edges,
):
    """Forward (left to right) isotropic slab sweep for one ordinate.

    Returns
    -------
    float
        Outgoing edge flux at i = I.
    """
    cells_x = numba.int32(medium_map.shape[0])
    mat = numba.int32
    ii = numba.int32
    edge2 = numba.float64(0.0)

    if edges == 1:
        flux[0] += angle_w * edge1

    for ii in range(cells_x):
        mat = medium_map[ii]

        edge2 = (
            xs_scatter[mat] * flux_old[ii]
            + external[ii]
            + off_scatter[ii]
            + edge1 * (angle_x / delta_x[ii] - 0.5 * xs_total[mat])
        ) / (angle_x / delta_x[ii] + 0.5 * xs_total[mat])

        # Update flux with cell edges or centers
        if edges == 1:
            flux[ii + 1] += angle_w * edge2
        else:
            flux[ii] += 0.5 * angle_w * (edge1 + edge2)
        edge1 = edge2

    return edge1


@numba.jit(
    "f8(f8[:], f8[:], f8[:], f8[:], f8[:], f8[:], f8, i4[:], \
            f8[:], f8, f8, i4)",
    nopython=True,
    cache=True,
)
def slab_backward_iso(
    flux,
    flux_old,
    xs_total,
    xs_scatter,
    off_scatter,
    external,
    edge1,
    medium_map,
    delta_x,
    angle_x,
    angle_w,
    edges,
):
    """Backward (right to left) isotropic slab sweep for one ordinate.

    Returns
    -------
    float
        Outgoing edge flux at i = 0.
    """
    cells_x = numba.int32(medium_map.shape[0])
    mat = numba.int32
    ii = numba.int32
    edge2 = numba.float64(0.0)

    # Determine flux edge
    if edges == 1:
        flux[cells_x] += angle_w * edge1

    for ii in range(cells_x - 1, -1, -1):
        mat = medium_map[ii]

        edge2 = (
            xs_scatter[mat] * flux_old[ii]
            + external[ii]
            + off_scatter[ii]
            + edge1 * (-angle_x / delta_x[ii] - 0.5 * xs_total[mat])
        ) / (-angle_x / delta_x[ii] + 0.5 * xs_total[mat])

        # Update flux with cell edges or centers
        if edges == 1:
            flux[ii] += angle_w * edge2
        else:
            flux[ii] += 0.5 * angle_w * (edge1 + edge2)
        edge1 = edge2

    return edge1


########################################################################
# Slab Sweeps — Anisotropic
########################################################################


def slab_anisotropic_sn(
    flux_old,
    xs_total,
    xs_self_scatter,
    xs_scatter_w,
    off_scatter,
    external,
    boundary,
    medium_map,
    delta_x,
    angle_x,
    angle_w,
    bc_x,
    edges,
    P,
    P_weights,
):
    """Anisotropic slab-geometry discrete-ordinates sweep.

    Parameters
    ----------
    flux_old : numpy.ndarray, shape (n_cells,)
        Initial scalar flux guess.
    xs_total : numpy.ndarray, shape (n_materials,)
        Total cross sections per material.
    xs_self_scatter : numpy.ndarray, shape (n_cells, L+1)
        Within-group Legendre scatter moments mapped to cells.
    xs_scatter_w : numpy.ndarray, shape (n_cells, L+1)
        Precomputed (2l+1) * xs_self_scatter.
    off_scatter : numpy.ndarray, shape (n_cells,)
        Off-diagonal scatter source.
    external : numpy.ndarray, shape (n_cells, n_angles or 1)
        External source.
    boundary : numpy.ndarray, shape (2, n_angles or 1)
        Boundary conditions.
    medium_map : numpy.ndarray, shape (n_cells,)
        Material index per cell.
    delta_x : numpy.ndarray, shape (n_cells,)
        Cell widths.
    angle_x, angle_w : numpy.ndarray, shape (angles,)
        Angular quadrature points and weights.
    bc_x : sequence
        Boundary condition flags.
    edges : int
        If 1, accumulate edge fluxes; else cell-centre fluxes.
    P : numpy.ndarray, shape (L+1, angles)
        Precomputed P[l, n] = P_l(mu_n).
    P_weights : numpy.ndarray, shape (L+1, angles)
        Precomputed w_n * P_l(mu_n).

    Returns
    -------
    numpy.ndarray, shape (n_cells,)
        Converged scalar flux at cell centres.
    """
    cells_x = flux_old.shape[0]
    angles = angle_x.shape[0]
    n_moments = xs_self_scatter.shape[1]

    flux = np.zeros((cells_x,))
    reflector = np.zeros((angles,))
    edge1 = 0.0

    # Seed L=0 flux moment from initial guess; higher moments start zero.
    flux_moments = np.zeros((cells_x, n_moments))
    flux_moments[:, 0] = flux_old.copy()
    anisotropic = np.zeros((cells_x, angles))

    converged = False
    count = 1
    change = 0.0

    while not converged:
        flux *= 0.0
        reflector *= 0.0

        # Build anisotropic scatter source from moments of previous sweep.
        anisotropic = np.einsum("xl,xl,ln->xn", xs_scatter_w, flux_moments, P)

        # Reset moment accumulator for this sweep.
        flux_moments *= 0.0

        for nn in range(angles):
            qq = 0 if external.shape[1] == 1 else nn
            bc = 0 if boundary.shape[1] == 1 else nn

            if angle_x[nn] > 0.0:
                edge1 = reflector[nn] + boundary[0, bc]
                edge1, psi_nn = slab_forward_aniso(
                    flux,
                    xs_total,
                    anisotropic[:, nn],
                    off_scatter,
                    external[:, qq],
                    edge1,
                    medium_map,
                    delta_x,
                    angle_x[nn],
                    angle_w[nn],
                    edges,
                )
            elif angle_x[nn] < 0.0:
                edge1 = reflector[nn] + boundary[1, bc]
                edge1, psi_nn = slab_backward_aniso(
                    flux,
                    xs_total,
                    anisotropic[:, nn],
                    off_scatter,
                    external[:, qq],
                    edge1,
                    medium_map,
                    delta_x,
                    angle_x[nn],
                    angle_w[nn],
                    edges,
                )
            else:
                raise Exception("Discontinuity at 0")

            # Accumulate flux moments: one rank-1 outer-product per ordinate.
            flux_moments += np.outer(psi_nn, P_weights[:, nn])
            tools.reflector_corrector(reflector, angle_x, edge1, nn, bc_x)

        try:
            change = np.linalg.norm((flux - flux_old) / flux / cells_x)
        except RuntimeWarning:
            change = 0.0
        converged = (change < change_nn) or (count >= count_nn)
        count += 1
        flux_old = flux.copy()

    return flux


@numba.jit(
    "Tuple((f8, f8[:]))(f8[:], f8[:], f8[:], f8[:], f8[:], f8, i4[:], "
    "f8[:], f8, f8, i4)",
    nopython=True,
    cache=True,
)
def slab_forward_aniso(
    flux,
    xs_total,
    anisotropic,
    off_scatter,
    external,
    edge1,
    medium_map,
    delta_x,
    angle_x,
    angle_w,
    edges,
):
    """Forward (left to right) anisotropic slab sweep for one ordinate.

    Returns
    -------
    edge1 : float
        Outgoing edge flux at i = I.
    psi : numpy.ndarray, shape (n_cells,)
        Cell-centre angular flux for moment accumulation.
    """
    cells_x = numba.int32(medium_map.shape[0])
    mat = numba.int32
    ii = numba.int32
    edge2 = numba.float64(0.0)
    psi = np.empty((cells_x,))

    if edges == 1:
        flux[0] += angle_w * edge1

    for ii in range(cells_x):
        mat = medium_map[ii]
        edge2 = (
            anisotropic[ii]
            + external[ii]
            + off_scatter[ii]
            + edge1 * (angle_x / delta_x[ii] - 0.5 * xs_total[mat])
        ) / (angle_x / delta_x[ii] + 0.5 * xs_total[mat])
        psi[ii] = 0.5 * (edge1 + edge2)

        # Update flux with cell edges or centers
        if edges == 1:
            flux[ii + 1] += angle_w * edge2
        else:
            flux[ii] += 0.5 * angle_w * (edge1 + edge2)
        edge1 = edge2

    return (edge1, psi)


@numba.jit(
    "Tuple((f8, f8[:]))(f8[:], f8[:], f8[:], f8[:], f8[:], f8, i4[:], "
    "f8[:], f8, f8, i4)",
    nopython=True,
    cache=True,
)
def slab_backward_aniso(
    flux,
    xs_total,
    anisotropic,
    off_scatter,
    external,
    edge1,
    medium_map,
    delta_x,
    angle_x,
    angle_w,
    edges,
):
    """Backward (right to left) anisotropic slab sweep for one ordinate.

    Returns
    -------
    edge1 : float
        Outgoing edge flux at i = 0.
    psi : numpy.ndarray, shape (n_cells,)
        Cell-centre angular flux for moment accumulation.
    """
    cells_x = numba.int32(medium_map.shape[0])
    mat = numba.int32
    ii = numba.int32
    edge2 = numba.float64(0.0)
    psi = np.empty((cells_x,))

    if edges == 1:
        flux[cells_x] += angle_w * edge1

    for ii in range(cells_x - 1, -1, -1):
        mat = medium_map[ii]
        edge2 = (
            anisotropic[ii]
            + external[ii]
            + off_scatter[ii]
            + edge1 * (-angle_x / delta_x[ii] - 0.5 * xs_total[mat])
        ) / (-angle_x / delta_x[ii] + 0.5 * xs_total[mat])
        psi[ii] = 0.5 * (edge1 + edge2)

        # Update flux with cell edges or centers
        if edges == 1:
            flux[ii] += angle_w * edge2
        else:
            flux[ii] += 0.5 * angle_w * (edge1 + edge2)
        edge1 = edge2

    return (edge1, psi)


########################################################################
# Sphere Sweeps
########################################################################


def sphere_isotropic_sn(
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
    edges,
):
    """Isotropic spherical-geometry discrete-ordinates spatial sweep.

    The spherical sweep uses half-angle bookkeeping and angular
    differencing coefficients appropriate for radial geometry. The
    implementation iterates over ordinates and updates a converged
    scalar flux similar to the slab driver.

    Parameters
    ----------
    flux_old, xs_total, xs_scatter, off_scatter, external, boundary, \
    medium_map, delta_x, angle_x, angle_w, bc_x, edges :
        See :func:`slab_isotropic_sn` for parameter shapes and meanings.

    Returns
    -------
    numpy.ndarray
        Converged scalar flux at cell centers (n_cells,).
    """

    cells_x = flux_old.shape[0]
    angles = angle_x.shape[0]

    flux = np.zeros((cells_x,))
    half_angle = np.zeros((cells_x,))

    converged = False
    count = 1
    change = 0.0

    while not (converged):

        angle_minus = -1.0
        alpha_minus = 0.0

        flux *= 0.0
        # Calculate the initial half angle
        _half_angle_iso(
            flux_old,
            half_angle,
            xs_total,
            xs_scatter,
            off_scatter,
            external[:, 0],
            medium_map,
            delta_x,
            boundary[1, 0],
        )

        for nn in range(angles):
            qq = 0 if external.shape[1] == 1 else nn
            bc = 0 if boundary.shape[1] == 1 else nn

            # Calculate the half angle coefficient
            angle_plus = angle_minus + 2 * angle_w[nn]
            # Calculate the weighted diamond
            tau = (angle_x[nn] - angle_minus) / (angle_plus - angle_minus)
            # Calculate the angular differencing coefficient
            alpha_plus = angle_coef_corrector(
                alpha_minus, angle_x[nn], angle_w[nn], nn, angles
            )

            # Iterate from 0 -> I
            if angle_x[nn] > 0.0:
                sphere_forward_iso(
                    flux,
                    flux_old,
                    half_angle,
                    xs_total,
                    xs_scatter,
                    off_scatter,
                    external[:, qq],
                    medium_map,
                    delta_x,
                    angle_x[nn],
                    angle_w[nn],
                    angle_w[nn],
                    tau,
                    alpha_plus,
                    alpha_minus,
                    edges,
                )

            # Iterate from I -> 0
            elif angle_x[nn] < 0.0:
                sphere_backward_iso(
                    flux,
                    flux_old,
                    half_angle,
                    xs_total,
                    xs_scatter,
                    off_scatter,
                    external[:, qq],
                    boundary[1, bc],
                    medium_map,
                    delta_x,
                    angle_x[nn],
                    angle_w[nn],
                    angle_w[nn],
                    tau,
                    alpha_plus,
                    alpha_minus,
                    edges,
                )
            else:
                raise Exception("Discontinuity at 0")

            alpha_minus = alpha_plus
            angle_minus = angle_plus

        change = np.linalg.norm((flux - flux_old) / flux / cells_x)
        converged = (change < change_nn) or (count >= count_nn)
        count += 1
        flux_old = flux.copy()

    return flux


@numba.jit(
    "void(f8[:], f8[:], f8[:], f8[:], f8[:], f8[:], i4[:], f8[:], \
            f8)",
    nopython=True,
    cache=True,
)
def _half_angle_iso(
    flux,
    half_angle,
    xs_total,
    xs_scatter,
    off_scatter,
    external,
    medium_map,
    delta_x,
    angle_plus,
):
    """Seed the half-angle array for the isotropic sphere sweep."""
    cells_x = numba.int32(flux.shape[0])
    mat = numba.int32
    ii = numba.int32
    half_angle *= 0.0

    for ii in range(cells_x - 1, -1, -1):
        mat = medium_map[ii]
        half_angle[ii] = (
            2 * angle_plus
            + delta_x[ii]
            * (external[ii] + off_scatter[ii] + xs_scatter[mat] * flux[ii])
        ) / (2 + xs_total[mat] * delta_x[ii])
        angle_plus = 2 * half_angle[ii] - angle_plus


@numba.jit("f8(f8, f8, f8, i4, i4)", nopython=True, cache=True)
def angle_coef_corrector(alpha_minus, angle_x, angle_w, nn, angles):
    """Calculate angular differencing coefficient for spherical geometry.

    Computes the angular differencing coefficient used in spherical
    geometry sweeps for proper treatment of angular derivatives.

    Parameters
    ----------
    alpha_minus : float
        Previous angular coefficient.
    angle_x : float
        Current angular ordinate.
    angle_w : float
        Quadrature weight for current ordinate.
    nn : int
        Current angle index.
    angles : int
        Total number of angles.

    Returns
    -------
    float
        Updated angular coefficient. Returns 0 for last angle.
    """
    if nn != angles - 1:
        return alpha_minus - angle_x * angle_w
    return 0.0


@numba.jit(
    "void(f8[:], f8[:], f8[:], f8[:], f8[:], f8[:], f8[:], i4[:], \
            f8[:], f8, f8, f8, f8, f8, f8, i4)",
    nopython=True,
    cache=True,
)
def sphere_forward_iso(
    flux,
    flux_old,
    half_angle,
    xs_total,
    xs_scatter,
    off_scatter,
    external,
    medium_map,
    delta_x,
    angle_x,
    angle_w,
    weight,
    tau,
    alpha_plus,
    alpha_minus,
    edges,
):
    """Forward sweep kernel for spherical geometry for a single ordinate.

    Updates ``flux`` in-place by marching from center toward the outer
    radius using the provided half-angle and differencing coefficients.
    This kernel is performance-critical and compiled with numba.
    """
    cells_x = numba.int32(flux.shape[0])
    mat = numba.int32
    ii = numba.int32
    edge1 = numba.float64(half_angle[0])

    if edges == 1:
        flux[0] += weight * edge1

    # Initialize surface area on cell edges, cell volume, flux center
    area1 = numba.float64
    area2 = numba.float64
    center = numba.float64
    volume = numba.float64

    for ii in range(cells_x):
        mat = medium_map[ii]

        # Calculate surface area and volume of cell
        area1 = 4 * np.pi * (ii * delta_x[ii]) ** 2
        area2 = 4 * np.pi * ((ii + 1) * delta_x[ii]) ** 2
        volume = (
            4 / 3.0 * np.pi * (((ii + 1) * delta_x[ii]) ** 3 - (ii * delta_x[ii]) ** 3)
        )

        center = (
            angle_x * (area2 + area1) * edge1
            + (1 / angle_w)
            * (area2 - area1)
            * (alpha_plus + alpha_minus)
            * (half_angle[ii])
            + volume * (external[ii] + off_scatter[ii] + flux_old[ii] * xs_scatter[mat])
        ) / (
            2 * angle_x * area2
            + 2 / angle_w * (area2 - area1) * alpha_plus
            + xs_total[mat] * volume
        )

        # Update flux with cell edges or cell centers
        if edges == 1:
            flux[ii + 1] += weight * (2 * center - edge1)
        else:
            flux[ii] += weight * center
        edge1 = 2 * center - edge1
        if ii != 0:
            half_angle[ii] = 1 / tau * (center - (1 - tau) * half_angle[ii])


@numba.jit(
    "void(f8[:], f8[:], f8[:], f8[:], f8[:], f8[:], f8[:], f8, i4[:], \
        f8[:], f8, f8, f8, f8, f8, f8, i4)",
    nopython=True,
    cache=True,
)
def sphere_backward_iso(
    flux,
    flux_old,
    half_angle,
    xs_total,
    xs_scatter,
    off_scatter,
    external,
    boundary,
    medium_map,
    delta_x,
    angle_x,
    angle_w,
    weight,
    tau,
    alpha_plus,
    alpha_minus,
    edges,
):
    """Backward sweep kernel for spherical geometry for a single ordinate.

    Marches from the outer radius toward the center, updating ``flux``
    in-place. The kernel uses geometry-specific surface area and
    volume factors to compute center contributions.
    """
    cells_x = numba.int32(flux.shape[0])
    mat = numba.int32
    ii = numba.int32
    edge1 = numba.float64(boundary)

    if edges == 1:
        flux[cells_x] += weight * edge1

    area1 = numba.float64
    area2 = numba.float64
    center = numba.float64
    volume = numba.float64

    for ii in range(cells_x - 1, -1, -1):
        mat = medium_map[ii]
        # Calculate surface area and volume of cell
        area1 = 4 * np.pi * (ii * delta_x[ii]) ** 2
        area2 = 4 * np.pi * ((ii + 1) * delta_x[ii]) ** 2
        volume = (
            4 / 3.0 * np.pi * (((ii + 1) * delta_x[ii]) ** 3 - (ii * delta_x[ii]) ** 3)
        )

        center = (
            -angle_x * (area2 + area1) * edge1
            + (1 / angle_w)
            * (area2 - area1)
            * (alpha_plus + alpha_minus)
            * (half_angle[ii])
            + volume * (external[ii] + off_scatter[ii] + flux_old[ii] * xs_scatter[mat])
        ) / (
            2 * -angle_x * area1
            + 2 / angle_w * (area2 - area1) * alpha_plus
            + xs_total[mat] * volume
        )

        # Update flux with cell edges or cell centers
        if edges == 1:
            flux[ii + 1] += weight * (2 * center - edge1)
        else:
            flux[ii] += weight * center
        edge1 = 2 * center - edge1
        if ii != 0:
            half_angle[ii] = 1 / tau * (center - (1 - tau) * half_angle[ii])


########################################################################
# Sphere Sweeps — Anisotropic
########################################################################


def sphere_anisotropic_sn(
    flux_old,
    xs_total,
    xs_self_scatter,
    xs_scatter_w,
    off_scatter,
    external,
    boundary,
    medium_map,
    delta_x,
    angle_x,
    angle_w,
    bc_x,
    edges,
    P,
    P_weights,
):
    """Anisotropic sphere-geometry discrete-ordinates sweep.

    Parameters match :func:`slab_anisotropic_sn` with sphere-specific
    half-angle bookkeeping. See that function for parameter descriptions.

    Returns
    -------
    numpy.ndarray, shape (n_cells,)
        Converged scalar flux at cell centres.
    """
    cells_x = flux_old.shape[0]
    angles = angle_x.shape[0]
    n_moments = xs_self_scatter.shape[1]

    flux = np.zeros((cells_x,))
    half_angle = np.zeros((cells_x,))

    flux_moments = np.zeros((cells_x, n_moments))
    flux_moments[:, 0] = flux_old
    anisotropic = np.zeros((cells_x, angles))

    converged = False
    count = 1
    change = 0.0

    while not converged:
        flux *= 0.0
        angle_minus = -1.0
        alpha_minus = 0.0

        anisotropic = np.einsum("xl,xl,ln->xn", xs_scatter_w, flux_moments, P)
        flux_moments *= 0.0

        _half_angle_aniso(
            flux_old,
            half_angle,
            xs_total,
            xs_self_scatter[:, 0],
            off_scatter,
            external[:, 0],
            medium_map,
            delta_x,
            boundary[1, 0],
        )

        for nn in range(angles):
            qq = 0 if external.shape[1] == 1 else nn
            bc = 0 if boundary.shape[1] == 1 else nn

            angle_plus = angle_minus + 2 * angle_w[nn]
            tau = (angle_x[nn] - angle_minus) / (angle_plus - angle_minus)
            alpha_plus = angle_coef_corrector(
                alpha_minus, angle_x[nn], angle_w[nn], nn, angles
            )

            if angle_x[nn] > 0.0:
                psi_nn = sphere_forward_aniso(
                    flux,
                    half_angle,
                    xs_total,
                    anisotropic[:, nn],
                    off_scatter,
                    external[:, qq],
                    medium_map,
                    delta_x,
                    angle_x[nn],
                    angle_w[nn],
                    angle_w[nn],
                    tau,
                    alpha_plus,
                    alpha_minus,
                    edges,
                )
            elif angle_x[nn] < 0.0:
                psi_nn = sphere_backward_aniso(
                    flux,
                    half_angle,
                    xs_total,
                    anisotropic[:, nn],
                    off_scatter,
                    external[:, qq],
                    boundary[1, bc],
                    medium_map,
                    delta_x,
                    angle_x[nn],
                    angle_w[nn],
                    angle_w[nn],
                    tau,
                    alpha_plus,
                    alpha_minus,
                    edges,
                )
            else:
                raise Exception("Discontinuity at 0")

            flux_moments += np.outer(psi_nn, P_weights[:, nn])
            alpha_minus = alpha_plus
            angle_minus = angle_plus

        change = np.linalg.norm((flux - flux_old) / flux / cells_x)
        converged = (change < change_nn) or (count >= count_nn)
        count += 1
        flux_old = flux.copy()

    return flux


@numba.jit(
    "void(f8[:], f8[:], f8[:], f8[:], f8[:], f8[:], i4[:], f8[:], f8)",
    nopython=True,
    cache=True,
)
def _half_angle_aniso(
    flux,
    half_angle,
    xs_total,
    xs_scatter_l0,
    off_scatter,
    external,
    medium_map,
    delta_x,
    angle_plus,
):
    """Seed the half-angle array for the anisotropic sphere sweep.

    Uses the L=0 scatter moment only, consistent with the treatment of
    the initial half-angle before any ordinate-specific information is
    available.
    """
    cells_x = numba.int32(flux.shape[0])
    mat = numba.int32
    ii = numba.int32
    half_angle *= 0.0
    for ii in range(cells_x - 1, -1, -1):
        mat = medium_map[ii]
        half_angle[ii] = (
            2 * angle_plus
            + delta_x[ii]
            * (external[ii] + off_scatter[ii] + xs_scatter_l0[ii] * flux[ii])
        ) / (2 + xs_total[mat] * delta_x[ii])
        angle_plus = 2 * half_angle[ii] - angle_plus


@numba.jit(
    "f8[:](f8[:], f8[:], f8[:], f8[:], f8[:], f8[:], i4[:], "
    "f8[:], f8, f8, f8, f8, f8, f8, i4)",
    nopython=True,
    cache=True,
)
def sphere_forward_aniso(
    flux,
    half_angle,
    xs_total,
    anisotropic,
    off_scatter,
    external,
    medium_map,
    delta_x,
    angle_x,
    angle_w,
    weight,
    tau,
    alpha_plus,
    alpha_minus,
    edges,
):
    """Forward (center to edge) anisotropic sphere sweep for one ordinate.

    Returns
    -------
    psi : numpy.ndarray, shape (n_cells,)
        Cell-centre angular flux for moment accumulation.
    """
    cells_x = numba.int32(flux.shape[0])
    mat = numba.int32
    ii = numba.int32
    edge1 = numba.float64(half_angle[0])
    psi = np.empty((cells_x,))

    if edges == 1:
        flux[0] += weight * edge1

    area1 = numba.float64
    area2 = numba.float64
    center = numba.float64
    volume = numba.float64

    for ii in range(cells_x):
        mat = medium_map[ii]

        # Calculate surface areas and volume of cell
        area1 = 4 * np.pi * (ii * delta_x[ii]) ** 2
        area2 = 4 * np.pi * ((ii + 1) * delta_x[ii]) ** 2
        volume = (
            4 / 3.0 * np.pi * (((ii + 1) * delta_x[ii]) ** 3 - (ii * delta_x[ii]) ** 3)
        )

        center = (
            angle_x * (area2 + area1) * edge1
            + (1 / angle_w)
            * (area2 - area1)
            * (alpha_plus + alpha_minus)
            * half_angle[ii]
            + volume * (external[ii] + off_scatter[ii] + anisotropic[ii])
        ) / (
            2 * angle_x * area2
            + 2 / angle_w * (area2 - area1) * alpha_plus
            + xs_total[mat] * volume
        )
        psi[ii] = center
        if edges == 1:
            flux[ii + 1] += weight * (2 * center - edge1)
        else:
            flux[ii] += weight * center
        edge1 = 2 * center - edge1
        if ii != 0:
            half_angle[ii] = 1 / tau * (center - (1 - tau) * half_angle[ii])

    return psi


@numba.jit(
    "f8[:](f8[:], f8[:], f8[:], f8[:], f8[:], f8[:], f8, i4[:], "
    "f8[:], f8, f8, f8, f8, f8, f8, i4)",
    nopython=True,
    cache=True,
)
def sphere_backward_aniso(
    flux,
    half_angle,
    xs_total,
    anisotropic,
    off_scatter,
    external,
    boundary,
    medium_map,
    delta_x,
    angle_x,
    angle_w,
    weight,
    tau,
    alpha_plus,
    alpha_minus,
    edges,
):
    """Backward (edge to center) anisotropic sphere sweep for one ordinate.

    Returns
    -------
    psi : numpy.ndarray, shape (n_cells,)
        Cell-centre angular flux for moment accumulation.
    """
    cells_x = numba.int32(flux.shape[0])
    mat = numba.int32
    ii = numba.int32
    edge1 = numba.float64(boundary)
    psi = np.empty((cells_x,))

    if edges == 1:
        flux[cells_x] += weight * edge1

    area1 = numba.float64
    area2 = numba.float64
    center = numba.float64
    volume = numba.float64

    for ii in range(cells_x - 1, -1, -1):
        mat = medium_map[ii]

        # Calculate surface areas and volume of cell
        area1 = 4 * np.pi * (ii * delta_x[ii]) ** 2
        area2 = 4 * np.pi * ((ii + 1) * delta_x[ii]) ** 2
        volume = (
            4 / 3.0 * np.pi * (((ii + 1) * delta_x[ii]) ** 3 - (ii * delta_x[ii]) ** 3)
        )

        center = (
            -angle_x * (area2 + area1) * edge1
            + (1 / angle_w)
            * (area2 - area1)
            * (alpha_plus + alpha_minus)
            * half_angle[ii]
            + volume * (external[ii] + off_scatter[ii] + anisotropic[ii])
        ) / (
            2 * -angle_x * area1
            + 2 / angle_w * (area2 - area1) * alpha_plus
            + xs_total[mat] * volume
        )
        psi[ii] = center
        if edges == 1:
            flux[ii + 1] += weight * (2 * center - edge1)
        else:
            flux[ii] += weight * center
        edge1 = 2 * center - edge1
        if ii != 0:
            half_angle[ii] = 1 / tau * (center - (1 - tau) * half_angle[ii])

    return psi


########################################################################
# Known Sweeps Source with no scatter iteration
########################################################################


def slab_known_source_sn(
    flux,
    xs_total,
    zero,
    source,
    boundary,
    medium_map,
    delta_x,
    angle_x,
    angle_w,
    bc_x,
    edges,
):
    """Slab sweep driver for a known (prescribed) external source.

    This variant is used when the external/source term is provided and
    the scatter contribution is not iterated. The function supports
    scalar or angular flux shapes depending on the ``flux`` input.

    Parameters
    ----------
    flux : numpy.ndarray
        Output array for scalar or angular fluxes (shape: n_cells, xdim).
    xs_total : numpy.ndarray, shape (n_materials,)
        Total cross sections per material.
    zero : numpy.ndarray, shape (n_cells,)
        Zero placeholder (passed to sweep kernels for unused arguments).
    source : numpy.ndarray, shape (n_cells, n_angles or 1)
        Prescribed source for this group.
    boundary : numpy.ndarray, shape (2, n_angles or 1)
        Boundary conditions.
    medium_map, delta_x, angle_x, angle_w, bc_x, edges :
        See :func:`slab_isotropic_sn`.

    Returns
    -------
    numpy.ndarray
        The flux array (same object as input) updated in-place.
    """

    _, xdim = flux.shape
    angles = angle_x.shape[0]
    reflector = np.zeros((angles,))
    edge1 = 0.0

    for nn in range(angles):
        qq = 0 if source.shape[1] == 1 else nn
        bc = 0 if boundary.shape[1] == 1 else nn

        # Scalar flux calculation
        if (angle_x[nn] > 0.0) and (xdim == 1):
            edge1 = reflector[nn] + boundary[0, bc]
            edge1 = slab_forward_iso(
                flux[:, 0],
                zero,
                xs_total,
                zero,
                zero,
                source[:, qq],
                edge1,
                medium_map,
                delta_x,
                angle_x[nn],
                angle_w[nn],
                edges,
            )

        elif (angle_x[nn] < 0.0) and (xdim == 1):
            edge1 = reflector[nn] + boundary[1, bc]
            edge1 = slab_backward_iso(
                flux[:, 0],
                zero,
                xs_total,
                zero,
                zero,
                source[:, qq],
                edge1,
                medium_map,
                delta_x,
                angle_x[nn],
                angle_w[nn],
                edges,
            )

        # Angular flux calculation
        elif (angle_x[nn] > 0.0) and (xdim > 1):
            edge1 = reflector[nn] + boundary[0, bc]
            edge1 = slab_forward_iso(
                flux[:, nn],
                zero,
                xs_total,
                zero,
                zero,
                source[:, qq],
                edge1,
                medium_map,
                delta_x,
                angle_x[nn],
                1.0,
                edges,
            )

        elif (angle_x[nn] < 0.0) and (xdim > 1):
            edge1 = reflector[nn] + boundary[1, bc]
            edge1 = slab_backward_iso(
                flux[:, nn],
                zero,
                xs_total,
                zero,
                zero,
                source[:, qq],
                edge1,
                medium_map,
                delta_x,
                angle_x[nn],
                1.0,
                edges,
            )

        tools.reflector_corrector(reflector, angle_x, edge1, nn, bc_x)

    return flux


def sphere_known_source_sn(
    flux,
    xs_total,
    zero,
    source,
    boundary,
    medium_map,
    delta_x,
    angle_x,
    angle_w,
    bc_x,
    edges,
):
    """Sphere sweep driver for prescribed external source.

    This variant implements the spherical sweep for cases with a known
    external source without iterating on scatter. Supports both scalar
    and angular flux output shapes based on input array dimensions.

    Parameters
    ----------
    flux : numpy.ndarray
        Output array for scalar/angular fluxes (n_cells, xdim).
    xs_total : array_like
        Total cross sections indexed by material.
    zero : numpy.ndarray
        Zero-array placeholder for kernel compatibility.
    source : numpy.ndarray
        External source (n_cells, n_angles or 1).
    boundary : numpy.ndarray
        Boundary conditions (2, n_angles or 1).
    medium_map : array_like
        Material indices (n_cells,).
    delta_x : array_like
        Cell widths (n_cells,).
    angle_x : array_like
        Angular ordinates (n_angles,).
    angle_w : array_like
        Quadrature weights (n_angles,).
    bc_x : sequence
        Boundary condition flags.
    edges : numpy.ndarray
        Edge fluxes at cell boundaries.

    Returns
    -------
    numpy.ndarray
        Flux array (same as input) updated in-place.
    """

    cells_x, xdim = flux.shape
    angles = angle_x.shape[0]
    flux = np.zeros((cells_x,))
    half_angle = np.zeros((cells_x,))

    angle_minus = -1.0
    alpha_minus = 0.0

    _half_angle_iso(
        zero,
        half_angle,
        xs_total,
        zero,
        zero,
        source[:, 0],
        medium_map,
        delta_x,
        boundary[1, 0],
    )

    for nn in range(angles):
        qq = 0 if source.shape[1] == 1 else nn
        bc = 0 if boundary.shape[1] == 1 else nn

        angle_plus = angle_minus + 2 * angle_w[nn]
        tau = (angle_x[nn] - angle_minus) / (angle_plus - angle_minus)
        alpha_plus = angle_coef_corrector(
            alpha_minus, angle_x[nn], angle_w[nn], nn, angles
        )

        # Scalar flux calculation
        if (angle_x[nn] > 0.0) and (xdim == 1):
            sphere_forward_iso(
                flux[:, 0],
                zero,
                half_angle,
                xs_total,
                zero,
                zero,
                source[:, qq],
                medium_map,
                delta_x,
                angle_x[nn],
                angle_w[nn],
                angle_w[nn],
                tau,
                alpha_plus,
                alpha_minus,
                edges,
            )

        elif (angle_x[nn] < 0.0) and (xdim == 1):
            sphere_backward_iso(
                flux[:, 0],
                zero,
                half_angle,
                xs_total,
                zero,
                zero,
                source[:, qq],
                boundary[1, bc],
                medium_map,
                delta_x,
                angle_x[nn],
                angle_w[nn],
                angle_w[nn],
                tau,
                alpha_plus,
                alpha_minus,
                edges,
            )

        # Angular flux calculation
        if (angle_x[nn] > 0.0) and (xdim > 1):
            sphere_forward_iso(
                flux[:, nn],
                zero,
                half_angle,
                xs_total,
                zero,
                zero,
                source[:, qq],
                medium_map,
                delta_x,
                angle_x[nn],
                angle_w[nn],
                1.0,
                tau,
                alpha_plus,
                alpha_minus,
                edges,
            )

        elif (angle_x[nn] < 0.0) and (xdim > 1):
            sphere_backward_iso(
                flux[:, nn],
                zero,
                half_angle,
                xs_total,
                zero,
                zero,
                source[:, qq],
                boundary[1, bc],
                medium_map,
                delta_x,
                angle_x[nn],
                angle_w[nn],
                1.0,
                tau,
                alpha_plus,
                alpha_minus,
                edges,
            )

        alpha_minus = alpha_plus
        angle_minus = angle_plus

    return flux


########################################################################
# Slab Sweeps - DJINN
########################################################################


def slab_scatter_source_sn(
    xs_total,
    scatter_source,
    external,
    boundary,
    medium_map,
    delta_x,
    angle_x,
    angle_w,
    bc_x,
):
    """Specialized slab sweep for known scatter source (DJINN variant).

    This variant is optimized for DJINN-accelerated transport where the
    scatter source is provided externally rather than computed from flux
    moments. The function performs a single pass without iteration.

    Parameters
    ----------
    xs_total : numpy.ndarray, shape (n_materials,)
        Total cross sections indexed by material.
    scatter_source : numpy.ndarray, shape (n_cells,)
        Precomputed scatter source.
    external : numpy.ndarray, shape (n_cells, n_angles or 1)
        External source term.
    boundary : numpy.ndarray, shape (2, n_angles or 1)
        Boundary conditions.
    medium_map : numpy.ndarray, shape (n_cells,)
        Material indices.
    delta_x : numpy.ndarray, shape (n_cells,)
        Cell widths.
    angle_x : numpy.ndarray, shape (n_angles,)
        Angular ordinates.
    angle_w : numpy.ndarray, shape (n_angles,)
        Quadrature weights.
    bc_x : sequence
        Boundary condition flags.

    Returns
    -------
    numpy.ndarray
        Scalar flux at cell centers (n_cells,).
    """

    cells_x = medium_map.shape[0]
    angles = angle_x.shape[0]

    flux = np.zeros((cells_x,))
    reflector = np.zeros((angles,))
    edge1 = 0.0

    for nn in range(angles):

        qq = 0 if external.shape[1] == 1 else nn
        bc = 0 if external.shape[1] == 1 else nn

        if angle_x[nn] > 0.0:
            edge1 = reflector[nn] + boundary[0, bc]
            edge1 = slab_forward_scatter(
                flux,
                xs_total,
                scatter_source,
                external[:, qq],
                edge1,
                medium_map,
                delta_x,
                angle_x[nn],
                angle_w[nn],
            )

        elif angle_x[nn] < 0.0:
            edge1 = reflector[nn] + boundary[1, bc]
            edge1 = slab_backward_scatter(
                flux,
                xs_total,
                scatter_source,
                external[:, qq],
                edge1,
                medium_map,
                delta_x,
                angle_x[nn],
                angle_w[nn],
            )
        else:
            raise Exception("Discontinuity at 0")

        tools.reflector_corrector(reflector, angle_x, edge1, nn, bc_x)

    return flux


@numba.jit(
    "f8(f8[:], f8[:], f8[:], f8[:], f8, i4[:], f8[:], f8, f8)",
    nopython=True,
    cache=True,
)
def slab_forward_scatter(
    flux,
    xs_total,
    scatter_source,
    external,
    edge1,
    medium_map,
    delta_x,
    angle_x,
    angle_w,
):
    """Forward sweep kernel for slab geometry with known scatter source.

    Performs a forward (left to right) sweep through slab cells using
    diamond difference with prescribed scatter source. Used by DJINN
    accelerated variants.

    Parameters
    ----------
    flux : numpy.ndarray, shape (n_cells,)
        Scalar flux array to update.
    xs_total : numpy.ndarray, shape (n_materials,)
        Total cross sections by material.
    scatter_source : numpy.ndarray, shape (n_cells)
        Known scatter source by cell.
    external : numpy.ndarray, shape (n_cells)
        External source by cell.
    edge1 : float
        Right edge flux for current cell.
    medium_map : numpy.ndarray, shape (n_cells,)
        Material indices by cell.
    delta_x : numpy.ndarray, shape (n_cells,)
        Cell widths.
    angle_x : float
        Current angular ordinate.
    angle_w : float
        Quadrature weight.

    Returns
    -------
    float
        Outgoing edge flux for sweep continuation.
    """
    cells_x = numba.int32(flux.shape[0])
    mat = numba.int32
    ii = numba.int32
    edge2 = numba.float64(0.0)

    for ii in range(cells_x):
        mat = medium_map[ii]

        edge2 = (
            scatter_source[ii]
            + external[ii]
            + edge1 * (angle_x / delta_x[ii] - 0.5 * xs_total[mat])
        ) / (angle_x / delta_x[ii] + 0.5 * xs_total[mat])

        flux[ii] += 0.5 * angle_w * (edge1 + edge2)
        edge1 = edge2

    return edge1


@numba.jit(
    "f8(f8[:], f8[:], f8[:], f8[:], f8, i4[:], f8[:], f8, f8)",
    nopython=True,
    cache=True,
)
def slab_backward_scatter(
    flux,
    xs_total,
    scatter_source,
    external,
    edge1,
    medium_map,
    delta_x,
    angle_x,
    angle_w,
):
    """Backward sweep kernel for slab geometry with known scatter source.

    Performs a backward (right to left) sweep through slab cells using
    diamond difference with prescribed scatter source. Used by DJINN
    accelerated variants.

    Parameters
    ----------
    flux : numpy.ndarray
        Scalar flux array to update, shape (n_cells,).
    xs_total : numpy.ndarray
        Total cross sections by material, shape (n_materials,).
    scatter_source : numpy.ndarray
        Known scatter source by cell, shape (n_cells).
    external : numpy.ndarray
        External source by cell, shape (n_cells).
    edge1 : float
        Right edge flux for current cell.
    medium_map : numpy.ndarray
        Material indices by cell, shape (n_cells,).
    delta_x : numpy.ndarray
        Cell widths, shape (n_cells,).
    angle_x : float
        Current angular ordinate.
    angle_w : float
        Quadrature weight.

    Returns
    -------
    float
        Outgoing edge flux for sweep continuation.
    """
    cells_x = numba.int32(flux.shape[0])
    mat = numba.int32
    ii = numba.int32
    edge2 = numba.float64(0.0)

    for ii in range(cells_x - 1, -1, -1):
        mat = medium_map[ii]

        edge2 = (
            scatter_source[ii]
            + external[ii]
            + edge1 * (-angle_x / delta_x[ii] - 0.5 * xs_total[mat])
        ) / (-angle_x / delta_x[ii] + 0.5 * xs_total[mat])

        flux[ii] += 0.5 * angle_w * (edge1 + edge2)
        edge1 = edge2

    return edge1


########################################################################
# Sphere Sweeps - DJINN
########################################################################


def sphere_scatter_source_sn(
    xs_total,
    scatter_source,
    external,
    boundary,
    medium_map,
    delta_x,
    angle_x,
    angle_w,
    bc_x,
):
    """Specialized sphere sweep for known scatter source (DJINN variant).

    This variant is optimized for DJINN-accelerated transport where the
    scatter source is provided externally rather than computed from flux
    moments. The function performs a single pass without iteration.

    Parameters
    ----------
    flux : numpy.ndarray, shape (n_cells,)
        Scalar flux array to update.
    xs_total : numpy.ndarray, shape (n_materials,)
        Total cross sections by material.
    scatter_source : numpy.ndarray, shape (n_cells)
        Known scatter source by cell.
    external : numpy.ndarray, shape (n_cells)
        External source by cell.
    edge1 : float
        Right edge flux for current cell.
    medium_map : numpy.ndarray, shape (n_cells,)
        Material indices by cell.
    delta_x : numpy.ndarray, shape (n_cells,)
        Cell widths.
    angle_x : float
        Current angular ordinate.
    angle_w : float
        Quadrature weight.
    bc_x : sequence
        Boundary condition flags.

    Returns
    -------
    numpy.ndarray, shape (n_cells)
        Scalar flux at cell centers.
    """

    cells_x = medium_map.shape[0]
    angles = angle_x.shape[0]

    flux = np.zeros((cells_x,))
    half_angle = np.zeros((cells_x,))

    angle_minus = -1.0
    alpha_minus = 0.0

    # Calculate the initial half angle
    _half_angle_scatter(
        half_angle,
        xs_total,
        scatter_source,
        external[:, 0],
        medium_map,
        delta_x,
        boundary[1, 0],
    )

    for nn in range(angles):
        qq = 0 if external.shape[1] == 1 else nn
        bc = 0 if external.shape[1] == 1 else nn

        angle_plus = angle_minus + 2 * angle_w[nn]
        tau = (angle_x[nn] - angle_minus) / (angle_plus - angle_minus)
        alpha_plus = angle_coef_corrector(
            alpha_minus, angle_x[nn], angle_w[nn], nn, angles
        )

        if angle_x[nn] > 0.0:
            sphere_forward_scatter(
                flux,
                half_angle,
                xs_total,
                scatter_source,
                external[:, qq],
                medium_map,
                delta_x,
                angle_x[nn],
                angle_w[nn],
                angle_w[nn],
                tau,
                alpha_plus,
                alpha_minus,
            )

        elif angle_x[nn] < 0.0:
            sphere_backward_scatter(
                flux,
                half_angle,
                xs_total,
                scatter_source,
                external[:, qq],
                boundary[1, bc],
                medium_map,
                delta_x,
                angle_x[nn],
                angle_w[nn],
                angle_w[nn],
                tau,
                alpha_plus,
                alpha_minus,
            )
        else:
            raise Exception("Discontinuity at 0")

        alpha_minus = alpha_plus
        angle_minus = angle_plus

    return flux


@numba.jit(
    "void(f8[:], f8[:], f8[:], f8[:], i4[:], f8[:], f8)", nopython=True, cache=True
)
def _half_angle_scatter(
    half_angle, xs_total, scatter_source, external, medium_map, delta_x, angle_plus
):
    cells_x = numba.int32(medium_map.shape[0])
    mat = numba.int32
    ii = numba.int32
    half_angle *= 0.0

    for ii in range(cells_x - 1, -1, -1):
        mat = medium_map[ii]
        half_angle[ii] = (
            2 * angle_plus + delta_x[ii] * (external[ii] + scatter_source[ii])
        ) / (2 + xs_total[mat] * delta_x[ii])
        angle_plus = 2 * half_angle[ii] - angle_plus


@numba.jit(
    "void(f8[:], f8[:], f8[:], f8[:], f8[:], i4[:], f8[:], f8, \
            f8, f8, f8, f8, f8)",
    nopython=True,
    cache=True,
)
def sphere_forward_scatter(
    flux,
    half_angle,
    xs_total,
    scatter_source,
    external,
    medium_map,
    delta_x,
    angle_x,
    angle_w,
    weight,
    tau,
    alpha_plus,
    alpha_minus,
):
    """Forward sweep kernel for spherical geometry with known scatter source.

    Performs a forward (center to edge) sweep through spherical shells using
    diamond difference with prescribed scatter source and proper spherical
    geometry factors. Used by DJINN accelerated variants.

    Parameters
    ----------
    flux : numpy.ndarray, shape (n_cells,)
        Scalar flux array to update.
    half_angle : numpy.ndarray, shape (n_cells,)
        Half-angle flux values at cell edges.
    xs_total : numpy.ndarray, shape (n_materials,)
        Total cross sections by material.
    scatter_source : numpy.ndarray, shape (n_cells)
        Known scatter source by cell.
    external : numpy.ndarray, shape (n_cells)
        External source by cell.
    medium_map : numpy.ndarray, shape (n_cells,)
        Material indices by cell.
    delta_x : numpy.ndarray, shape (n_cells,)
        Cell widths.
    angle_x : float
        Current angular ordinate.
    angle_w : float
        Quadrature weight.
    weight : float
        Angular weight factor.
    tau : float
        Weighted diamond difference factor.
    alpha_plus : float
        Forward angular coefficient.
    alpha_minus : float
        Backward angular coefficient.
    """
    cells_x = numba.int32(flux.shape[0])
    mat = numba.int32
    ii = numba.int32
    edge1 = numba.float64(half_angle[0])

    area1 = numba.float64
    area2 = numba.float64
    center = numba.float64
    volume = numba.float64

    for ii in range(cells_x):
        mat = medium_map[ii]

        # Calculate surface areas and volume of cell
        area1 = 4 * np.pi * (ii * delta_x[ii]) ** 2
        area2 = 4 * np.pi * ((ii + 1) * delta_x[ii]) ** 2
        volume = (
            4 / 3.0 * np.pi * (((ii + 1) * delta_x[ii]) ** 3 - (ii * delta_x[ii]) ** 3)
        )

        center = (
            angle_x * (area2 + area1) * edge1
            + (1 / angle_w)
            * (area2 - area1)
            * (alpha_plus + alpha_minus)
            * (half_angle[ii])
            + volume * (external[ii] + scatter_source[ii])
        ) / (
            2 * angle_x * area2
            + 2 / angle_w * (area2 - area1) * alpha_plus
            + xs_total[mat] * volume
        )

        flux[ii] += weight * center
        edge1 = 2 * center - edge1
        if ii != 0:
            half_angle[ii] = 1 / tau * (center - (1 - tau) * half_angle[ii])


@numba.jit(
    "void(f8[:], f8[:], f8[:], f8[:], f8[:], f8, i4[:], f8[:], \
            f8, f8, f8, f8, f8, f8)",
    nopython=True,
    cache=True,
)
def sphere_backward_scatter(
    flux,
    half_angle,
    xs_total,
    scatter_source,
    external,
    boundary,
    medium_map,
    delta_x,
    angle_x,
    angle_w,
    weight,
    tau,
    alpha_plus,
    alpha_minus,
):
    """Backward sweep kernel for spherical geometry with known scatter source.

    Performs a backward (edge-to-center) sweep through spherical shells using
    diamond difference with prescribed scatter source and proper spherical
    geometry factors. Used by DJINN accelerated variants.

    Parameters
    ----------
    flux : numpy.ndarray, shape (n_cells,)
        Scalar flux array to update.
    half_angle : numpy.ndarray, shape (n_cells,)
        Half-angle flux values at cell edges.
    xs_total : numpy.ndarray, shape (n_materials,)
        Total cross sections by material.
    scatter_source : numpy.ndarray, shape (n_cells)
        Known scatter source by cell.
    external : numpy.ndarray, shape (n_cells)
        External source by cell.
    medium_map : numpy.ndarray, shape (n_cells,)
        Material indices by cell.
    delta_x : numpy.ndarray, shape (n_cells,)
        Cell widths.
    angle_x : float
        Current angular ordinate.
    angle_w : float
        Quadrature weight.
    weight : float
        Angular weight factor.
    tau : float
        Weighted diamond difference factor.
    alpha_plus : float
        Forward angular coefficient.
    alpha_minus : float
        Backward angular coefficient.
    """
    cells_x = numba.int32(flux.shape[0])
    mat = numba.int32
    ii = numba.int32
    # edge1 = numba.float64(boundary)
    edge1 = numba.float64(0.0)

    area1 = numba.float64
    area2 = numba.float64
    center = numba.float64
    volume = numba.float64

    for ii in range(cells_x - 1, -1, -1):
        mat = medium_map[ii]

        # Calculate surface areas and volume of cell
        area1 = 4 * np.pi * (ii * delta_x[ii]) ** 2
        area2 = 4 * np.pi * ((ii + 1) * delta_x[ii]) ** 2
        volume = (
            4 / 3.0 * np.pi * (((ii + 1) * delta_x[ii]) ** 3 - (ii * delta_x[ii]) ** 3)
        )

        center = (
            -angle_x * (area2 + area1) * edge1
            + (1 / angle_w)
            * (area2 - area1)
            * (alpha_plus + alpha_minus)
            * (half_angle[ii])
            + volume * (external[ii] + scatter_source[ii])
        ) / (
            2 * -angle_x * area1
            + 2 / angle_w * (area2 - area1) * alpha_plus
            + xs_total[mat] * volume
        )

        flux[ii] += weight * center
        edge1 = 2 * center - edge1
        if ii != 0:
            half_angle[ii] = 1 / tau * (center - (1 - tau) * half_angle[ii])
