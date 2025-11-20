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
- ``slab_ordinates_source`` / ``sphere_ordinates_source``: known-source
- ``slab_ordinates_scatter`` / ``sphere_ordinates_scatter``: DJINN scatter

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
):
    """Dispatch to the appropriate spatial sweep for the geometry.

    This function chooses the slab or sphere sweeping routine based on
    the ``geometry`` argument and returns the updated scalar flux array
    evaluated at cell centers.

    Parameters
    ----------
    flux_old : numpy.ndarray
        Previous iterate of the scalar flux (shape: n_cells,).
    xs_total : array_like
        Total macroscopic cross sections per material.
    xs_scatter : array_like
        Scatter cross section (per material) used in source evaluation.
    off_scatter : numpy.ndarray
        Off-diagonal scatter correction/source term (shape: n_cells,).
    external : numpy.ndarray
        External source array (shape varies: spatial x angles x groups).
    boundary : numpy.ndarray
        Boundary condition array (2, n_angles, ...).
    medium_map : array_like
        Integer material index per spatial cell (length n_cells).
    delta_x : array_like
        Cell widths (length n_cells).
    angle_x : array_like
        Angular ordinates (length n_angles).
    angle_w : array_like
        Quadrature weights (length n_angles).
    bc_x : sequence
        Boundary condition flags for left/right boundaries.
    geometry : int
        Geometry selector: 1 -> slab, 2 -> sphere.

    Returns
    -------
    numpy.ndarray
        Updated scalar flux at cell centers (shape: n_cells,).
    """
    edges = 0

    # Slab geometry
    if geometry == 1:
        return slab_ordinates(
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

    # Sphere geometry
    elif geometry == 2:
        return sphere_ordinates(
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


def _known_source(
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

    # Slab geometry
    if geometry == 1:
        return slab_ordinates_source(
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
        return sphere_ordinates_source(
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


def _known_scatter(
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

    # Slab geometry
    if geometry == 1:
        return slab_ordinates_scatter(
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
        return sphere_ordinates_scatter(
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
# Slab Sweeps
########################################################################


def slab_ordinates(
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
    """Perform a slab-geometry discrete-ordinates spatial sweep.

    The solver iterates over angular ordinates and performs forward and
    backward diamond-difference passes until the scalar flux converges
    (or a maximum iteration count is reached). This routine returns the
    converged cell-centered scalar flux.

    Parameters
    ----------
    flux_old : numpy.ndarray
        Initial flux guess (n_cells,).
    xs_total : array_like
        Total cross sections indexed by material.
    xs_scatter : array_like
        Scatter cross sections indexed by material.
    off_scatter : numpy.ndarray
        Precomputed off-scatter source (n_cells,).
    external : numpy.ndarray
        External source (n_x, n_angles or 1, 1).
    boundary : numpy.ndarray
        Boundary conditions (2, n_angles, ...).
    medium_map : array_like
        Material index per cell (n_cells,).
    delta_x : array_like
        Cell widths (n_cells,).
    angle_x, angle_w : array_like
        Angular ordinates and weights.
    bc_x : sequence
        Boundary condition flags for left/right.
    edges : int
        If 1, compute edge flux contributions; otherwise compute center flux.

    Returns
    -------
    numpy.ndarray
        Converged scalar flux at cell centers (n_cells,).
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
                edge1 = slab_forward(
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
                edge1 = slab_backward(
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
def slab_forward(
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
    """Perform the forward sweep for slab geometry for one ordinate.

    This kernel marches from the left boundary toward the right and
    computes either edge or center flux contributions for a single
    angular ordinate.

    Notes
    -----
    This function is JIT-compiled with numba for performance and
    operates in-place on the ``flux`` array.
    """

    # Get iterables
    cells_x = numba.int32(medium_map.shape[0])
    mat = numba.int32
    ii = numba.int32
    # Initialize unknown cell edge
    edge2 = numba.float64(0.0)
    # Determine flux edge
    if edges == 1:
        flux[0] += angle_w * edge1
    # Iterate over cells
    for ii in range(cells_x):
        # Determining material cross section
        mat = medium_map[ii]
        # Calculate cell edge unknown
        edge2 = (
            (
                xs_scatter[mat] * flux_old[ii]
                + external[ii]
                + off_scatter[ii]
                + edge1 * (angle_x / delta_x[ii] - 0.5 * xs_total[mat])
            )
            * 1
            / (angle_x / delta_x[ii] + 0.5 * xs_total[mat])
        )
        # Update flux with cell edges
        if edges == 1:
            flux[ii + 1] += angle_w * edge2
        # Update flux with cell centers
        else:
            flux[ii] += 0.5 * angle_w * (edge1 + edge2)
        # Update unknown cell edge
        edge1 = edge2
    # Return cell at i = I
    return edge1


@numba.jit(
    "f8(f8[:], f8[:], f8[:], f8[:], f8[:], f8[:], f8, i4[:], \
            f8[:], f8, f8, i4)",
    nopython=True,
    cache=True,
)
def slab_backward(
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
    """Perform the backward sweep for slab geometry for one ordinate.

    Marches from the right boundary toward the left and updates the
    provided ``flux`` array with center or edge contributions for the
    given angular ordinate. This kernel is jitted with numba and
    mutates its inputs in-place for performance.
    """

    # Get iterables
    cells_x = numba.int32(medium_map.shape[0])
    mat = numba.int32
    ii = numba.int32
    # Initialize unknown cell edge
    edge2 = numba.float64(0.0)
    # Determine flux edge
    if edges == 1:
        flux[cells_x] += angle_w * edge1
    # Iterate over cells
    for ii in range(cells_x - 1, -1, -1):
        # Determining material cross section
        mat = medium_map[ii]
        # Calculate cell edge unknown
        edge2 = (
            (
                xs_scatter[mat] * flux_old[ii]
                + external[ii]
                + off_scatter[ii]
                + edge1 * (-angle_x / delta_x[ii] - 0.5 * xs_total[mat])
            )
            * 1
            / (-angle_x / delta_x[ii] + 0.5 * xs_total[mat])
        )
        # Update flux with cell edges
        if edges == 1:
            flux[ii] += angle_w * edge2
        # Update flux with cell centers
        else:
            flux[ii] += 0.5 * angle_w * (edge1 + edge2)
        # Update unknown cell edge
        edge1 = edge2
    # Return cell at i = 0
    return edge1


########################################################################
# Sphere Sweeps
########################################################################


def sphere_ordinates(
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
    """Perform a spherical-geometry discrete-ordinates spatial sweep.

    The spherical sweep uses half-angle bookkeeping and angular
    differencing coefficients appropriate for radial geometry. The
    implementation iterates over ordinates and updates a converged
    scalar flux similar to the slab driver.

    Parameters
    ----------
    flux_old, xs_total, xs_scatter, off_scatter, external, boundary, \
    medium_map, delta_x, angle_x, angle_w, bc_x, edges :
        See :func:`slab_ordinates` for parameter shapes and meanings.

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
        _half_angle(
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

        # Iterate over the discrete ordinates
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
                sphere_forward(
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
                sphere_backward(
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

            # Update the angular differencing coefficient
            alpha_minus = alpha_plus
            # Update the half angle
            angle_minus = angle_plus

        # Check for convergence
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
def _half_angle(
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
    # Get iterables
    cells_x = numba.int32(flux.shape[0])
    mat = numba.int32
    ii = numba.int32
    # Zero out half angle
    half_angle *= 0.0
    # Iterate from sphere surface to center
    for ii in range(cells_x - 1, -1, -1):
        mat = medium_map[ii]
        # Calculate angular flux half angle
        half_angle[ii] = (
            2 * angle_plus
            + delta_x[ii]
            * (external[ii] + off_scatter[ii] + xs_scatter[mat] * flux[ii])
        ) / (2 + xs_total[mat] * delta_x[ii])
        # Update half angle coefficient
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
def sphere_forward(
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

    # Get iterables
    cells_x = numba.int32(flux.shape[0])
    mat = numba.int32
    ii = numba.int32

    # Initialize known cell edge
    edge1 = numba.float64(half_angle[0])

    # Determine flux edge
    if edges == 1:
        flux[0] += weight * edge1

    # Initialize surface area on cell edges, cell volume, flux center
    area1 = numba.float64
    area2 = numba.float64
    center = numba.float64
    volume = numba.float64

    # Iterate over cells from 0 -> I (center to edge)
    for ii in range(cells_x):
        # For determining the material cross sections
        mat = medium_map[ii]

        # Calculate surface areas
        area1 = 4 * np.pi * (ii * delta_x[ii]) ** 2
        area2 = 4 * np.pi * ((ii + 1) * delta_x[ii]) ** 2
        # Calculate volume of cell
        volume = (
            4 / 3.0 * np.pi * (((ii + 1) * delta_x[ii]) ** 3 - (ii * delta_x[ii]) ** 3)
        )

        # Calculate flux at cell center
        center = (
            angle_x * (area2 + area1) * edge1
            + 1
            / angle_w
            * (area2 - area1)
            * (alpha_plus + alpha_minus)
            * (half_angle[ii])
            + volume * (external[ii] + off_scatter[ii] + flux_old[ii] * xs_scatter[mat])
        ) / (
            2 * angle_x * area2
            + 2 / angle_w * (area2 - area1) * alpha_plus
            + xs_total[mat] * volume
        )

        # Update flux with cell edges
        if edges == 1:
            flux[ii + 1] += weight * (2 * center - edge1)
        # Update flux with cell centers
        else:
            flux[ii] += weight * center

        # Update cell edge
        edge1 = 2 * center - edge1

        # Update half angle coefficient
        if ii != 0:
            half_angle[ii] = 1 / tau * (center - (1 - tau) * half_angle[ii])


@numba.jit(
    "void(f8[:], f8[:], f8[:], f8[:], f8[:], f8[:], f8[:], f8, i4[:], \
        f8[:], f8, f8, f8, f8, f8, f8, i4)",
    nopython=True,
    cache=True,
)
def sphere_backward(
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

    # Get iterables
    cells_x = numba.int32(flux.shape[0])
    mat = numba.int32
    ii = numba.int32

    # Initialize known cell edge
    edge1 = numba.float64(boundary)

    # Determine flux edge
    if edges == 1:
        flux[cells_x] += weight * edge1

    # Initialize surface area on cell edges, cell volume, flux center
    area1 = numba.float64
    area2 = numba.float64
    center = numba.float64
    volume = numba.float64

    # Iterate over cells from I -> 0 (edge to center)
    for ii in range(cells_x - 1, -1, -1):
        # For determining the material cross sections
        mat = medium_map[ii]

        # Calculate surface areas
        area1 = 4 * np.pi * (ii * delta_x[ii]) ** 2
        area2 = 4 * np.pi * ((ii + 1) * delta_x[ii]) ** 2
        # Calculate volume of cell
        volume = (
            4 / 3.0 * np.pi * (((ii + 1) * delta_x[ii]) ** 3 - (ii * delta_x[ii]) ** 3)
        )

        # Calculate the flux at the cell center
        center = (
            -angle_x * (area2 + area1) * edge1
            + 1
            / angle_w
            * (area2 - area1)
            * (alpha_plus + alpha_minus)
            * (half_angle[ii])
            + volume * (external[ii] + off_scatter[ii] + flux_old[ii] * xs_scatter[mat])
        ) / (
            2 * -angle_x * area1
            + 2 / angle_w * (area2 - area1) * alpha_plus
            + xs_total[mat] * volume
        )

        # Update flux with cell edges
        if edges == 1:
            flux[ii + 1] += weight * (2 * center - edge1)
        # Update flux with cell centers
        else:
            flux[ii] += weight * center

        # Update cell edge
        edge1 = 2 * center - edge1

        # Update half angle coefficient
        if ii != 0:
            half_angle[ii] = 1 / tau * (center - (1 - tau) * half_angle[ii])


########################################################################
# Slab Sweeps - Known Source
########################################################################


def slab_ordinates_source(
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
        Output array for scalar or angular fluxes (shape: n_cells x xdim).
    xs_total : array_like
        Total cross sections indexed by material.
    zero : numpy.ndarray
        Zero-array placeholder used for kernels that expect an array.
    source : numpy.ndarray
        Prescribed source (spatial x angular-or-1).
    boundary, medium_map, delta_x, angle_x, angle_w, bc_x, edges
        See :func:`slab_ordinates` for descriptions.

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
            edge1 = slab_forward(
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
            edge1 = slab_backward(
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
            edge1 = slab_forward(
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
            edge1 = slab_backward(
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


def sphere_ordinates_source(
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
        Output array for scalar/angular fluxes (n_cells x xdim).
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

    # Initialize sphere coefficients
    angle_minus = -1.0
    alpha_minus = 0.0

    # Calculate the initial half angle
    _half_angle(
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

    # Iterate over the discrete ordinates
    for nn in range(angles):

        qq = 0 if source.shape[1] == 1 else nn
        bc = 0 if boundary.shape[1] == 1 else nn

        # Calculate the half angle coefficient
        angle_plus = angle_minus + 2 * angle_w[nn]
        # Calculate the weighted diamond
        tau = (angle_x[nn] - angle_minus) / (angle_plus - angle_minus)
        # Calculate the angular differencing coefficient
        alpha_plus = angle_coef_corrector(
            alpha_minus, angle_x[nn], angle_w[nn], nn, angles
        )

        # Scalar flux calculation
        if (angle_x[nn] > 0.0) and (xdim == 1):
            sphere_forward(
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
            sphere_backward(
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
            sphere_forward(
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
            sphere_backward(
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

        # Update the angular differencing coefficient
        alpha_minus = alpha_plus
        # Update the half angle
        angle_minus = angle_plus

    return flux


########################################################################
# Slab Sweeps - DJINN
########################################################################


def slab_ordinates_scatter(
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
    xs_total : array_like
        Total cross sections indexed by material.
    scatter_source : numpy.ndarray
        Precomputed scatter source (n_cells,).
    external : numpy.ndarray
        External source term (n_cells, n_angles or 1).
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

    Performs a forward (left-to-right) sweep through slab cells using
    diamond difference with prescribed scatter source. Used by DJINN
    accelerated variants.

    Parameters
    ----------
    flux : numpy.ndarray
        Scalar flux array to update (n_cells).
    xs_total : array_like
        Total cross sections by material.
    scatter_source : array_like
        Known scatter source by cell.
    external : array_like
        External source by cell.
    edge1 : float
        Left edge flux for current cell.
    medium_map : array_like
        Material indices by cell.
    delta_x : array_like
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
    # Get iterables
    cells_x = numba.int32(flux.shape[0])
    mat = numba.int32
    ii = numba.int32
    # Initialize unknown cell edge
    edge2 = numba.float64(0.0)
    # Iterate over cells
    for ii in range(cells_x):
        # Determining material cross section
        mat = medium_map[ii]
        # Calculate cell edge unknown
        edge2 = (
            (
                scatter_source[ii]
                + external[ii]
                + edge1 * (angle_x / delta_x[ii] - 0.5 * xs_total[mat])
            )
            * 1
            / (angle_x / delta_x[ii] + 0.5 * xs_total[mat])
        )
        # Update flux with cell centers
        flux[ii] += 0.5 * angle_w * (edge1 + edge2)
        # Update unknown cell edge
        edge1 = edge2
    # Return cell at i = I
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

    Performs a backward (right-to-left) sweep through slab cells using
    diamond difference with prescribed scatter source. Used by DJINN
    accelerated variants.

    Parameters
    ----------
    flux : numpy.ndarray
        Scalar flux array to update (n_cells).
    xs_total : array_like
        Total cross sections by material.
    scatter_source : array_like
        Known scatter source by cell.
    external : array_like
        External source by cell.
    edge1 : float
        Right edge flux for current cell.
    medium_map : array_like
        Material indices by cell.
    delta_x : array_like
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
    # Get iterables
    cells_x = numba.int32(flux.shape[0])
    mat = numba.int32
    ii = numba.int32
    # Initialize unknown cell edge
    edge2 = numba.float64(0.0)
    # Iterate over cells
    for ii in range(cells_x - 1, -1, -1):
        # Determining material cross section
        mat = medium_map[ii]
        # Calculate cell edge unknown
        edge2 = (
            (
                scatter_source[ii]
                + external[ii]
                + edge1 * (-angle_x / delta_x[ii] - 0.5 * xs_total[mat])
            )
            * 1
            / (-angle_x / delta_x[ii] + 0.5 * xs_total[mat])
        )
        # Update flux with cell centers
        flux[ii] += 0.5 * angle_w * (edge1 + edge2)
        # Update unknown cell edge
        edge1 = edge2
    # Return cell at i = 0
    return edge1


########################################################################
# Sphere Sweeps - DJINN
########################################################################


def sphere_ordinates_scatter(
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
    xs_total : array_like
        Total cross sections indexed by material.
    scatter_source : numpy.ndarray
        Precomputed scatter source (n_cells,).
    external : numpy.ndarray
        External source term (n_cells, n_angles or 1).
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

    Returns
    -------
    numpy.ndarray
        Scalar flux at cell centers (n_cells,).
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

    # Iterate over the discrete ordinates
    for nn in range(angles):

        qq = 0 if external.shape[1] == 1 else nn
        bc = 0 if external.shape[1] == 1 else nn

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

        # Iterate from I -> 0
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

        # Update the angular differencing coefficient
        alpha_minus = alpha_plus
        # Update the half angle
        angle_minus = angle_plus

    return flux


@numba.jit(
    "void(f8[:], f8[:], f8[:], f8[:], i4[:], f8[:], f8)", nopython=True, cache=True
)
def _half_angle_scatter(
    half_angle, xs_total, scatter_source, external, medium_map, delta_x, angle_plus
):
    # Get iterables
    cells_x = numba.int32(medium_map.shape[0])
    mat = numba.int32
    ii = numba.int32
    # Zero out half angle
    half_angle *= 0.0
    # Iterate from sphere surface to center
    for ii in range(cells_x - 1, -1, -1):
        mat = medium_map[ii]
        # Calculate angular flux half angle
        half_angle[ii] = (
            2 * angle_plus + delta_x[ii] * (external[ii] + scatter_source[ii])
        ) / (2 + xs_total[mat] * delta_x[ii])
        # Update half angle coefficient
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

    Performs a forward (center-to-edge) sweep through spherical shells using
    diamond difference with prescribed scatter source and proper spherical
    geometry factors. Used by DJINN accelerated variants.

    Parameters
    ----------
    flux : numpy.ndarray
        Scalar flux array to update (n_cells).
    half_angle : array_like
        Half-angle flux values at cell edges.
    xs_total : array_like
        Total cross sections by material.
    scatter_source : array_like
        Known scatter source by cell.
    external : array_like
        External source by cell.
    medium_map : array_like
        Material indices by cell.
    delta_x : array_like
        Shell thicknesses.
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
    # Get iterables
    cells_x = numba.int32(flux.shape[0])
    mat = numba.int32
    ii = numba.int32

    # Initialize known cell edge
    edge1 = numba.float64(half_angle[0])

    # Initialize surface area on cell edges, cell volume, flux center
    area1 = numba.float64
    area2 = numba.float64
    center = numba.float64
    volume = numba.float64

    # Iterate over cells from 0 -> I (center to edge)
    for ii in range(cells_x):
        # For determining the material cross sections
        mat = medium_map[ii]

        # Calculate surface areas
        area1 = 4 * np.pi * (ii * delta_x[ii]) ** 2
        area2 = 4 * np.pi * ((ii + 1) * delta_x[ii]) ** 2

        # Calculate volume of cell
        volume = (
            4 / 3.0 * np.pi * (((ii + 1) * delta_x[ii]) ** 3 - (ii * delta_x[ii]) ** 3)
        )

        # Calculate flux at cell center
        center = (
            angle_x * (area2 + area1) * edge1
            + 1
            / angle_w
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

        # Update half angle coefficient
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
    flux : numpy.ndarray
        Scalar flux array to update (n_cells).
    half_angle : array_like
        Half-angle flux values at cell edges.
    xs_total : array_like
        Total cross sections by material.
    scatter_source : array_like
        Known scatter source by cell.
    external : array_like
        External source by cell.
    boundary : float
        Outer boundary condition.
    medium_map : array_like
        Material indices by cell.
    delta_x : array_like
        Shell thicknesses.
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
    # Get iterables
    cells_x = numba.int32(flux.shape[0])
    mat = numba.int32
    ii = numba.int32

    # Initialize known cell edge
    edge1 = numba.float64(boundary)
    edge1 = 0.0

    # Initialize surface area on cell edges, cell volume, flux center
    area1 = numba.float64
    area2 = numba.float64
    center = numba.float64
    volume = numba.float64

    # Iterate over cells from I -> 0 (edge to center)
    for ii in range(cells_x - 1, -1, -1):
        # For determining the material cross sections
        mat = medium_map[ii]

        # Calculate surface areas
        area1 = 4 * np.pi * (ii * delta_x[ii]) ** 2
        area2 = 4 * np.pi * ((ii + 1) * delta_x[ii]) ** 2
        # Calculate volume of cell
        volume = (
            4 / 3.0 * np.pi * (((ii + 1) * delta_x[ii]) ** 3 - (ii * delta_x[ii]) ** 3)
        )

        # Calculate the flux at the cell center
        center = (
            -angle_x * (area2 + area1) * edge1
            + 1
            / angle_w
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

        # Update half angle coefficient
        if ii != 0:
            half_angle[ii] = 1 / tau * (center - (1 - tau) * half_angle[ii])
