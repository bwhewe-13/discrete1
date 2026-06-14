"""Anisotropic critical slab: standalone diamond-difference S_N reference."""

import numpy as np

import discrete1
from discrete1 import tools

count_kk = 100
change_kk = 1e-06

count_nn = 100
change_nn = 1e-12

count_gg = 100
change_gg = 1e-08


def power_iteration(
    xs_total,
    xs_scatter,
    xs_fission,
    medium_map,
    delta_x,
    angle_x,
    angle_w,
    bc_x,
    chi=None,
    geometry=1,
):
    """Run power iteration for 1D multigroup problems.

    Parameters
    ----------
    xs_total, xs_scatter, xs_fission : numpy.ndarray
        Cross section arrays indexed by material.
    medium_map : array_like
        Spatial medium mapping (length I).
    delta_x : array_like
        Cell widths.
    angle_x, angle_w : array_like
        Angular ordinates and weights.
    bc_x : list-like
        Boundary condition indicators.
    chi : numpy.ndarray, optional
        Fission Neutron Distribution. Must be included if xs_fission is nusigf.
    geometry : int, optional
        Geometry selector (1=slab, 2=sphere).

    Returns
    -------
    tuple
        (flux, keff) converged scalar flux and multiplication factor.
    """
    # Set boundary source
    boundary = np.zeros((2, 1, 1))

    # Initialize and normalize flux
    cells_x = medium_map.shape[0]
    flux_old = np.random.rand(cells_x, xs_total.shape[1])
    keff = np.linalg.norm(flux_old)
    flux_old /= np.linalg.norm(keff)

    # Initialize power source
    source = np.zeros((cells_x, 1, xs_total.shape[1]))

    converged = False
    count = 0
    change = 0.0

    while not (converged):
        # Update power source term
        if chi is None:
            tools.fission_mat_prod(flux_old, xs_fission, source, medium_map, keff)
        else:
            tools.fission_vec_prod(flux_old, chi, xs_fission, source, medium_map, keff)

        # Solve for scalar flux
        flux = source_iteration(
            flux_old,
            xs_total,
            xs_scatter,
            source,
            boundary,
            medium_map,
            delta_x,
            angle_x,
            angle_w,
            bc_x,
            geometry,
        )

        # Update keffective
        if chi is None:
            keff = tools._update_keff_mat(flux, flux_old, xs_fission, medium_map, keff)
        else:
            keff = tools._update_keff_vec(
                flux, flux_old, chi, xs_fission, medium_map, keff
            )

        # Normalize flux
        flux /= np.linalg.norm(flux)

        # Check for convergence
        change = np.linalg.norm((flux - flux_old) / flux / cells_x)
        print(f"Count: {count:>2}\tKeff: {keff:.8f}", end="\r")
        converged = (change < change_kk) or (count >= count_kk)
        count += 1

        flux_old = flux.copy()

    print(f"\nConvergence: {change:2.6e}")
    return flux, keff


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
):
    """Multigroup source iteration solver.

    Performs outer iterations over energy groups using inner discrete
    ordinates sweeps for each group. Off-diagonal scatter (up- and
    down-scatter) contributions are handled via an "off_scatter"
    accumulator that uses the previous and current flux iterates.

    Parameters
    ----------
    flux_old : numpy.ndarray
        Initial scalar flux guess with shape (n_cells, n_groups).
    xs_total : array_like
        Total cross sections (n_materials or n_cells, n_groups).
    xs_scatter : array_like
        Scatter cross sections (indexed by material, GxG).
    external : numpy.ndarray
        External source array (n_cells, n_angles, n_groups) or with
        singleton dimensions where appropriate.
    boundary : numpy.ndarray
        Boundary conditions array (2, n_angles, n_groups) or with
        singleton dimensions where appropriate.
    medium_map : array_like
        Material index per spatial cell (n_cells,).
    delta_x : array_like
        Cell widths (n_cells,).
    angle_x : array_like
        Angular ordinates (n_angles,).
    angle_w : array_like
        Quadrature weights (n_angles,).
    bc_x : sequence
        Boundary condition flags for left/right boundaries.
    geometry : int
        Geometry selector (1=slab, 2=sphere).

    Returns
    -------
    numpy.ndarray
        Converged scalar flux array with shape (n_cells, n_groups).
    """
    cells_x, groups = flux_old.shape
    flux = np.zeros((cells_x, groups))
    off_scatter = np.zeros((cells_x,))

    converged = False
    count = 1
    change = 0.0

    while not (converged):
        flux *= 0.0

        for gg in range(groups):
            # Check for sizes
            qq = 0 if external.shape[2] == 1 else gg
            bc = 0 if boundary.shape[2] == 1 else gg

            # Update off scatter source
            tools._off_scatter(flux, flux_old, medium_map, xs_scatter, off_scatter, gg)

            # Run discrete ordinates for one group
            flux[:, gg] = discrete_ordinates(
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
                geometry,
            )

        # Check for convergence
        try:
            change = np.linalg.norm((flux - flux_old) / flux / cells_x)
        except RuntimeWarning:
            change = 0.0
        converged = (change < change_gg) or (count >= count_gg)
        count += 1

        flux_old = flux.copy()

    return flux


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
    cells_x = medium_map.shape[0]
    # Initialize unknown cell edge
    edge2 = 0.0
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
    cells_x = medium_map.shape[0]
    # Initialize unknown cell edge
    edge2 = 0.0
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


def legendre_polynomial(moment, mu):
    """Evaluate the Legendre polynomial of order `moment` at `mu`."""
    if moment == 0:
        return 1.0
    elif moment == 1:
        return mu
    else:
        return (
            (2 * moment - 1) * mu * legendre_polynomial(moment - 1, mu)
            - (moment - 1) * legendre_polynomial(moment - 2, mu)
        ) / moment


def anisotropic_scatter(angular_flux, xs_scatter, angle_x, angle_w):
    """Compute the anisotropic scatter contribution for a given set of moments."""
    source = 0.0
    for moment in range(xs_scatter.shape[0]):
        source += (
            (2 * moment + 1)
            * xs_scatter[moment]
            * angle_w
            * legendre_polynomial(moment, angle_x)
            * angular_flux
        )
    return source


if __name__ == "__main__":
    cells_x = 50
    angles = 8
    bc_x = [0, 0]
    angle_x, angle_w = discrete1.angular_x(angles, bc_x)
    xs_total = np.array([[1.0]])
    xs_scatter = np.array([[[[0.733333]], [[0.2]], [[0.075]]]])
    xs_fission = np.array([[[2.5 * 0.266667]]])
    L = 2
    medium_map = np.zeros((cells_x,), dtype=np.int32)
    length = 0.77032 * 2
    edges_x = np.linspace(0, length, cells_x + 1)
    delta_x = np.repeat(length / cells_x, cells_x)
    flux, keff = power_iteration(
        xs_total, xs_scatter, xs_fission, medium_map, delta_x, angle_x, angle_w, bc_x
    )
    print(f"Final Keff: {keff:.8f}")
