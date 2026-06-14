"""Anisotropic slab with angular-flux Legendre-moment scatter source."""

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


def build_anisotropic_source(angular_flux, xs_scatter_cell, angle_x, angle_w):
    """Build the within-group anisotropic scatter source S(x, mu).

    Parameters
    ----------
    angular_flux : numpy.ndarray, shape (cells_x, angles)
        Angular flux from the current inner iteration.
    xs_scatter_cell : numpy.ndarray, shape (cells_x, L+1)
        Legendre moments of the scatter cross section mapped to cells.
    angle_x : numpy.ndarray, shape (angles,)
    angle_w : numpy.ndarray, shape (angles,)

    Returns
    -------
    aniso_source : numpy.ndarray, shape (cells_x, angles)
    """
    cells_x, angles = angular_flux.shape
    n_moments = xs_scatter_cell.shape[1]

    # P[l, n] = P_l(mu_n), shape (L+1, angles)
    P = np.array(
        [
            [legendre_polynomial(ll, angle_x[nn]) for nn in range(angles)]
            for ll in range(n_moments)
        ]
    )

    # Flux moments from the angular flux:
    # phi_l(x) = sum_n w_n * P_l(mu_n) * psi(x, mu_n)
    # shape: (cells_x, L+1)
    # einsum: for each cell x and moment l, sum over angles n
    flux_moments = np.einsum("xn,n,ln->xl", angular_flux, angle_w, P)

    # S(x, mu_n) = sum_l (2l+1) * xs_l(x) * P_l(mu_n) * phi_l(x)
    weights = np.arange(n_moments) * 2 + 1  # (2l+1)
    aniso_source = np.einsum(
        "xl,xl,ln->xn",
        xs_scatter_cell * weights[np.newaxis, :],
        flux_moments,
        P,
    )
    return aniso_source  # shape (cells_x, angles)


def discrete_ordinates(
    flux_old,
    xs_total,
    xs_scatter_cell,
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
    """Inner S_N sweep with a rebuilt anisotropic source each iteration."""
    cells_x = flux_old.shape[0]
    angles = angle_x.shape[0]

    flux = np.zeros((cells_x,))
    reflector = np.zeros((angles,))
    edge1 = 0.0

    # Angular flux must be tracked so higher Legendre moments can be formed
    angular_flux = np.zeros((cells_x, angles))

    converged = False
    count = 1
    change = 0.0

    while not converged:

        flux *= 0.0
        reflector *= 0.0

        # Seed aniso source from angular flux of previous inner iteration.
        # On the first pass angular_flux is zero so this reduces to the
        # external + off_scatter only — correct behaviour for iteration 1.
        aniso_source = build_anisotropic_source(
            angular_flux, xs_scatter_cell, angle_x, angle_w
        )

        # Reset angular flux accumulator for this sweep
        angular_flux *= 0.0

        for nn in range(angles):

            qq = 0 if external.shape[1] == 1 else nn
            bc = 0 if boundary.shape[1] == 1 else nn

            aniso_nn = aniso_source[:, nn]

            if angle_x[nn] > 0.0:
                edge1 = reflector[nn] + boundary[0, bc]
                edge1, psi_nn = slab_forward(
                    flux,
                    xs_total,
                    aniso_nn,
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
                edge1, psi_nn = slab_backward(
                    flux,
                    xs_total,
                    aniso_nn,
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

            # Store cell-centre angular flux for this ordinate
            angular_flux[:, nn] = psi_nn

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
    edges=0,
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
        Total cross sections (n_materials, n_groups).
    xs_scatter : array_like
        Scatter cross sections (n_materials, n_groups, n_groups, L+1),
        where the last axis holds the Legendre moments.
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
        Quadrature weights.
    bc_x : sequence
        Boundary condition flags for left/right boundaries.
    edges : int
        If 1, compute edge flux; otherwise compute cell-centre flux.

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

    while not converged:
        flux *= 0.0

        for gg in range(groups):
            # Check for sizes
            qq = 0 if external.shape[2] == 1 else gg
            bc = 0 if boundary.shape[2] == 1 else gg

            # Update off-diagonal scatter source (L=0 only)
            tools._off_scatter(
                flux, flux_old, medium_map, xs_scatter[..., 0], off_scatter, gg
            )

            # Extract within-group Legendre moments for group gg -> (n_materials, L+1)
            # then map to cells -> (cells_x, L+1)
            xs_scatter_cell = xs_scatter[:, gg, gg, :][medium_map]

            # Run discrete ordinates for one group, passing xs_scatter_cell
            # so the anisotropic source can be rebuilt each inner iteration
            flux[:, gg] = discrete_ordinates(
                flux_old[:, gg],
                xs_total[:, gg],
                xs_scatter_cell,
                off_scatter,
                external[:, :, qq],
                boundary[:, :, bc],
                medium_map,
                delta_x,
                angle_x,
                angle_w,
                bc_x,
                edges,
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


def slab_forward(
    flux,
    xs_total,
    aniso_source,
    off_scatter,
    external,
    edge1,
    medium_map,
    delta_x,
    angle_x,
    angle_w,
    edges,
):
    """Forward slab sweep returning the boundary edge and cell-centre flux."""
    cells_x = medium_map.shape[0]
    edge2 = 0.0
    psi = np.zeros((cells_x,))  # cell-centre angular flux for this ordinate

    if edges == 1:
        flux[0] += angle_w * edge1

    for ii in range(cells_x):
        mat = medium_map[ii]
        edge2 = (
            aniso_source[ii]
            + external[ii]
            + off_scatter[ii]
            + edge1 * (angle_x / delta_x[ii] - 0.5 * xs_total[mat])
        ) / (angle_x / delta_x[ii] + 0.5 * xs_total[mat])
        psi[ii] = 0.5 * (edge1 + edge2)  # cell-centre angular flux
        if edges == 1:
            flux[ii + 1] += angle_w * edge2
        else:
            flux[ii] += angle_w * psi[ii]
        edge1 = edge2

    return edge1, psi  # return both the boundary edge and angular flux


def slab_backward(
    flux,
    xs_total,
    aniso_source,
    off_scatter,
    external,
    edge1,
    medium_map,
    delta_x,
    angle_x,
    angle_w,
    edges,
):
    """Backward slab sweep returning the boundary edge and cell-centre flux."""
    cells_x = medium_map.shape[0]
    edge2 = 0.0
    psi = np.zeros((cells_x,))

    if edges == 1:
        flux[cells_x] += angle_w * edge1

    for ii in range(cells_x - 1, -1, -1):
        mat = medium_map[ii]
        edge2 = (
            aniso_source[ii]
            + external[ii]
            + off_scatter[ii]
            + edge1 * (-angle_x / delta_x[ii] - 0.5 * xs_total[mat])
        ) / (-angle_x / delta_x[ii] + 0.5 * xs_total[mat])
        psi[ii] = 0.5 * (edge1 + edge2)
        if edges == 1:
            flux[ii] += angle_w * edge2
        else:
            flux[ii] += angle_w * psi[ii]
        edge1 = edge2

    return edge1, psi


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
    """Compute the anisotropic scatter contribution for a given set of moments.

    .. deprecated::
        This helper is superseded by :func:`build_anisotropic_source` and
        is retained only for reference.
    """
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
    # cells_x = 500
    # angles = 64
    # bc_x = [0, 0]
    # angle_x, angle_w = discrete1.angular_x(angles, bc_x)
    # xs_total = np.array([[1.0]])  # (n_materials=1, n_groups=1)
    # xs_fission = np.array([[[2.5 * 0.266667]]])  # (n_materials=1, n_groups=1)

    # # xs_scatter shape: (n_materials, n_groups, n_groups, L+1)
    # # Last axis holds Legendre moments [sigma_s0, sigma_s1, sigma_s2]
    # xs_scatter = np.array([[[[0.733333, 0.2, 0.075]]]])  # (1, 1, 1, 3)
    # # xs_scatter = np.array([[[[0.733333, 0.0, 0.0]]]])
    # medium_map = np.zeros((cells_x,), dtype=np.int32)
    # # length = 0.77032 * 2
    # length = 0.76378 * 2
    # delta_x = np.repeat(length / cells_x, cells_x)
    # flux, keff = power_iteration(
    #     xs_total, xs_scatter, xs_fission, medium_map, delta_x, angle_x, angle_w, bc_x
    # )
    # print(f"Final Keff: {keff:.8f}")

    bc_x = [0, 0]
    cells_x = 200
    angles = 16
    angle_x, angle_w = discrete1.angular_x(angles, bc_x)
    xs_total = np.array([[2.52025, 0.65696]])
    # xs_scatter = np.array([np.array([[[2.44383, 0.0], [0.029227, 0.62568]]]).T])
    xs_scatter_l0 = np.array([[2.44383, 0.029227], [0.0, 0.62568]])
    # shape (1, 2, 2, 1) — one moment, which is isotropic
    # Build explicitly with correct [mat, destination, source, moment] layout
    xs_scatter = np.array(
        [[[[2.44383], [0.029227]], [[0.0], [0.62568]]]]
    )  # shape (1, 2, 2, 1)
    print(xs_scatter.shape)
    chi = np.array([[0.0], [1.0]])
    nu = np.array([[2.5, 2.5]])
    sigmaf = np.array([[0.050632, 0.0010484]])
    xs_fission = np.array([chi @ (nu * sigmaf)])

    length = 7.566853 * 2
    delta_x = np.repeat(length / cells_x, cells_x)
    medium_map = np.zeros((cells_x), dtype=np.int32)
    flux, keff = power_iteration(
        xs_total,
        xs_scatter,
        xs_fission,
        medium_map,
        delta_x,
        angle_x,
        angle_w,
        bc_x,
    )
    print(f"Final Keff: {keff:.8f}")
    # print(xs_scatter[0, :, :, 0])
