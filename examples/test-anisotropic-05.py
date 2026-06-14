"""Anisotropic slab S_N with rank-1 outer-product Legendre moment updates."""

import numpy as np

import discrete1
from discrete1 import tools

count_kk = 100
change_kk = 1e-06

count_nn = 100
change_nn = 1e-12

count_gg = 100
change_gg = 1e-08


def legendre_all(n_moments, angle_x):
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
    for ll in range(2, n_moments):
        P[ll] = ((2 * ll - 1) * angle_x * P[ll - 1] - (ll - 1) * P[ll - 2]) / ll
    return P


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
    xs_total : numpy.ndarray, shape (n_materials, n_groups)
        Total cross sections.
    xs_scatter : numpy.ndarray, shape (n_materials, n_groups, n_groups, L+1)
        Scatter cross sections; last axis holds Legendre moments.
    xs_fission : numpy.ndarray, shape (n_materials, n_groups)
        Fission cross sections (nu*sigma_f or sigma_f when chi is provided).
    medium_map : numpy.ndarray, shape (cells_x,)
        Material index per spatial cell.
    delta_x : numpy.ndarray, shape (cells_x,)
        Cell widths.
    angle_x : numpy.ndarray, shape (angles,)
        Angular quadrature points.
    angle_w : numpy.ndarray, shape (angles,)
        Angular quadrature weights.
    bc_x : list-like
        Boundary condition indicators [left, right].
    chi : numpy.ndarray, optional
        Fission neutron spectrum. Required when xs_fission is sigma_f.
    geometry : int, optional
        Geometry selector (1=slab, 2=sphere).

    Returns
    -------
    flux : numpy.ndarray, shape (cells_x, n_groups)
        Converged scalar flux.
    keff : float
        Converged multiplication factor.
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

    # Precompute Legendre polynomials and weighted products once per solve
    n_moments = xs_scatter.shape[3]
    P = legendre_all(n_moments, angle_x)  # (L+1, angles)
    wP = angle_w[np.newaxis, :] * P  # (L+1, angles): w_n * P_l(mu_n)

    converged = False
    count = 0
    change = 0.0

    while not converged:
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
            P,
            wP,
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
    P,
    wP,
    edges=0,
):
    """Multigroup source iteration solver.

    Performs outer iterations over energy groups using inner discrete
    ordinates sweeps for each group. Off-diagonal scatter (up- and
    down-scatter) contributions are handled via an "off_scatter"
    accumulator that uses the previous and current flux iterates.

    Parameters
    ----------
    flux_old : numpy.ndarray, shape (n_cells, n_groups)
        Initial scalar flux guess.
    xs_total : numpy.ndarray, shape (n_materials, n_groups)
        Total cross sections.
    xs_scatter : numpy.ndarray, shape (n_materials, n_groups, n_groups, L+1)
        Scatter cross sections; last axis holds Legendre moments.
    external : numpy.ndarray, shape (n_cells, n_angles or 1, n_groups or 1)
        External source array.
    boundary : numpy.ndarray, shape (2, n_angles or 1, n_groups or 1)
        Boundary conditions array.
    medium_map : numpy.ndarray, shape (n_cells,)
        Material index per spatial cell.
    delta_x : numpy.ndarray, shape (n_cells,)
        Cell widths.
    angle_x : numpy.ndarray, shape (angles,)
        Angular quadrature points.
    angle_w : numpy.ndarray, shape (angles,)
        Angular quadrature weights.
    bc_x : sequence
        Boundary condition flags for left/right boundaries.
    P : numpy.ndarray, shape (L+1, angles)
        Precomputed Legendre polynomials P[l, n] = P_l(mu_n).
    wP : numpy.ndarray, shape (L+1, angles)
        Precomputed w_n * P_l(mu_n).
    edges : int, optional
        If 1, compute edge flux; otherwise compute cell-centre flux.

    Returns
    -------
    numpy.ndarray, shape (n_cells, n_groups)
        Converged scalar flux.
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

            # Update off-diagonal scatter source using L=0 moment only
            tools._off_scatter(
                flux, flux_old, medium_map, xs_scatter[..., 0], off_scatter, gg
            )

            # Map within-group Legendre moments to cells -> (cells_x, L+1)
            xs_scatter_cell = xs_scatter[:, gg, gg, :][medium_map]

            # Precompute (2l+1) * xs_l(x) once per group per outer iteration;
            # this product is fixed throughout the inner sweep loop.
            weights = 2 * np.arange(xs_scatter_cell.shape[1]) + 1
            xs_w = xs_scatter_cell * weights[np.newaxis, :]  # (cells_x, L+1)

            # Run discrete ordinates for one group
            flux[:, gg] = discrete_ordinates(
                flux_old[:, gg],
                xs_total[:, gg],
                xs_scatter_cell,
                xs_w,
                off_scatter,
                external[:, :, qq],
                boundary[:, :, bc],
                medium_map,
                delta_x,
                angle_x,
                angle_w,
                bc_x,
                P,
                wP,
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


def discrete_ordinates(
    flux_old,
    xs_total,
    xs_scatter_cell,
    xs_w,
    off_scatter,
    external,
    boundary,
    medium_map,
    delta_x,
    angle_x,
    angle_w,
    bc_x,
    P,
    wP,
    edges,
):
    """Perform a slab-geometry discrete-ordinates spatial sweep.

    Iterates over angular ordinates, performing forward and backward
    diamond-difference passes. The anisotropic scatter source is rebuilt
    from angular flux moments accumulated during each sweep. Moments are
    updated via a single rank-1 outer-product per ordinate after each
    sweep call, avoiding both full angular flux storage and per-cell
    NumPy dispatch overhead inside the cell loop.

    Parameters
    ----------
    flux_old : numpy.ndarray, shape (n_cells,)
        Initial scalar flux guess for this group.
    xs_total : numpy.ndarray, shape (n_materials,)
        Total cross sections per material.
    xs_scatter_cell : numpy.ndarray, shape (cells_x, L+1)
        Within-group Legendre scatter moments mapped to cells.
    xs_w : numpy.ndarray, shape (cells_x, L+1)
        Precomputed (2l+1) * xs_scatter_cell; fixed for the inner loop.
    off_scatter : numpy.ndarray, shape (n_cells,)
        Precomputed off-diagonal scatter source.
    external : numpy.ndarray, shape (n_cells, n_angles or 1)
        External source.
    boundary : numpy.ndarray, shape (2, n_angles or 1)
        Boundary conditions.
    medium_map : numpy.ndarray, shape (n_cells,)
        Material index per cell.
    delta_x : numpy.ndarray, shape (n_cells,)
        Cell widths.
    angle_x : numpy.ndarray, shape (angles,)
        Angular quadrature points.
    angle_w : numpy.ndarray, shape (angles,)
        Angular quadrature weights.
    bc_x : sequence
        Boundary condition flags for left/right.
    P : numpy.ndarray, shape (L+1, angles)
        Precomputed P[l, n] = P_l(mu_n).
    wP : numpy.ndarray, shape (L+1, angles)
        Precomputed w_n * P_l(mu_n).
    edges : int
        If 1, compute edge flux contributions; otherwise cell-centre.

    Returns
    -------
    numpy.ndarray, shape (n_cells,)
        Converged scalar flux at cell centres.
    """
    cells_x = flux_old.shape[0]
    angles = angle_x.shape[0]
    n_moments = xs_scatter_cell.shape[1]

    flux = np.zeros((cells_x,))
    reflector = np.zeros((angles,))
    edge1 = 0.0

    # flux_moments[x, l] = sum_n w_n * P_l(mu_n) * psi(x, mu_n)
    # Initialised to zero; first-iteration aniso_source is therefore zero,
    # which is the correct seed before any angular flux has been computed.
    flux_moments = np.zeros((cells_x, n_moments))

    converged = False
    count = 1
    change = 0.0

    while not converged:

        flux *= 0.0
        reflector *= 0.0

        # Build anisotropic scatter source from moments of previous sweep.
        # S(x, mu_n) = sum_l (2l+1) * xs_l(x) * P_l(mu_n) * phi_l(x)
        # xs_w = (2l+1) * xs_scatter_cell is precomputed and fixed.
        aniso_source = np.einsum("xl,xl,ln->xn", xs_w, flux_moments, P)
        # print(aniso_source.shape)
        # shape (cells_x, angles)

        # Reset moment accumulator for this sweep
        flux_moments *= 0.0

        for nn in range(angles):

            qq = 0 if external.shape[1] == 1 else nn
            bc = 0 if boundary.shape[1] == 1 else nn

            if angle_x[nn] > 0.0:
                edge1 = reflector[nn] + boundary[0, bc]
                edge1, psi_nn = slab_forward(
                    flux,
                    xs_total,
                    aniso_source[:, nn],
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
                    aniso_source[:, nn],
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

            # Accumulate flux moments via a single rank-1 outer-product update.
            # flux_moments[x, l] += w_n * P_l(mu_n) * psi(x, mu_n)
            # np.outer(psi_nn, wP[:, nn]) is one BLAS call, replacing
            # cells_x individual NumPy dispatches inside the cell loop.
            flux_moments += np.outer(psi_nn, wP[:, nn])

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
    """Forward sweep for slab geometry for one ordinate.

    Marches from the left boundary toward the right, updating the scalar
    flux in-place and returning the cell-centre angular flux so that the
    caller can accumulate flux moments with a single outer-product update.
    The cell loop contains only scalar Python arithmetic — no NumPy calls.

    Parameters
    ----------
    flux : numpy.ndarray, shape (cells_x,)
        Scalar flux accumulator (updated in-place).
    xs_total : numpy.ndarray, shape (n_materials,)
        Total cross sections per material.
    aniso_source : numpy.ndarray, shape (cells_x,)
        Anisotropic scatter source for this ordinate.
    off_scatter : numpy.ndarray, shape (cells_x,)
        Off-diagonal scatter source.
    external : numpy.ndarray, shape (cells_x,)
        External source.
    edge1 : float
        Incoming edge flux (left boundary).
    medium_map : numpy.ndarray, shape (cells_x,)
        Material index per cell.
    delta_x : numpy.ndarray, shape (cells_x,)
        Cell widths.
    angle_x : float
        Angular ordinate (mu > 0).
    angle_w : float
        Quadrature weight.
    edges : int
        If 1, accumulate edge fluxes; else accumulate cell-centre fluxes.

    Returns
    -------
    edge1 : float
        Outgoing edge flux at i = I (right boundary).
    psi : numpy.ndarray, shape (cells_x,)
        Cell-centre angular flux for this ordinate.
    """
    cells_x = medium_map.shape[0]
    edge2 = 0.0
    psi = np.empty((cells_x,))

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
        psi[ii] = 0.5 * (edge1 + edge2)
        if edges == 1:
            flux[ii + 1] += angle_w * edge2
        else:
            flux[ii] += angle_w * psi[ii]
        edge1 = edge2

    return edge1, psi


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
    """Backward sweep for slab geometry for one ordinate.

    Marches from the right boundary toward the left, updating the scalar
    flux in-place and returning the cell-centre angular flux so that the
    caller can accumulate flux moments with a single outer-product update.
    The cell loop contains only scalar Python arithmetic — no NumPy calls.

    Parameters
    ----------
    flux : numpy.ndarray, shape (cells_x,)
        Scalar flux accumulator (updated in-place).
    xs_total : numpy.ndarray, shape (n_materials,)
        Total cross sections per material.
    aniso_source : numpy.ndarray, shape (cells_x,)
        Anisotropic scatter source for this ordinate.
    off_scatter : numpy.ndarray, shape (cells_x,)
        Off-diagonal scatter source.
    external : numpy.ndarray, shape (cells_x,)
        External source.
    edge1 : float
        Incoming edge flux (right boundary).
    medium_map : numpy.ndarray, shape (cells_x,)
        Material index per cell.
    delta_x : numpy.ndarray, shape (cells_x,)
        Cell widths.
    angle_x : float
        Angular ordinate (mu < 0).
    angle_w : float
        Quadrature weight.
    edges : int
        If 1, accumulate edge fluxes; else accumulate cell-centre fluxes.

    Returns
    -------
    edge1 : float
        Outgoing edge flux at i = 0 (left boundary).
    psi : numpy.ndarray, shape (cells_x,)
        Cell-centre angular flux for this ordinate.
    """
    cells_x = medium_map.shape[0]
    edge2 = 0.0
    psi = np.empty((cells_x,))

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


if __name__ == "__main__":
    cells_x = 250
    angles = 8
    bc_x = [0, 0]
    angle_x, angle_w = discrete1.angular_x(angles, bc_x)
    xs_total = np.array([[1.0]])  # (n_materials=1, n_groups=1)
    xs_fission = np.array([[[2.5 * 0.266667]]])  # (n_materials=1, n_groups=1)

    # xs_scatter shape: (n_materials, n_groups, n_groups, L+1)
    # Last axis holds Legendre moments [sigma_s0, sigma_s1, sigma_s2]
    xs_scatter = np.array([[[[0.733333, 0.2, 0.075]]]])  # (1, 1, 1, 3)

    medium_map = np.zeros((cells_x,), dtype=np.int32)

    # Critical half-length for PUa-1-2-SL from Sood et al. (1999) Table 25
    length = 0.76378 * 2
    delta_x = np.repeat(length / cells_x, cells_x)

    flux, keff = power_iteration(
        xs_total, xs_scatter, xs_fission, medium_map, delta_x, angle_x, angle_w, bc_x
    )
    print(f"Final Keff: {keff:.8f}")
