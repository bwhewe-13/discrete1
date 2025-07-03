import numpy as np

from discrete1 import tools
from discrete1.spatial_sweep import discrete_ordinates, _known_scatter, _known_source

count_gg = 100
change_gg = 1e-08

count_pp = 50
change_pp = 1e-08


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


def variable_source_iteration(
    flux_old,
    xs_total,
    star_coef_c,
    xs_scatter,
    external,
    boundary,
    medium_map,
    delta_x,
    angle_x,
    angle_w,
    bc_x,
    delta_coarse,
    delta_fine,
    edges_gidx_c,
    geometry,
):

    cells_x, groups = flux_old.shape
    flux = np.zeros((cells_x, groups))
    off_scatter = np.zeros((cells_x,), dtype=np.float64)

    converged = False
    count = 1
    change = 0.0

    while not (converged):
        flux *= 0.0

        for gg in range(groups):
            # Check for sizes
            qq = 0 if external.shape[2] == 1 else gg
            bc = 0 if boundary.shape[2] == 1 else gg

            idx1 = edges_gidx_c[gg]
            idx2 = edges_gidx_c[gg + 1]

            xs_scatter_c = (
                np.sum(
                    xs_scatter[:, idx1:idx2, idx1:idx2] * delta_fine[idx1:idx2],
                    axis=(1, 2),
                )
                / delta_coarse[gg]
            )

            xs_total_c = (
                np.sum(xs_total[:, idx1:idx2] * delta_fine[idx1:idx2], axis=1)
                / delta_coarse[gg]
            )
            xs_total_c += star_coef_c[gg]

            # Update off scatter source
            tools._variable_off_scatter(
                flux / delta_coarse,
                flux_old / delta_coarse,
                medium_map,
                xs_scatter[:, edges_gidx_c[gg] : edges_gidx_c[gg + 1]] * delta_fine,
                off_scatter,
                gg,
                edges_gidx_c,
            )

            # Run discrete ordinates for one group
            flux[:, gg] = discrete_ordinates(
                flux_old[:, gg],
                xs_total_c,
                xs_scatter_c,
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


def dynamic_mode_decomp(
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
    R,
    K,
):

    cells_x, groups = flux_old.shape
    flux = np.zeros((cells_x, groups))
    off_scatter = np.zeros((cells_x,))

    # Initialize Y_plus and Y_minus
    y_plus = np.zeros((cells_x, groups, K - 1))
    y_minus = np.zeros((cells_x, groups, K - 1))

    converged = False
    change = 0.0

    for rk in range(R + K):

        # Return flux if there is convergence
        if converged:
            return flux

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
        change = np.linalg.norm((flux - flux_old) / flux / cells_x)
        converged = change < change_gg

        # Collect difference for DMD on K iterations
        if rk >= R:
            kk = rk - R
            if kk < (K - 1):
                y_minus[:, :, kk] = flux - flux_old
            if kk > 0:
                y_plus[:, :, kk - 1] = flux - flux_old

        flux_old = flux.copy()

    # Perform DMD
    flux = tools.dmd(flux, y_minus, y_plus, K)

    return flux


########################################################################
# Multigroup Known Source Problems
########################################################################


def known_source_angular(
    xs_total,
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

    cells_x, angles, groups = source.shape

    # Initialize scalar flux
    angular_flux = np.zeros((cells_x + edges, angles, groups))

    # Initialize dummy variable
    zero = np.zeros((cells_x + edges))

    for gg in range(groups):

        qq = 0 if source.shape[2] == 1 else gg
        bc = 0 if boundary.shape[2] == 1 else gg

        _known_source(
            angular_flux[:, :, gg],
            xs_total[:, gg],
            zero,
            source[:, :, qq],
            boundary[:, :, bc],
            medium_map,
            delta_x,
            angle_x,
            angle_w,
            bc_x,
            geometry,
            edges,
        )

    return angular_flux


def known_source_scalar(
    xs_total,
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

    cells_x, angles, groups = source.shape

    # Initialize scalar flux
    scalar_flux = np.zeros((cells_x + edges, groups, 1))

    # Initialize dummy variable
    # zero = 1e-15 * np.ones((cells_x + edges))
    zero = np.zeros((cells_x + edges))

    for gg in range(groups):

        qq = 0 if source.shape[2] == 1 else gg
        bc = 0 if boundary.shape[2] == 1 else gg

        _known_source(
            scalar_flux[:, gg],
            xs_total[:, gg],
            zero,
            source[:, :, qq],
            boundary[:, :, bc],
            medium_map,
            delta_x,
            angle_x,
            angle_w,
            bc_x,
            geometry,
            edges,
        )

    return scalar_flux[:, :, 0]


########################################################################
# Multigroup DJINN Problems
########################################################################


def source_iteration_collect(
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
    iteration,
    filepath,
):

    cells_x, groups = flux_old.shape
    flux = np.zeros((cells_x, groups))
    off_scatter = np.zeros((cells_x,))
    tracked_flux = np.zeros((count_gg, cells_x, groups))

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
        change = np.linalg.norm((flux - flux_old) / flux / cells_x)
        converged = (change < change_gg) or (count >= count_gg)
        count += 1

        # Update old flux and tracked flux
        flux_old = flux.copy()
        tracked_flux[count - 2] = flux.copy()

    fiteration = str(iteration).zfill(3)
    np.save(filepath + f"flux_scatter_model_{fiteration}", tracked_flux[: count - 1])

    return flux


def source_iteration_djinn(
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
    scatter_models=[],
    scatter_labels=None,
):

    cells_x, groups = flux_old.shape
    flux = np.zeros((cells_x, groups))
    scatter_source = np.zeros((cells_x, groups))

    converged = False
    count = 1
    change = 0.0

    while not (converged):
        flux *= 0.0

        tools._djinn_scatter_predict(
            flux_old,
            xs_scatter,
            scatter_source,
            medium_map,
            scatter_models,
            scatter_labels,
        )

        for gg in range(groups):
            # Check for sizes
            qq = 0 if external.shape[2] == 1 else gg
            bc = 0 if boundary.shape[2] == 1 else gg

            # Run discrete ordinates for one group
            flux[:, gg] = _known_scatter(
                xs_total[:, gg],
                scatter_source[:, gg],
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
        change = np.linalg.norm((flux - flux_old) / flux / cells_x)
        converged = (change < change_pp) or (count >= count_pp)
        count += 1

        # Update old flux and tracked flux
        flux_old = flux.copy()

    return flux
