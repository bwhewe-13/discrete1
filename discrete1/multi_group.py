
import numpy as np

from discrete1 import tools
from discrete1.spatial_sweep import discrete_ordinates, sn_known_scatter


count_gg = 100
change_gg = 1e-08


def source_iteration(flux_old, xs_total, xs_scatter, external, boundary, \
        medium_map, delta_x, angle_x, angle_w, bc_x, geometry):

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
            tools._off_scatter(flux, flux_old, medium_map, xs_scatter, \
                                off_scatter, gg)

            # Run discrete ordinates for one group
            flux[:,gg] = discrete_ordinates(flux_old[:,gg], xs_total[:,gg], \
                                        xs_scatter[:,gg,gg], off_scatter, \
                                        external[:,:,qq], boundary[:,:,bc], \
                                        medium_map, delta_x, angle_x, \
                                        angle_w, bc_x, geometry)

        # Check for convergence
        change = np.linalg.norm((flux - flux_old) / flux / cells_x)
        converged = (change < change_gg) or (count >= count_gg)
        count += 1

        flux_old = flux.copy()

    return flux


def source_iteration_collect(flux_old, xs_total, xs_scatter, external, \
        boundary, medium_map, delta_x, angle_x, angle_w, bc_x, geometry, \
        iteration, filepath):

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
            tools._off_scatter(flux, flux_old, medium_map, xs_scatter, \
                                off_scatter, gg)

            # Run discrete ordinates for one group
            flux[:,gg] = discrete_ordinates(flux_old[:,gg], xs_total[:,gg], \
                                        xs_scatter[:,gg,gg], off_scatter, \
                                        external[:,:,qq], boundary[:,:,bc], \
                                        medium_map, delta_x, angle_x, \
                                        angle_w, bc_x, geometry)

        # Check for convergence
        change = np.linalg.norm((flux - flux_old) / flux / cells_x)
        converged = (change < change_gg) or (count >= count_gg)
        count += 1

        # Update old flux and tracked flux
        flux_old = flux.copy()
        tracked_flux[count-2] = flux.copy()

    fiteration = str(iteration).zfill(3)
    np.save(filepath + f"flux_scatter_model_{fiteration}", tracked_flux[:count-1])

    return flux


def source_iteration_djinn(flux_old, xs_total, xs_scatter, external, \
        boundary, medium_map, delta_x, angle_x, angle_w, bc_x, geometry, \
        scatter_models=[], scatter_map=[], scatter_labels=None):

    cells_x, groups = flux_old.shape
    flux = np.zeros((cells_x, groups))
    scatter_source = np.zeros((cells_x, groups))

    converged = False
    count = 1
    change = 0.0

    while not (converged):
        flux *= 0.0

        tools._djinn_source_predict(flux_old, xs_scatter, scatter_source, \
                        scatter_models, scatter_map, scatter_labels)
        tools._djinn_scatter_pass(flux_old, xs_scatter, scatter_source, \
                        medium_map, scatter_map)

        for gg in range(groups):
            # Check for sizes
            qq = 0 if external.shape[2] == 1 else gg
            bc = 0 if boundary.shape[2] == 1 else gg

            # Run discrete ordinates for one group
            flux[:,gg] = sn_known_scatter(xs_total[:,gg], scatter_source[:,gg], \
                                external[:,:,qq], boundary[:,:,bc], medium_map, \
                                delta_x, angle_x, angle_w, bc_x, geometry)

        # Check for convergence
        change = np.linalg.norm((flux - flux_old) / flux / cells_x)
        converged = (change < change_gg) or (count >= count_gg)
        count += 1

        # Update old flux and tracked flux
        flux_old = flux.copy()

    return flux
