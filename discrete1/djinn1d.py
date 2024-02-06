
# For running DJINN Problems

import numpy as np

from discrete1 import multi_group as mg
from discrete1 import tools

count_kk = 200
change_kk = 1e-06


def collection(xs_total, xs_scatter, xs_fission, medium_map, delta_x, \
        angle_x, angle_w, bc_x, filepath, geometry=1):

    # Set boundary source
    boundary = np.zeros((2, 1, 1))

    # Initialize and normalize flux
    cells_x = medium_map.shape[0]
    groups = xs_total.shape[1]

    flux_old = np.random.rand(cells_x, groups)
    keff = np.linalg.norm(flux_old)
    flux_old /= np.linalg.norm(keff)

    # Initialize power source
    source = np.zeros((cells_x, 1, groups))
    tracked_flux = np.zeros((count_kk, cells_x, groups))

    converged = False
    count = 0
    change = 0.0

    while not (converged):
        # Update power source term
        tools._fission_source(flux_old, xs_fission, source, medium_map, keff)

        # Solve for scalar flux
        flux = mg.source_iteration_collect(flux_old, xs_total, xs_scatter, \
                            source, boundary, medium_map, delta_x, angle_x, \
                            angle_w, bc_x, geometry, count, filepath)

        # Update keffective
        keff = tools._update_keffective(flux, flux_old, xs_fission, \
                                        medium_map, keff)

        # Normalize flux
        flux /= np.linalg.norm(flux)

        # Check for convergence
        change = np.linalg.norm((flux - flux_old) / flux / cells_x)
        print(f"Count: {count:>2}\tKeff: {keff:.8f}", end="\r")
        converged = (change < change_kk) or (count >= count_kk)
        count += 1

        # Update old flux and tracked flux
        flux_old = flux.copy()
        tracked_flux[count-2] = flux.copy()

    print(f"\nConvergence: {change:2.6e}")
    np.save(filepath + "flux_fission_model", tracked_flux[:count-1])

    # Save relevant information to file
    np.save(filepath + "fission_cross_sections", xs_fission)
    np.save(filepath + "scatter_cross_sections", xs_scatter)
    np.save(filepath + "medium_map", medium_map)

    return flux, keff

# xs_scatter and xs_fission are tuples and not lists
def power_iteration(flux_old, xs_total, xs_scatter, xs_fission, \
        medium_map, delta_x, angle_x, angle_w, bc_x, geometry, \
        fission_models=[], scatter_models=[], fission_map=[], \
        scatter_map=[], fission_labels=None, scatter_labels=None):

    # Set boundary source
    boundary = np.zeros((2, 1, 1))

    # Initialize keff
    cells_x = medium_map.shape[0]
    keff = 0.95

    # Initialize power source
    fission_source = np.zeros((cells_x, 1, xs_total.shape[1]))

    converged = False
    count = 0
    change = 0.0

    while not (converged):
        # Update power source term
        tools._djinn_source_predict(flux_old, xs_fission, fission_source, \
                        fission_models, fission_map, fission_labels, keff)
        tools._djinn_fission_pass(flux_old, xs_fission, fission_source, \
                            medium_map, keff, fission_map)

        # Solve for scalar flux
        # No DJINN predictions
        if len(scatter_models) == 0:
            flux = mg.source_iteration(flux_old, xs_total, xs_scatter, \
                                    fission_source, boundary, medium_map, \
                                    delta_x, angle_x, angle_w, bc_x, geometry)
        # DJINN predictions
        else:
            flux = mg.source_iteration_djinn(flux_old, xs_total, xs_scatter, \
                            fission_source, boundary, medium_map, delta_x, \
                            angle_x, angle_w, bc_x, geometry, scatter_models, \
                            scatter_map, scatter_labels)

        # Update keffective
        keff = tools._update_keffective(flux, flux_old, xs_fission, \
                                        medium_map, keff)

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