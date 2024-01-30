
import numpy as np

from discrete1 import multi_group as mg
from discrete1 import tools


count_kk = 100
change_kk = 1e-06


def power_iteration(xs_total, xs_scatter, xs_fission, medium_map, delta_x, \
        angle_x, angle_w, bc_x, geometry=1):

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
        tools._fission_source(flux_old, xs_fission, source, medium_map, keff)

        # Solve for scalar flux
        flux = mg.source_iteration(flux_old, xs_total, xs_scatter, source, \
                                   boundary, medium_map, delta_x, angle_x, \
                                   angle_w, bc_x, geometry)

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