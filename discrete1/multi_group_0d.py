
import numpy as np

from discrete1 import tools

count_nn = 100
change_nn = 1e-12

count_gg = 100
change_gg = 1e-08

# count_pp = 50
# change_pp = 1e-08

def source_iteration(flux_old, xs_total, xs_scatter, external, angle_x, \
        angle_w):

    groups = flux_old.shape[0]
    flux = np.zeros((groups,))
    off_scatter = 0.0

    converged = False
    count = 1
    change = 0.0

    while not (converged):
        flux *= 0.0

        for gg in range(groups):
            # Check for sizes
            qq = 0 if external.shape[1] == 1 else gg

            # Update off scatter source
            off_scatter = tools._off_scatter_0d(flux, flux_old, xs_scatter, gg)

            # Run discrete ordinates for one group
            flux[gg] = discrete_ordinates(flux_old[gg], xs_total[gg], \
                                        xs_scatter[gg,gg], off_scatter, \
                                        external[:,qq], angle_x, angle_w)

        # Check for convergence
        try:
            change = np.linalg.norm((flux - flux_old) / flux)
        except RuntimeWarning:
            change = 0.0
        converged = (change < change_gg) or (count >= count_gg)
        count += 1

        flux_old = flux.copy()

    return flux


def discrete_ordinates(flux_old, xs_total, xs_scatter, off_scatter, \
        external, angle_x, angle_w):

    flux = 0.0
    angles = angle_x.shape[0]

    converged = False
    count = 1
    change = 0.0

    while not (converged):

        flux *= 0.0

        for nn in range(angles):

            qq = 0 if external.shape[0] == 1 else nn
            flux += (xs_scatter * flux_old + external[qq] + off_scatter) \
                    * angle_w[nn] / xs_total

        # Check for convergence
        try:
            change = np.linalg.norm((flux - flux_old) / flux)
        except RuntimeWarning:
            change = 0.0
        converged = (change < change_nn) or (count >= count_nn)
        count += 1

        flux_old = flux

    return flux