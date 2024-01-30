
import numpy as np
import numba

from discrete1 import tools

count_gg = 100
count_nn = 100

change_gg = 1e-08
change_nn = 1e-12


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


def discrete_ordinates(flux_old, xs_total, xs_scatter, off_scatter, external, \
        boundary, medium_map, delta_x, angle_x, angle_w, bc_x, geometry):

    # Slab geometry
    if geometry == 1:
        return slab_ordinates(flux_old, xs_total, xs_scatter, off_scatter, \
                              external, boundary, medium_map, delta_x, \
                              angle_x, angle_w, bc_x)

    elif geometry == 2:
        return sphere_ordinates(flux_old, xs_total, xs_scatter, off_scatter, \
                                external, boundary, medium_map, delta_x, \
                                angle_x, angle_w, bc_x)


def slab_ordinates(flux_old, xs_total, xs_scatter, off_scatter, external, \
        boundary, medium_map, delta_x, angle_x, angle_w, bc_x):

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
            bc = 0 if external.shape[1] == 1 else nn

            if angle_x[nn] > 0.0:
                edge1 = reflector[nn] + boundary[0,bc]
                edge1 = slab_forward(flux, flux_old, xs_total, xs_scatter, \
                                off_scatter, external[:,qq], edge1, medium_map, \
                                delta_x, angle_x[nn], angle_w[nn])

            elif angle_x[nn] < 0.0:
                edge1 = reflector[nn] + boundary[1,bc]
                edge1 = slab_backward(flux, flux_old, xs_total, xs_scatter, \
                                off_scatter, external[:,qq], edge1, medium_map, \
                                delta_x, angle_x[nn], angle_w[nn])
            else:
                raise Exception("Discontinuity at 0")

            tools.reflector_corrector(reflector, angle_x, edge1, nn, bc_x)

        # Check for convergence
        change = np.linalg.norm((flux - flux_old) / flux / cells_x)
        converged = (change < change_nn) or (count >= count_nn)
        count += 1

        flux_old = flux.copy()

    return flux


@numba.jit("f8(f8[:], f8[:], f8[:], f8[:], f8[:], f8[:], f8, i4[:], \
            f8[:], f8, f8)", nopython=True, cache=True)
def slab_forward(flux, flux_old, xs_total, xs_scatter, off_scatter, \
        external, edge1, medium_map, delta_x, angle_x, angle_w):
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
        edge2 = (xs_scatter[mat] * flux_old[ii] + external[ii] + off_scatter[ii] \
                + edge1 * (angle_x / delta_x[ii] - 0.5 * xs_total[mat])) \
                * 1 / (angle_x / delta_x[ii] + 0.5 * xs_total[mat])
        # Update flux with cell centers
        flux[ii] += 0.5 * angle_w * (edge1 + edge2)
        # Update unknown cell edge
        edge1 = edge2
    # Return cell at i = I
    return edge1


@numba.jit("f8(f8[:], f8[:], f8[:], f8[:], f8[:], f8[:], f8, i4[:], \
            f8[:], f8, f8)", nopython=True, cache=True)
def slab_backward(flux, flux_old, xs_total, xs_scatter, off_scatter, \
        external, edge1, medium_map, delta_x, angle_x, angle_w):
    # Get iterables
    cells_x = numba.int32(flux.shape[0])
    mat = numba.int32
    ii = numba.int32
    # Initialize unknown cell edge
    edge2 = numba.float64(0.0)
    # Iterate over cells
    for ii in range(cells_x-1, -1, -1):
        # Determining material cross section
        mat = medium_map[ii]
        # Calculate cell edge unknown
        edge2 = (xs_scatter[mat] * flux_old[ii] + external[ii] + off_scatter[ii] \
                + edge1 * (-angle_x / delta_x[ii] - 0.5 * xs_total[mat])) \
                * 1 / (-angle_x / delta_x[ii] + 0.5 * xs_total[mat])
        # Update flux with cell centers
        flux[ii] += 0.5 * angle_w * (edge1 + edge2)
        # Update unknown cell edge
        edge1 = edge2
    # Return cell at i = 0
    return edge1


def sphere_ordinates(flux_old, xs_total, xs_scatter, off_scatter, external, \
        boundary, medium_map, delta_x, angle_x, angle_w, bc_x):

    cells_x = flux_old.shape[0]
    angles = angle_x.shape[0]

    flux = np.zeros((cells_x,))
    half_angle = np.zeros((cells_x,))
    edge1 = 0.0

    converged = False
    count = 1
    change = 0.0

    while not (converged):

        angle_minus = -1.0
        alpha_minus = 0.0

        flux *= 0.0
        # Calculate the initial half angle
        initialize_half_angle(flux_old, half_angle, xs_total, xs_scatter, \
                              off_scatter, external[:,0], medium_map, \
                              delta_x, boundary[1,0])

        # Iterate over the discrete ordinates
        for nn in range(angles):

            qq = 0 if external.shape[1] == 1 else nn
            bc = 0 if external.shape[1] == 1 else nn

            # Calculate the half angle coefficient
            angle_plus = angle_minus + 2 * angle_w[nn]
            # Calculate the weighted diamond
            tau = (angle_x[nn] - angle_minus) / (angle_plus - angle_minus)
            # Calculate the angular differencing coefficient
            alpha_plus = angle_coef_corrector(alpha_minus, angle_x[nn], \
                                              angle_w[nn], nn, angles)

            # Iterate from 0 -> I
            if angle_x[nn] > 0.0:
                sphere_forward(flux, flux_old, half_angle, xs_total, \
                        xs_scatter, off_scatter, external[:,qq], \
                        medium_map, delta_x, angle_x[nn], angle_w[nn], \
                        angle_w[nn], tau, alpha_plus, alpha_minus)

            # Iterate from I -> 0
            elif angle_x[nn] < 0.0:
                sphere_backward(flux, flux_old, half_angle, xs_total, \
                        xs_scatter, off_scatter, external[:,qq], \
                        boundary[1,bc], medium_map, delta_x, angle_x[nn], \
                        angle_w[nn], angle_w[nn], tau, alpha_plus, alpha_minus)
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


@numba.jit("void(f8[:], f8[:], f8[:], f8[:], f8[:], f8[:], i4[:], f8[:], \
            f8)", nopython=True, cache=True)
def initialize_half_angle(flux, half_angle, xs_total, xs_scatter, \
        off_scatter, external, medium_map, delta_x, angle_plus):
    # Get iterables
    cells_x = numba.int32(flux.shape[0])
    mat = numba.int32
    ii = numba.int32
    # Zero out half angle
    half_angle *= 0.0
    # Iterate from sphere surface to center
    for ii in range(cells_x-1, -1, -1):
        mat = medium_map[ii]
        # Calculate angular flux half angle
        half_angle[ii] = (2 * angle_plus + delta_x[ii] * (external[ii] \
                        + off_scatter[ii] + xs_scatter[mat] * flux[ii])) \
                        / (2 + xs_total[mat] * delta_x[ii])
        # Update half angle coefficient
        angle_plus = 2 * half_angle[ii] - angle_plus


@numba.jit("f8(f8, f8, f8, i4, i4)", nopython=True, cache=True)
def angle_coef_corrector(alpha_minus, angle_x, angle_w, nn, angles):
    if nn != angles - 1:
        return alpha_minus - angle_x * angle_w
    return 0.0


@numba.jit("void(f8[:], f8[:], f8[:], f8[:], f8[:], f8[:], f8[:], i4[:], \
            f8[:], f8, f8, f8, f8, f8, f8)", nopython=True, cache=True)
def sphere_forward(flux, flux_old, half_angle, xs_total, xs_scatter, \
        off_scatter, external, medium_map, delta_x, angle_x, angle_w, \
        weight, tau, alpha_plus, alpha_minus):
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
        area1 = 4 * np.pi * (ii * delta_x[ii])**2
        area2 = 4 * np.pi * ((ii + 1) * delta_x[ii])**2
        # Calculate volume of cell
        volume = 4/3. * np.pi * (((ii + 1) * delta_x[ii])**3 - (ii * delta_x[ii])**3)

        # Calculate flux at cell center
        center = (angle_x * (area2 + area1) * edge1 + 1 / angle_w * (area2 - area1) \
                * (alpha_plus + alpha_minus) * (half_angle[ii]) + volume \
                * (external[ii] + off_scatter[ii] + flux_old[ii] * xs_scatter[mat])) \
                / (2 * angle_x * area2 + 2 / angle_w * (area2 - area1) * alpha_plus \
                + xs_total[mat] * volume)

        flux[ii] += weight * center
        edge1 = 2 * center - edge1

        # Update half angle coefficient
        if ii != 0:
            half_angle[ii] = 1 / tau * (center - (1 - tau) * half_angle[ii])


@numba.jit("void(f8[:], f8[:], f8[:], f8[:], f8[:], f8[:], f8[:], f8, \
        i4[:], f8[:], f8, f8, f8, f8, f8, f8)", nopython=True, cache=True)
def sphere_backward(flux, flux_old, half_angle, xs_total, xs_scatter, \
        off_scatter, external, boundary, medium_map, delta_x, angle_x, \
        angle_w, weight, tau, alpha_plus, alpha_minus):
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
    for ii in range(cells_x-1, -1, -1):
        # For determining the material cross sections
        mat = medium_map[ii]

        # Calculate surface areas
        area1 = 4 * np.pi * (ii * delta_x[ii])**2
        area2 = 4 * np.pi * ((ii + 1) * delta_x[ii])**2
        # Calculate volume of cell
        volume = 4/3. * np.pi * (((ii + 1) * delta_x[ii])**3 - (ii * delta_x[ii])**3)

        # Calculate the flux at the cell center
        center = (-angle_x * (area2 + area1) * edge1 + 1 / angle_w * (area2 - area1) \
                * (alpha_plus + alpha_minus) * (half_angle[ii]) + volume \
                * (external[ii] + off_scatter[ii] + flux_old[ii] * xs_scatter[mat])) \
                / (2 * -angle_x * area1 + 2 / angle_w * (area2 - area1) * alpha_plus \
                + xs_total[mat] * volume)

        flux[ii] += weight * center
        edge1 = 2 * center - edge1

        # Update half angle coefficient
        if ii != 0:
            half_angle[ii] = 1 / tau * (center - (1 - tau) * half_angle[ii])
