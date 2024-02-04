
# Called for fixed source problems

import numpy as np

from discrete1 import multi_group as mg
from discrete1 import tools


def source_iteration(xs_total, xs_scatter, xs_fission, external, boundary, \
        medium_map, delta_x, angle_x, angle_w, bc_x, geometry=1, \
        angular=False, edges=0):

    # Initialize flux
    cells_x = medium_map.shape[0]
    groups = xs_total.shape[1]
    flux_old = np.zeros((cells_x, groups))

    # Combine scattering and fission
    xs_matrix = xs_scatter + xs_fission

    # Run source iteration for scalar flux centers
    flux = mg.source_iteration(flux_old, xs_total, xs_matrix, external, \
                               boundary, medium_map, delta_x, angle_x, \
                               angle_w, bc_x, geometry)

    if (angular is False) and (edges == 0):
        return flux

    # For angular flux or scalar flux edges
    return known_source_calculation(flux, xs_total, xs_matrix, external, \
                                boundary, medium_map, delta_x, angle_x, \
                                angle_w, bc_x, geometry, angular, edges)


def dynamic_mode_decomp(xs_total, xs_scatter, xs_fission, external, \
        boundary, medium_map, delta_x, angle_x, angle_w, bc_x, geometry=1, \
        angular=False, edges=0, R=2, K=10):

    # Initialize flux
    cells_x = medium_map.shape[0]
    groups = xs_total.shape[1]
    flux_old = np.zeros((cells_x, groups))

    # Combine scattering and fission
    xs_matrix = xs_scatter + xs_fission

    # Run dynamic mode decomposition for scalar flux centers
    flux = mg.dynamic_mode_decomp(flux_old, xs_total, xs_matrix, external, \
                               boundary, medium_map, delta_x, angle_x, \
                               angle_w, bc_x, geometry, R, K)

    if (angular is False) and (edges == 0):
        return flux

    # For angular flux or scalar flux edges
    return known_source_calculation(flux, xs_total, xs_matrix, external, \
                                boundary, medium_map, delta_x, angle_x, \
                                angle_w, bc_x, geometry, angular, edges)


# This is for calculating angular flux or edge flux
def known_source_calculation(flux, xs_total, xs_scatter, external, \
        boundary, medium_map, delta_x, angle_x, angle_w, bc_x, geometry, \
        angular, edges):

    cells_x, groups = flux.shape
    angles = angle_x.shape[0]

    # Create (sigma_s + sigma_f) * phi + external source
    source = np.zeros((cells_x, angles, groups))
    tools._source_total(source, flux, xs_scatter, medium_map, external)

    # Scalar Edges
    if (angular is False) and (edges == 1):
        return mg.known_source_scalar(xs_total, source, boundary, medium_map, \
                            delta_x, angle_x, angle_w, bc_x, geometry, edges)

    # Angular centers or edges
    return mg.known_source_angular(xs_total, source, boundary, medium_map, \
                        delta_x, angle_x, angle_w, bc_x, geometry, edges)