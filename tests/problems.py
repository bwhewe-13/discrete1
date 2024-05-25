
# Default Problems for Testing

import numpy as np

import discrete1
from discrete1.utils import manufactured as mms


def manufactured_ss_01(cells_x, angles):
    # General parameters
    groups = 1
    bc_x = [0, 0]

    # Spatial
    length_x = 1.
    delta_x = np.repeat(length_x / cells_x, cells_x)
    edges_x = np.linspace(0, length_x, cells_x+1)

    # Angular
    angle_x, angle_w = discrete1.angular_x(angles, bc_x)
    
    # Materials
    xs_total = np.array([[1.0]])
    xs_scatter = np.array([[[0.0]]])
    xs_fission = np.array([[[0.0]]])

    # Externals
    external = np.ones((cells_x, 1, 1))
    boundary = np.zeros((2, 1, 1))
    boundary[0] = 1.

    # Layout
    medium_map = np.zeros((cells_x), dtype=np.int32)
    
    return xs_total, xs_scatter, xs_fission, external, boundary, \
            medium_map, delta_x, angle_x, angle_w, edges_x, bc_x


def manufactured_ss_02(cells_x, angles):
    # General parameters
    groups = 1
    bc_x = [0, 0]

    # Spatial
    length_x = 1.
    delta_x = np.repeat(length_x / cells_x, cells_x)
    edges_x = np.linspace(0, length_x, cells_x+1)

    # Angular
    angle_x, angle_w = discrete1.angular_x(angles, bc_x)
    
    # Materials
    xs_total = np.array([[1.0]])
    xs_scatter = np.array([[[0.0]]])
    xs_fission = np.array([[[0.0]]])

    # Sources
    external = 0.5 * np.ones((cells_x, 1, 1))
    boundary = np.zeros((2, 1, 1))
    boundary[0] = 1.

    # Layout
    medium_map = np.zeros((cells_x), dtype=np.int32)
    
    return xs_total, xs_scatter, xs_fission, external, boundary, \
            medium_map, delta_x, angle_x, angle_w, edges_x, bc_x


def manufactured_ss_03(cells_x, angles):
    # General parameters
    groups = 1
    bc_x = [0, 0]

    # Spatial
    length_x = 1.
    delta_x = np.repeat(length_x / cells_x, cells_x)
    edges_x = np.linspace(0, length_x, cells_x+1)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

    # Angular
    angle_x, angle_w = discrete1.angular_x(angles, bc_x)

    # Materials
    xs_total = np.array([[1.0]])
    xs_scatter = np.array([[[0.9]]])
    xs_fission = np.array([[[0.0]]])

    # Sources
    external = discrete1.external1d.manufactured_ss_03(centers_x, angle_x)
    boundary = discrete1.boundary1d.manufactured_ss_03(angle_x)

    # Layout
    medium_map = np.zeros((cells_x), dtype=np.int32)

    return xs_total, xs_scatter, xs_fission, external, boundary, \
            medium_map, delta_x, angle_x, angle_w, edges_x, bc_x


def manufactured_ss_04(cells_x, angles):
    # General parameters
    groups = 1
    bc_x = [0, 0]

    # Spatial
    length_x = 2.
    delta_x = np.repeat(length_x / cells_x, cells_x)
    edges_x = np.linspace(0, length_x, cells_x+1)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

    # Angular
    angle_x, angle_w = discrete1.angular_x(angles, bc_x)

    # Materials
    xs_total = np.array([[1.0], [1.0]])
    xs_scatter = np.array([[[0.3]], [[0.9]]])
    xs_fission = np.array([[[0.0]], [[0.0]]])

    # Sources
    external = discrete1.external1d.manufactured_ss_04(centers_x, angle_x)
    boundary = discrete1.boundary1d.manufactured_ss_04()

    # Layout
    materials = [[0, "quasi", "0-1"], [1, "scatter", "1-2"]]
    medium_map = discrete1.spatial1d(materials, edges_x)

    return xs_total, xs_scatter, xs_fission, external, boundary, \
            medium_map, delta_x, angle_x, angle_w, edges_x, bc_x


def manufactured_ss_05(cells_x, angles):
    # General parameters
    groups = 1
    bc_x = [0, 0]
    
    # Spatial
    length_x = 2.
    delta_x = np.repeat(length_x / cells_x, cells_x)
    edges_x = np.linspace(0, length_x, cells_x+1)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

    # Angular
    angle_x, angle_w = discrete1.angular_x(angles, bc_x)
    
    # Materials
    xs_total = np.array([[1.0], [1.0]])
    xs_scatter = np.array([[[0.3]], [[0.9]]])
    xs_fission = np.array([[[0.0]], [[0.0]]])

    # Sources
    external = discrete1.external1d.manufactured_ss_05(centers_x, angle_x)
    boundary = discrete1.boundary1d.manufactured_ss_05()

    # Layout
    layout = [[0, "quasi", "0-1"], [1, "scatter", "1-2"]]
    medium_map = discrete1.spatial1d(layout, edges_x)

    return xs_total, xs_scatter, xs_fission, external, boundary, \
            medium_map, delta_x, angle_x, angle_w, edges_x, bc_x


def manufactured_td_01(cells_x, angles, edges_t, temporal=1):
    # General parameters
    groups = 1
    bc_x = [0, 0]

    # Spatial
    length_x = 2
    delta_x = np.repeat(length_x / cells_x, cells_x)
    edges_x = np.linspace(0, length_x, cells_x+1)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

    # Angular
    angle_x, angle_w = discrete1.angular_x(angles, bc_x)

    # Materials
    xs_total = np.array([[1.0]])
    xs_scatter = np.array([[[0.0]]])
    xs_fission = np.array([[[0.0]]])
    velocity = np.ones((groups,))

    # Sources
    # Backward Euler
    if temporal == 1:
        initial_flux = mms.solution_td_01(centers_x, angle_x, np.array([0.0]))[0]
        external = discrete1.external1d.manufactured_td_01(centers_x, angle_x, edges_t)[1:]
    # BDF2
    elif temporal == 3:
        initial_flux = mms.solution_td_01(centers_x, angle_x, np.array([0.0]))[0]
        external = discrete1.external1d.manufactured_td_01(centers_x, angle_x, edges_t)[1:]

    boundary = 2 * np.ones((1, 2, 1, 1))
    
    # Layout
    medium_map = np.zeros((cells_x), dtype=np.int32)

    return initial_flux, xs_total, xs_scatter, xs_fission, velocity, \
        external, boundary, medium_map, delta_x, angle_x, angle_w, bc_x


def manufactured_td_02(cells_x, angles, edges_t, temporal=1):
    # General parameters
    groups = 1
    bc_x = [0, 0]

    # Spatial
    length_x = np.pi
    delta_x = np.repeat(length_x / cells_x, cells_x)
    edges_x = np.linspace(0, length_x, cells_x+1)
    centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

    # Angular
    angle_x, angle_w = discrete1.angular_x(angles, bc_x)

    # Materials
    xs_total = np.array([[1.0]])
    xs_scatter = np.array([[[0.0]]])
    xs_fission = np.array([[[0.0]]])
    velocity = np.ones((groups,))

    # Sources
    # Backward Euler
    if temporal == 1:
        initial_flux = mms.solution_td_02(centers_x, angle_x, np.array([0.0]))[0]
        external = discrete1.external1d.manufactured_td_02(centers_x, angle_x, edges_t)[1:]
        boundary = discrete1.boundary1d.manufactured_td_02(angle_x, edges_t)[1:]
    # BDF2
    elif temporal == 3:
        initial_flux = mms.solution_td_02(centers_x, angle_x, np.array([0.0]))[0]
        external = discrete1.external1d.manufactured_td_02(centers_x, angle_x, edges_t)[1:]
        boundary = discrete1.boundary1d.manufactured_td_02(angle_x, edges_t)[1:]

    # Layout
    medium_map = np.zeros((cells_x), dtype=np.int32)

    return initial_flux, xs_total, xs_scatter, xs_fission, velocity, \
        external, boundary, medium_map, delta_x, angle_x, angle_w, bc_x
