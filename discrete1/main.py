# Running basic functions for collecting parameters

import numpy as np
import pkg_resources

from discrete1.utils.hybrid import energy_coarse_index

DATA_PATH = pkg_resources.resource_filename("discrete1","sources/energy/")


def angular_x(angles, bc_x=[0, 0]):
    angle_x, angle_w = np.polynomial.legendre.leggauss(angles)
    angle_w /= np.sum(angle_w)
    # Ordering for reflective boundaries
    if np.sum(bc_x) > 0.0:
        if bc_x == [1, 0]:
            idx = angle_x.argsort()
        elif bc_x == [0, 1]:
            idx = angle_x.argsort()[::-1]
        angle_x = angle_x[idx].copy()
        angle_w = angle_w[idx].copy()
    return angle_x, angle_w


def energy_grid(grid, groups_fine, groups_coarse=None):
    """
    Calculate energy grid bounds (MeV) and index for coarsening
    Arguments:
        groups (int): Number of energy groups for problem
        grid (int): specified energy grid to use (87, 361, 618)
    Returns:
        edges_g (float [grid + 1]): MeV energy group bounds
        edges_gidx (int [groups + 1]): Location of grid index for problem
    """
    # Create energy grid
    if grid in [87, 361, 618]:
        edges_g = np.load(DATA_PATH + "energy_bounds.npz")[str(grid)]
    else:
        edges_g = np.arange(groups_fine + 1, dtype=float)

    # Calculate the indicies for the specific grid
    if grid == 361:
        label_fine = str(groups_fine).zfill(3)
        label_coarse = str(groups_coarse).zfill(3)
        edges_data = np.load(DATA_PATH + "G361_grid_index.npz")
        edges_gidx_fine = edges_data[label_fine]
        edges_gidx_coarse = edges_data[label_coarse]
    else:
        edges_gidx_fine = energy_coarse_index(len(edges_g)-1, groups_fine)
    # Convert to correct type
    edges_gidx_fine = edges_gidx_fine.astype(np.int32)

    # Check for multiple energy groups
    if groups_coarse is not None:
        edges_gidx_coarse = energy_coarse_index(groups_fine, groups_coarse)
        edges_gidx_coarse = edges_gidx_coarse.astype(np.int32)
        return edges_g, edges_gidx_fine, edges_gidx_coarse

    return edges_g, edges_gidx_fine


def energy_velocity(groups, edges_g=None):
    """ Convert energy edges to speed at cell centers, Relative Physics
    Arguments:
        groups: Number of energy groups
        edges_g: energy grid bounds
    Returns:
        speeds at cell centers (cm/s)   """
    if np.all(edges_g == None):
        return np.ones((groups,))
    centers_gg = 0.5 * (edges_g[1:] + edges_g[:-1])
    gamma = (EV_TO_JOULES * centers_gg) / (MASS_NEUTRON * LIGHT_SPEED**2) + 1
    velocity = LIGHT_SPEED / gamma * np.sqrt(gamma**2 - 1) * 100
    return velocity


def gamma_time_steps(edges_t, gamma=0.5, half_step=True):
    """ Add gamma half time steps to original time steps with initial step
    where gamma = 0.5 or 2 - sqrt(2). For external source with TR-BDF2 problems.

    Arguments:
        edges_t: array of length (steps + 1)
        half_step: if True, first half step is 0.5, else gamma
    Returns:
        array of length (steps * 2 + 1)
    """

    if gamma == 0.5:
        half_steps = 0.5 * (edges_t[1:] + edges_t[:-1])
    else:
        half_steps = edges_t[:-1] + np.diff(edges_t) * gamma
    # Combine half steps
    combined_steps = np.sort(np.concatenate((edges_t, half_steps)))
    if half_step:
        combined_steps[1] = 0.5 * (combined_steps[0] + combined_steps[2])
    return combined_steps


def spatial1d(layers, edges_x, labels=False):
    """ Creating one-dimensional medium map

    :param layers: list of lists where each layer is a new material. A
        layer is comprised of an index (int), material name (str), and
        the width (str) in the form [index, material, width]. The width
        is the starting and ending points of the material (in cm)
        separated by a dash. If there are multiple regions, a comma can
        separate them. I.E. layer = [0, "plutonium", "0 - 2, 3 - 4"].
    :param edges_x: Array of length I + 1 with the location of the cell edges
    :return: One-dimensional array of length I, identifying the locations
        of the materials
    """
    if labels:
        # Initialize label map
        medium_map = -1 * np.ones((len(edges_x) - 1)) * -1
    else:
        # Initialize medium_map
        medium_map = -1 * np.ones((len(edges_x) - 1), dtype=np.int32)
    # Iterate over all layers
    for layer in layers:
        for region in layer[2].split(","):
            start, stop = region.split("-")
            idx1 = np.argmin(np.fabs(float(start) - edges_x))
            idx2 = np.argmin(np.fabs(float(stop) - edges_x))
            medium_map[idx1:idx2] = layer[0]
    # Verify all cells are filled
    assert np.all(medium_map != -1)
    return medium_map
