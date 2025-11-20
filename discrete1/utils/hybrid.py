"""Functions to be used with the collision-based hybrid method.

Allows for discretizing energy grids and material cross sections
"""

import numpy as np

import discrete1


########################################################################
# Coarsening Arrays for Hybrid Methods
########################################################################
def energy_coarse_index(fine, coarse):
    """Get the indices for resizing matrices.

    Parameters
    ----------
    fine : int
        Larger energy group size.
    coarse : int
        Coarser energy group size.

    Returns
    -------
    numpy.ndarray, int, shape (coarse + 1,)
    """
    index = np.ones((coarse)) * int(fine / coarse)
    index[np.linspace(0, coarse - 1, fine % coarse, dtype=np.int32)] += 1
    assert index.sum() == fine
    return np.cumsum(np.insert(index, 0, 0), dtype=np.int32)


def coarsen_materials(xs_total, xs_scatter, xs_fission, edges_g, edges_gidx):
    r"""Coarsen (materials, groups) arrays to (materials, groups\').

    Parameters
    ----------
    xs_total : numpy.ndarray
        2D array with float type of shape (materials, groups).
        Total cross section.
    xs_scatter : numpy.ndarray
        3D array with float type of shape (materials, groups, groups).
        Scatter cross section.
    xs_fission : numpy.ndarray
        3D array with float type of shape (materials, groups, groups).
        Fission cross section.
    edges_g : numpy.ndarray
        1D array with float type of shape (groups + 1,).
        Energy group bounds.
    edges_gidx : numpy.ndarray
        1D array with int type of shape (groups\' + 1,).
        Index of energy group bounds for new energy grid.

    Returns
    -------
    coarse_total : numpy.ndarray
        2D array with float type of shape (materials, groups\').
        Coarsened total cross section.
    coarse_scatter : numpy.ndarray
        3D array with float type of shape (materials, groups\', groups\').
        Coarsened scatter cross section.
    coarse_fission : numpy.ndarray
        3D array with float type of shape (materials, groups\', groups\').
        Coarsened fission cross section.
    """
    coarse_total = _xs_vector_coarsen(xs_total, edges_g, edges_gidx)
    coarse_scatter = _xs_matrix_coarsen(xs_scatter, edges_g, edges_gidx)
    coarse_fission = _xs_matrix_coarsen(xs_fission, edges_g, edges_gidx)
    return coarse_total, coarse_scatter, coarse_fission


def coarsen_external(external, edges_g, edges_gidx, weight=False):
    r"""Coarsen (... x groups) arrays to (... x groups\').

    Parameters
    ----------
    external : numpy.ndarray
        Array with float type of shape (..., groups).
        Array to coarsen.
    edges_g : numpy.ndarray
        1D array with float type of shape (groups + 1,).
        Energy group bounds.
    edges_gidx : numpy.ndarray
        1D array with int type of shape (groups\' + 1,).
        Index of energy group bounds for new energy grid.
    weight : bool, optional
        Weighting the collapsing groups. Default is False.

    Returns
    -------
    numpy.ndarray
        Array with float type of shape (..., groups\').
    """
    if external.shape[-1] == 1:
        return external
    groups_coarse = edges_gidx.shape[0] - 1
    # Create coarsened array
    coarse = np.zeros(external.shape[:-1] + (groups_coarse,))
    # Create energy bin widths
    delta_fine = np.diff(edges_g)
    delta_coarse = np.diff(np.asarray(edges_g)[edges_gidx])
    # Condition vector with energy bin width
    fine = external * delta_fine[..., :] if weight else external.copy()
    for gg, (gg1, gg2) in enumerate(zip(edges_gidx[:-1], edges_gidx[1:])):
        coarse[..., gg] = np.sum(fine[..., gg1:gg2], axis=-1)
    # Coarsen
    if weight:
        coarse /= delta_coarse[..., :]
    return coarse


def _xs_vector_coarsen(vector, edges_g, edges_gidx):
    r"""Coarsen (materials, groups) arrays to (materials, groups\').

    Parameters
    ----------
    vector : numpy.ndarray
        2D array with float type of shape (materials, groups).
        Array to coarsen.
    edges_g : numpy.ndarray
        1D array with float type of shape (groups + 1,).
        Energy group bounds.
    edges_gidx : numpy.ndarray
        1D array with int type of shape (groups\' + 1,).
        Index of energy group bounds for new energy grid.

    Returns
    -------
    numpy.ndarray
        2D array with float type of shape (materials, groups\').
    """
    materials = vector.shape[0]
    groups_coarse = edges_gidx.shape[0] - 1
    # Create coarsened array
    coarse = np.zeros((materials, groups_coarse))
    # Create energy bin widths
    delta_fine = np.diff(edges_g)
    delta_coarse = np.diff(np.asarray(edges_g)[edges_gidx])
    # Condition vector with energy bin width
    fine = np.asarray(vector) * delta_fine[None, :]
    # Loop over all materials
    for mat in range(materials):
        coarse[mat] = [
            np.sum(fine[mat, gg1:gg2])
            for gg1, gg2 in zip(edges_gidx[:-1], edges_gidx[1:])
        ]
    # Coarsen
    coarse /= delta_coarse[None, :]
    return coarse


def _xs_matrix_coarsen(matrix, edges_g, edges_gidx):
    r"""Coarsen (materials, groups, groups) arrays to (materials, groups\', groups\').

    Parameters
    ----------
    matrix : numpy.ndarray
        3D array with float type of shape (materials, groups, groups)
        Array to coarsen.
    edges_g : numpy.ndarray
        1D array with float type of shape (groups + 1,).
        Energy group bounds.
    edges_gidx : numpy.ndarray
        1D array with int type of shape (groups\' + 1,).
        Index of energy group bounds for new energy grid.

    Returns
    -------
    numpy.ndarray
        3D array of float type with shape (materials, groups\', groups\').
    """
    if matrix is None:
        return None
    materials = matrix.shape[0]
    groups_coarse = edges_gidx.shape[0] - 1
    # Create coarsened array
    coarse = np.zeros((materials, groups_coarse, groups_coarse))
    # Create energy bin widths
    delta_fine = np.diff(edges_g)
    delta_coarse = np.diff(np.asarray(edges_g)[edges_gidx])
    fine = np.asarray(matrix) * delta_fine[None, :]
    for mat in range(materials):
        coarse[mat] = [
            [
                np.sum(fine[mat][aa1:aa2, bb1:bb2])
                for bb1, bb2 in zip(edges_gidx[:-1], edges_gidx[1:])
            ]
            for aa1, aa2 in zip(edges_gidx[:-1], edges_gidx[1:])
        ]
    # Coarsen
    coarse /= delta_coarse[None, :]
    return coarse


def coarsen_velocity(vector, edges_gidx):
    r"""Coarsen (groups,) vector to (groups\',) where groups > groups\'.

    Parameters
    ----------
    vector : numpy.ndarray
        1D array with float type of shape (groups,).
        Array to coarsen.
    edges_gidx : numpy.ndarray
        1D array with int type of shape (groups\' + 1,).
        Index of energy group bounds for new energy grid

    Returns
    -------
    numpy.npdarray
        1D array with float type of shape (groups\',).
    """
    groups_coarse = edges_gidx.shape[0] - 1
    # Create coarsened array
    coarse = np.zeros((groups_coarse))
    for gg, (gg1, gg2) in enumerate(zip(edges_gidx[:-1], edges_gidx[1:])):
        coarse[gg] = np.mean(vector[gg1:gg2])
    return coarse


########################################################################
# Indexing for Hybrid Methods
########################################################################


def indexing(edges_g, edges_gidx_fine, edges_gidx_coarse):
    """Calculate the variables needed for refining and coarsening fluxes.

    Parameters
    ----------
    edges_g : numpy.ndarray
        1D array with float type of shape (groups_fine + 1,).
        Energy group bounds for fine grid.
    edges_gidx_fine : numpy.ndarray
        1D array with int type of shape (groups_fine + 1,).
        Index of energy group bounds for fine energy grid.
    edges_gidx_coarse : numpy.ndarray
        1D array with int type of shape (groups_coarse + 1,)
        Index of energy group bounds for coarse energy grid.

    Returns
    -------
    coarse_idx : numpy.ndarray
        1D array with int type of shape (groups_fine,).
        Coarse group mapping.
    fine_idx : numpy.ndarray
        1D array with int type of shape (groups_coarse + 1,).
        Location of edges between the coarse and fine energy grids.
    factor : numpy.ndarray
        1D array with float type of shape (groups_fine,).
        Fine energy bin width / coarse energy bin width for specific location.
    """
    # Calculate fine and coarse groups
    groups_fine = edges_gidx_fine.shape[0] - 1
    groups_coarse = edges_gidx_coarse.shape[0] - 1

    # Collect indexes
    fine_idx = uncollided_index(groups_coarse, edges_gidx_coarse)
    coarse_idx = collided_index(groups_fine, edges_gidx_coarse)

    # Convert from memoryview
    edges_g = np.asarray(edges_g)

    # Calculate energy bin widths
    delta_fine = np.diff(edges_g[edges_gidx_fine])
    delta_coarse = np.diff(edges_g[edges_gidx_fine][edges_gidx_coarse])
    factor = hybrid_factor(delta_fine, delta_coarse, edges_gidx_coarse)

    return fine_idx, coarse_idx, factor


def collided_index(groups_fine, edges_gidx):
    """Calculate which coarse group a fine energy group is a part of.

    This is done through:

    .. code-block:: text

        fine grid  :    |---g1---|--g2--|---g3---|--g4--|
        coarse grid:    |-------g1------|-------g2------|

    results in coarse_idx = [0, 0, 1, 1]

    Parameters
    ----------
    groups_fine : int
        Number of fine energy groups.
    edges_gidx : numpy.ndarray
        1D array with int type of shape (groups_coarse + 1,).
        Index of energy group bounds for coarse energy grid.

    Returns
    -------
    numpy.ndarray
        1D array with int type of shape (groups_fine,).
    """
    coarse_idx = np.zeros((groups_fine), dtype=np.int32)
    splits = [slice(ii, jj) for ii, jj in zip(edges_gidx[:-1], edges_gidx[1:])]
    for count, split in enumerate(splits):
        coarse_idx[split] = count
    return coarse_idx


def uncollided_index(groups_coarse, edges_gidx):
    """Calculate the location of edges between the coarse and fine energy grids.

    This is done through

    .. code-block:: text

        edge       :    0        1      2        3      4
        fine grid  :    |---g1---|--g2--|---g3---|--g4--|
        coarse grid:    |-------g1------|-------g2------|

    results in fine_idx = [0, 2, 4]

    Parameters
    ----------
    groups_coarse : int
        Number of coarse energy groups where groups_fine >= groups_coarse.
    edges_gidx : numpy.ndarray
        1D array with int type of shape (groups_coarse + 1,).
        Index of energy group bounds for coarse energy grid.

    Returns
    -------
    numpy.ndarray
        1D array with int type of shape (groups_coarse + 1,).
        Location of edges between the coarse and fine energy grids.
    """
    fine_idx = np.zeros((groups_coarse + 1), dtype=np.int32)
    splits = [slice(ii, jj) for ii, jj in zip(edges_gidx[:-1], edges_gidx[1:])]
    for count, split in enumerate(splits):
        fine_idx[count + 1] = split.stop
    return fine_idx


def hybrid_factor(delta_fine, delta_coarse, edges_gidx):
    """Calculate the fine energy bin width per coarse energy bin width.

    This is done through

    .. code-block:: text

        location (eV):    0        3      5        8      10
        fine grid    :    |---g1---|--g2--|---g3---|--g4--|
        coarse grid  :    |-------g1------|-------g2------|

    results in factor = [3/5, 2/5, 3/5, 2/5] with widths [3, 2, 2, 3]
    for the fine groups and [5, 5] for the coarse groups

    Parameters
    ----------
    delta_fine : numpy.ndarray
        1D array with float type of shape (groups_fine,).
        Energy group width for fine grid.
    delta_coarse : numpy.ndarray
        1D array with float type of shape (groups_fine,).
        Energy group width for coarse grid.
    edges_gidx : numpy.ndarray
        1D array with int type of shape (groups_coarse + 1,).
        Index of energy group bounds for coarse energy grid.

    Returns
    -------
    numpy.ndarray
        1D array of float type with shape (groups_fine,).
        Fine energy bin width / coarse energy bin width for specific location.
    """
    factor = delta_fine.copy()
    splits = [slice(ii, jj) for ii, jj in zip(edges_gidx[:-1], edges_gidx[1:])]
    for count, split in enumerate(splits):
        for ii in range(split.start, split.stop):
            factor[ii] /= delta_coarse[count]
    return factor


########################################################################
# On the fly changing energy grids
########################################################################
def energy_grid_change(starting_grid, groups_u, groups_c):
    """Calculate the variables needed for refining and coarsening fluxes in vhybrid.

    Parameters
    ----------
    starting_grid : int
        Value of starting energy grid, could be 87, 361, 618.
    groups_u : int
        Number of uncollided energy groups.
    groups_c : int
        Number of collided energy groups.

    Returns
    -------
    coarse_idx : numpy.ndarray
        1D array with int type of shape (groups_fine,).
        Coarse group mapping.
    factor : numpy.ndarray
        1D array with float type of shape (groups_fine,).
        Fine energy bin width / coarse energy bin width for specific location.
    edges_gidx_c : numpy.ndarray
        1D array with int type of shape (groups_c + 1,).
        Coarse group index mapping.
    edges_g : numpy.ndarray
        1D array with float type of shape (groups_u + 1,).
        Energy bound locations.
    """

    energy_grid = discrete1.energy_grid(starting_grid, groups_u, groups_c)
    edges_g, edges_gidx_u, edges_gidx_c = energy_grid

    # Collect indexes
    coarse_idx = collided_index(groups_u, edges_gidx_c)

    # Convert from memoryview
    edges_g = np.asarray(edges_g)

    # Calculate energy bin widths
    delta_fine = np.diff(edges_g[edges_gidx_u])
    delta_coarse = np.diff(edges_g[edges_gidx_u][edges_gidx_c])
    factor = hybrid_factor(delta_fine, delta_coarse, edges_gidx_c)

    return coarse_idx, factor, edges_gidx_c, edges_g
