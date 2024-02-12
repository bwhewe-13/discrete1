
import numpy as np


def reaction_rates(flux, xs_matrix, medium_map):
    # Flux parameters
    cells_x, groups = flux.shape
    # Initialize reaction rate data
    rate = np.zeros((cells_x, groups))
    # Iterate over spatial cells
    for ii, mat in enumerate(medium_map):
        rate[ii] = flux[ii] @ xs_matrix[mat].T
    # return reaction rate
    return rate


def spatial_error(approx, reference):
    """ Calculating the spatial error between an approximation and the
    reference solution
    Arguments:
        approx (array double): approximate flux
        reference (array double): Reference flux
    Returns:
        L2 error normalized for the number of spatial cells
    """
    assert approx.shape == reference.shape, "Not the same array shape"
    normalized = (approx.shape[0])**(-0.5)
    return normalized * np.linalg.norm(approx - reference)


def order_accuracy(error1, error2, ratio):
    """ Finding the order of accuracy between errors on different 
    grids, where error2 is the refined grid
    Arguments:
        error1 (double): Error between an approximate solution and the
            reference solution on the same grid
        error2 (double): Error between an approximate solution and the
            reference solution on a more refined grid
        ratio (double): Ratio between the spatial cell width of the error1
            grid and the error2 grid (delta x1 / delta x2)
    Returns:
        Order of accuracy
    """
    return np.log(error1 / error2) / np.log(ratio)


def wynn_epsilon(lst, rank):
    """ Perform Wynn Epsilon Convergence Algorithm
    Arguments:
        lst: list of values for convergence
        rank: rank of system
    Returns:
        2D Array where diagonal is convergence
    """
    N = 2 * rank + 1
    error = np.zeros((N + 1, N + 1))
    for ii in range(1, N + 1):
        error[ii, 1] = lst[ii - 1]
    for ii in range(3, N + 2):
        for jj in range(3, ii + 1):
            if (error[ii-1,jj-2] - error[ii-2,jj-2]) == 0.0:
                error[ii-1,jj-1] = error[ii-2,jj-3]
            else:
                error[ii-1,jj-1] = error[ii-2,jj-3] \
                            + 1 / (error[ii-1,jj-2] - error[ii-2,jj-2])
    return abs(error[-1,-1])