
import numpy as np
import numba


########################################################################
# Multigroup functions
########################################################################

@numba.jit("void(f8[:,:], f8[:,:], i4[:], f8[:,:,:], f8[:], i4)", \
            nopython=True, cache=True)
def _off_scatter(flux, flux_old, medium_map, xs_scatter, off_scatter, gg):
    # Get parameters
    cells_x, groups = flux.shape
    ii = numba.int32
    mat = numba.int32
    og = numba.int32
    # Zero out previous off scatter term
    off_scatter *= 0.0
    # Iterate over cells and groups
    for ii in range(cells_x):
        mat = medium_map[ii]
        for og in range(0, gg):
            off_scatter[ii] += xs_scatter[mat,gg,og] * flux[ii,og]
        for og in range(gg+1, groups):
            off_scatter[ii] += xs_scatter[mat,gg,og] * flux_old[ii,og]


def reflector_corrector(reflector, angle_x, edge, nn, bc_x):
    # Get opposite angle
    reflected_idx = numba.int32(angle_x.shape[0] - nn - 1)
    if (angle_x[nn] > 0.0 and bc_x[1] == 1) or (angle_x[nn] < 0.0 and bc_x[0] == 1):
        reflector[reflected_idx] = edge


########################################################################
# K-eigenvalue functions
########################################################################

@numba.jit("void(f8[:,:], f8[:,:,:], f8[:,:,:], i4[:], f8)", \
            nopython=True, cache=True)
def _fission_source(flux, xs_fission, source, medium_map, keff):
    # Get parameters
    cells_x, groups = flux.shape
    ii = numba.int32
    mat = numba.int32
    og = numba.int32
    ig = numba.int32
    # Zero out previous source
    source *= 0.0
    # Iterate over cells and groups
    for ii in range(cells_x):
        mat = medium_map[ii]
        for og in range(groups):
            # source[ii,0,og] = np.sum(flux[ii] @ xs_fission[mat,og]) / keff
            for ig in range(groups):
                source[ii,0,og] += flux[ii,ig] * xs_fission[mat,og,ig]
            source[ii,0,og] /= keff



@numba.jit("f8(f8[:,:], f8[:,:], f8[:,:,:], i4[:], f8)", \
            nopython=True, cache=True)
def _update_keffective(flux, flux_old, xs_fission, medium_map, keff):
    # Get iterables
    cells_x, groups = flux.shape
    ii = numba.int32
    mat = numba.int32
    og = numba.int32
    ig = numba.int32
    # Initialize fission rates
    rate_new = 0.0
    rate_old = 0.0
    # Iterate over cells and groups
    for ii in range(cells_x):
        mat = medium_map[ii]
        for og in range(groups):
            for ig in range(groups):
                rate_new += flux[ii,ig] * xs_fission[mat,og,ig]
                rate_old += flux_old[ii,ig] * xs_fission[mat,og,ig]
    return (rate_new * keff) / rate_old
