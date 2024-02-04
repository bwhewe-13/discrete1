
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


@numba.jit("void(f8[:,:,:], f8[:,:], f8[:,:,:], i4[:], f8[:,:,:])", \
            nopython=True, cache=True)
def _source_total(source, flux, xs_scatter, medium_map, external):
    # Get parameters
    cells_x, angles, groups = source.shape
    ii = numba.int32
    nn = numba.int32
    nn_q = numba.int32
    mat = numba.int32
    og = numba.int32
    og_q = numba.int32
    ig = numba.int32
    one_group = numba.float64
    # Zero out previous source term
    source *= 0.0
    # Iterate over cells, angles, groups
    for ii in range(cells_x):
        mat = medium_map[ii]

        # Iterate over groups
        for og in range(groups):
            og_q = 0 if external.shape[2] == 1 else og
            one_group = 0.0
            for ig in range(groups):
                one_group += flux[ii,ig] * xs_scatter[mat,og,ig]

            # Iterate over angles
            for nn in range(angles):
                nn_q = 0 if external.shape[1] == 1 else nn
                source[ii,nn,og] = one_group + external[ii,nn_q,og_q]


########################################################################
# Multigroup functions - DMD
########################################################################

def _svd_dmd(A, K):
    residual = 1e-09

    # Compute SVD
    U, S, V = np.linalg.svd(A, full_matrices=False)

    # Find the non-zero singular values
    if (S[(1-np.cumsum(S)/np.sum(S)) > residual].size >= 1):
        spos = S[(1 - np.cumsum(S) / np.sum(S)) > residual].copy()
    else:
        spos = S[S > 0].copy()

    # Create diagonal matrix
    mat_size = np.min([K, len(spos)])
    S = np.zeros((mat_size, mat_size))

    # Select the u and v that correspond with the nonzero singular values
    U = U[:, :mat_size].copy()
    V = V[:mat_size, :].copy()

    # S will be the inverse of the singular value diagonal matrix
    S[np.diag_indices(mat_size)] = 1 / spos

    return U, S, V


# @numba.jit("f8[:,:](f8[:,:], f8[:,:,:], f8[:,:,:], i4)", nopython=True, cache=True)
def dmd(flux_old, y_minus, y_plus, K):
    # Collect dimensions
    cells_x, groups = flux_old.shape

    # Flatten y_minus, y_plus
    y_minus = y_minus.reshape(cells_x * groups, K - 1)
    y_plus = y_plus.reshape(cells_x * groups, K - 1)

    # Call SVD
    U, S, V = _svd_dmd(y_minus, K)

    # Calculate Atilde
    Atilde = U.T @ y_plus @ V.T @ S.T

    # Calculate delta_y
    I = np.identity(Atilde.shape[0])
    delta_y = np.linalg.solve(I - Atilde, (U.T @ y_plus[:,-1]).T)

    # Estimate new flux
    flux = (flux_old.flatten() - y_plus[:,K-2]) + (U @ delta_y).T
    flux = flux.reshape(cells_x, groups)

    return flux


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


########################################################################
# DJINN functions
########################################################################

def _djinn_source_predict(flux, xs_matrix, source, models, model_map, \
        model_labels=None):
    # Zero out previous source
    source *= 0.0
    # Iterate over models
    for nn, model in enumerate(models):
        # Find which cells model is predicting
        model_idx = np.argwhere(model_map == nn).flatten()
        # Separate predicting flux
        predictor = flux[model_idx].copy()
        # Check for zero values
        if np.sum(predictor) == 0:
            continue
        # Get scaling factor
        scale = np.sum(predictor * xs_matrix[nn], axis=1)
        # Check for labels and predict
        if model_labels is not None:
            predictor = np.hstack((labels[model_idx][:,None], predictor))
            predictor = model.predict(predictor)
        else:
            predictor = model.predict(predictor)
        # Scale back and add to source
        source[model_idx] = predictor * (scale / np.sum(predictor, axis=1))[:,None]


@numba.jit("void(f8[:,:], f8[:,:,:], f8[:,:,:], i4[:], f8, i4[:])", \
            nopython=True, cache=True)
def _djinn_fission_pass(flux, xs_fission, source, medium_map, keff, model_map):
    # Get parameters
    cells_x, groups = flux.shape
    ii = numba.int32
    mat = numba.int32
    og = numba.int32
    ig = numba.int32
    # Check for zero flux
    if flux.sum() == 0.0:
        return
    # Iterate over cells and groups
    for ii in range(cells_x):
        # Check if there is a model to run
        if model_map[ii] == -1:
            continue
        # Calculate material
        mat = medium_map[ii]
        # Iterate over groups
        for og in range(groups):
            for ig in range(groups):
                source[ii,0,og] += flux[ii,ig] * xs_fission[mat,og,ig]
            source[ii,0,og] /= keff


@numba.jit("void(f8[:,:], f8[:,:,:], f8[:,:], i4[:], i4[:])", \
            nopython=True, cache=True)
def _djinn_scatter_pass(flux, xs_scatter, source, medium_map, model_map):
    # Get parameters
    cells_x, groups = flux.shape
    ii = numba.int32
    mat = numba.int32
    og = numba.int32
    ig = numba.int32
    # Iterate over cells and groups
    for ii in range(cells_x):
        # Check if there is a model to run
        if model_map[ii] == -1:
            continue
        # Calculate material
        mat = medium_map[ii]
        # Iterate over groups
        for og in range(groups):
            for ig in range(groups):
                source[ii,og] += flux[ii,ig] * xs_scatter[mat,og,ig]