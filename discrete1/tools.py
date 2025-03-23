
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


@numba.jit("void(f8[:,:,:], f8[:,:], f8[:,:,:], i4[:])", nopython=True, cache=True)
def _time_right_side(q_star, flux, xs_scatter, medium_map):
    # Create (sigma_s + sigma_f) * phi + external + 1/(v*dt) * psi function
    # Get parameters
    cells_x, angles, groups = q_star.shape
    ii = numba.int32
    mat = numba.int32
    og = numba.int32
    ig = numba.int32
    one_group = numba.float64
    # Iterate over cells and groups
    for ii in range(cells_x):
        mat = medium_map[ii]
        # Iterate over groups
        for og in range(groups):
            one_group = 0.0
            for ig in range(groups):
                one_group += flux[ii,ig] * xs_scatter[mat,og,ig]
            q_star[ii,:,og] += one_group


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

@numba.jit("f8[:,:,:](f8[:,:], f8[:,:,:], f8[:,:,:], i4[:], f8)", \
            nopython=True, cache=True)
def _fission_source(flux, xs_fission, source, medium_map, keff):
    # Get parameters
    cells_x, groups = flux.shape
    ii = numba.int32
    mat = numba.int32
    og = numba.int32
    ig = numba.int32
    one_group = numba.float64
    # Zero out previous source
    source *= 0.0
    # Iterate over cells and groups
    for ii in range(cells_x):
        mat = medium_map[ii]
        if xs_fission[mat].sum() == 0.0:
            continue
        for og in range(groups):
            # source[ii,0,og] = np.sum(flux[ii] @ xs_fission[mat,og]) / keff
            one_group = 0.0
            for ig in range(groups):
                one_group += flux[ii,ig] * xs_fission[mat,og,ig]
            source[ii,0,og] = one_group / keff
    # Return matrix vector product
    return source



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
        if xs_fission[mat].sum() == 0.0:
            continue
        for og in range(groups):
            for ig in range(groups):
                rate_new += flux[ii,ig] * xs_fission[mat,og,ig]
                rate_old += flux_old[ii,ig] * xs_fission[mat,og,ig]
    return (rate_new * keff) / rate_old


########################################################################
# DJINN functions
########################################################################
# Fission source is of size (I x 1 x G) - for source iteration dimensions
# Scatter source is of size (I x G) - for discrete ordinates dimensions

def _djinn_fission_predict(flux, xs_fission, source, medium_map, keff, \
        models, model_labels=None):
    # Zero out previous source
    source *= 0.0
    
    # Iterate over models
    for nn, model in enumerate(models):
        # Find which cells model is predicting
        model_idx = np.argwhere(medium_map == nn).flatten()

        # Check if model available
        if isinstance(model, int):
            # Calculate standard source
            source[model_idx] = _fission_source(flux[model_idx], xs_fission, \
                            source[model_idx], medium_map[model_idx], keff)
            continue

        # Separate predicting flux
        predictor = flux[model_idx].copy()
        # Check for zero values
        if np.sum(predictor) == 0:
            continue
        # Get scaling factor
        scale = np.sum(predictor * xs_fission[nn,:,0], axis=1)
        # Check for labels and predict
        if model_labels is not None:
            predictor = np.hstack((model_labels[model_idx][:,None], predictor))
            predictor = model.predict(predictor)
        else:
            predictor = model.predict(predictor)
        # Scale back and add to source
        source[model_idx,0] = predictor / keff * \
                            (scale / np.sum(predictor, axis=1))[:,None]


def _djinn_scatter_predict(flux, xs_scatter, source, medium_map, models, \
        model_labels=None):
    # Zero out previous source
    source *= 0.0

    # Iterate over models
    for nn, model in enumerate(models):
        # Find which cells model is predicting
        model_idx = np.argwhere(medium_map == nn).flatten()

        # Check if model available
        if isinstance(model, int):
            # Calculate standard source
            source[model_idx] = _scatter_source(flux[model_idx], xs_scatter, \
                                    source[model_idx], medium_map[model_idx])
            continue

        # Separate predicting flux
        predictor = flux[model_idx].copy()
        # Check for zero values
        if np.sum(predictor) == 0:
            continue
        # Get scaling factor
        scale = np.sum(predictor * xs_scatter[nn,:,0], axis=1)
        # Check for labels and predict
        if model_labels is not None:
            predictor = np.hstack((model_labels[model_idx][:,None], predictor))
            predictor = model.predict(predictor)
        else:
            predictor = model.predict(predictor)
        # Scale back and add to source
        source[model_idx] = predictor * (scale / np.sum(predictor, axis=1))[:,None]


@numba.jit("f8[:,:](f8[:,:], f8[:,:,:], f8[:,:], i4[:])", nopython=True, cache=True)
def _scatter_source(flux, xs_scatter, source, medium_map):
    # Get parameters
    cells_x, groups = flux.shape
    ii = numba.int32
    mat = numba.int32
    og = numba.int32
    ig = numba.int32
    one_group = numba.float64
    # Iterate over cells and groups
    for ii in range(cells_x):
        # Calculate material
        mat = medium_map[ii]
        if xs_scatter[mat].sum() == 0.0:
            continue
        # Iterate over groups
        for og in range(groups):
            one_group = 0.0
            for ig in range(groups):
                one_group += flux[ii,ig] * xs_scatter[mat,og,ig]
            source[ii,og] = one_group
    # Return matrix vector product
    return source

########################################################################
# Hybrid functions
########################################################################

@numba.jit("void(f8[:,:], f8[:,:,:], f8[:,:,:], i4[:], i4[:])", nopython=True, cache=True)
def _hybrid_source_collided(flux_u, xs_scatter, source_c, medium_map, coarse_idx):
    # Get parameters
    cells_x, groups = flux_u.shape
    ii = numba.int32
    mat = numba.int32
    og = numba.int32
    ig = numba.int32
    # Zero out previous source
    source_c *= 0.0
    # Iterate over all spatial cells
    for ii in range(cells_x):
        mat = medium_map[ii]
        for og in range(groups):
            for ig in range(groups):
                source_c[ii,0,coarse_idx[og]] += flux_u[ii,ig] * xs_scatter[mat,og,ig]


@numba.jit("void(f8[:,:], f8[:,:], f8[:,:,:], f8[:,:,:], i4[:], i4[:], f8[:])", \
            nopython=True, cache=True)
def _hybrid_source_total(flux_u, flux_c, xs_scatter_u, q_star, medium_map, \
        coarse_idx, factor):
    # Get parameters
    cells_x, angles, groups = q_star.shape
    ii = numba.int32
    mat = numba.int32
    nn = numba.int32
    og = numba.int32
    ig = numba.int32
    one_group = numba.float64
    # Assume that source is already (Qu + 1 / (v * dt) * psi^{\ell-1})
    for ii in range(cells_x):
        mat = medium_map[ii]
        for og in range(groups):
            flux_u[ii,og] = flux_u[ii,og] + flux_c[ii,coarse_idx[og]] * factor[og]
        for og in range(groups):
            one_group = 0.0
            for ig in range(groups):
                one_group += flux_u[ii,ig] * xs_scatter_u[mat,og,ig]
            for nn in range(angles):
                q_star[ii,nn,og] += one_group


########################################################################
# Display Options
########################################################################

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


########################################################################
# Spatially Independent Functions
########################################################################

@numba.jit("f8(f8[:], f8[:], f8[:,:], i4)", nopython=True, cache=True)
def _off_scatter_0d(flux, flux_old, xs_scatter, gg):
    # Get parameters
    groups = flux.shape[0]
    off_scatter = numba.float64
    off_scatter = 0.0
    og = numba.int32
    # Iterate over groups
    for og in range(0, gg):
        off_scatter += xs_scatter[gg,og] * flux[og]
    for og in range(gg+1, groups):
        off_scatter += xs_scatter[gg,og] * flux_old[og]
    return off_scatter


@numba.jit("f8[:,:](f8[:], f8[:,:], f8[:,:], f8)", nopython=True, cache=True)
def _fission_source_0d(flux, xs_fission, source, keff):
    # Get parameters
    groups = flux.shape[0]
    og = numba.int32
    ig = numba.int32
    one_group = numba.float64
    # Zero out previous source
    source *= 0.0
    # Iterate over groups
    for og in range(groups):
        one_group = 0.0
        for ig in range(groups):
            one_group += flux[ig] * xs_fission[og,ig]
        source[0,og] = one_group / keff
    # Return matrix vector product
    return source


@numba.jit("f8(f8[:], f8[:], f8[:,:], f8)", nopython=True, cache=True)
def _update_keffective_0d(flux, flux_old, xs_fission, keff):
    # Get iterables
    groups = flux.shape[0]
    og = numba.int32
    ig = numba.int32
    # Initialize fission rates
    rate_new = 0.0
    rate_old = 0.0
    # Iterate over groups
    for og in range(groups):
        for ig in range(groups):
            rate_new += flux[ig] * xs_fission[og,ig]
            rate_old += flux_old[ig] * xs_fission[og,ig]
    return (rate_new * keff) / rate_old