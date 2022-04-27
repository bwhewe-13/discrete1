import numpy as np

""" Tools for discrete ordinates codes (dimensional purposes/convergence) """
def group_reduction(new_group,grid,energy=False,**kwargs):
    """ Used to reduce the number of groups for cross sections and energy levels
    Arguments:
        new_group: the reduction in the number of groups from the original
        grid: the original energy grid
        energy: True/False to return the new energy grid, default is False
        kwargs:
            total: total cross section of one spatial cell
            scatter: scatter cross section of one spatial cell
            fission: fission cross section of one spatial cell
    Returns:
        The reduced matrices and vectors of the cross sections or energy
        (Specified in the kwargs)   """

    # Remove the extra grid boundary
    old_group = len(grid) - 1
    # How many groups are combined (approximate)
    split = int(old_group/new_group)
    # Calculate the leftovers
    rmdr = old_group % new_group
    # Create array showing the number of groups combined
    new_grid = np.ones(new_group) * split
    # Add the remainder groups to the first x number
    # new_grid[:rmdr] += 1
    # new_grid = new_grid[::-1]
    new_grid[np.linspace(0,new_group-1,rmdr,dtype=int)] += 1
    assert (new_grid.sum() == old_group)

    # Calculate the indices while including the left-most (insert)
    inds = np.cumsum(np.insert(new_grid,0,0),dtype=int)

    # Return the new energy grid (for graphing purposes)
    if energy:
        return np.array([sum(grid[inds[ii]:inds[ii+1]]) / (inds[ii+1] - inds[ii]) for ii in range(new_group)])

    # This is for scaling the new groups properly
    # Calculate the change in energy for each of the new boundaries (of size new_group)
    new_diff_grid = np.array([grid[inds[ii+1]]-grid[inds[ii]] for ii in range(new_group)])
    # Repeated for the matrix (fission and scatter)
    new_diff_matrix = new_diff_grid[:,None] @ new_diff_grid[None,:]
    # Calculate the change in energy for each of the old boundaries
    old_diff_grid = np.diff(grid)
    # Repeated for the matrix (fission and scatter)
    old_diff_matrix = old_diff_grid[:,None] @ old_diff_grid[None,:]

    new_total = None; new_scatter = None; new_fission = None
    # Work through the cross section terms
    # print(sn.kwargs["total"].shape)
    if "total" in kwargs:
        total = kwargs["total"]
        total *= old_diff_grid
        new_total = sn.vector_reduction(total,inds)
        new_total /= new_diff_grid

    if "scatter" in kwargs:
        scatter = kwargs["scatter"]
        scatter *= old_diff_grid
        new_scatter = sn.matrix_reduction(scatter,inds)
        new_scatter /= new_diff_grid

    if "fission" in kwargs:
        fission = kwargs["fission"]
        fission *= old_diff_grid
        new_fission = sn.matrix_reduction(fission,inds)
        new_fission /= new_diff_grid

    return new_total, new_scatter, new_fission


def matrix_reduction(matrix,indices):
    """ Sum the matrix according to the indicies
    Arguments:
        matrix: the full size matrix that will be reduced
        indices: the location of which cells will be combined
    Returns:
        a matrix of size len(indices) - 1
    """
    # Remove the extra grid boundary
    new_group = len(indices) - 1
    reduced = np.array([[np.sum(matrix[indices[ii]:indices[ii+1],indices[jj]:indices[jj+1]]) for jj in range(new_group)] for ii in range(new_group)])
    return reduced

def vector_reduction(vector,indices):
    """ Sum the vector according to the indicies
    Arguments:
        vector: the full size matrix that will be reduced
        indices: the location of which cells will be combined
    Returns:
        a vector of size len(indices) - 1
    """
    # Remove the extra grid boundary
    new_group = len(indices) - 1
    # Sum the vector
    reduced = np.array([sum(vector[indices[ii]:indices[ii+1]]) for ii in range(new_group)])
    return reduced

def cat(lst,splits):
    return np.concatenate(([lst[ii] for ii in splits]))

def enrich_list(length,enrich,splits):
    lst = np.zeros((length))
    # Ensuring lists
    # if type(enrich) != list:
    #     enrich = [enrich]
    if type(splits) != list:
        splits = [splits]
    if type(enrich) == list:
        for ii in range(len(splits)):
            lst[splits[ii]] = enrich[ii]
        return lst
    for ii in range(len(splits)):
        lst[splits[ii]] = enrich
    return lst

def enriche(phi,model_sca,model_fis,dtype,splits,enrich=None):
    # Remove L+1 dimension
    normed = sn.cat(phi[:,0],splits)
    if enrich is not None:
        if dtype == 'scatter' or dtype == 'both':
            djinn_scatter_ns = model_sca.predict(np.concatenate((np.expand_dims(sn.cat(enrich,splits),axis=1),normed),axis=1))
        if dtype == 'fission' or dtype == 'both':
            djinn_fission_ns = model_fis.predict(np.concatenate((np.expand_dims(sn.cat(enrich,splits),axis=1),normed),axis=1))
    else:
        if dtype == 'scatter' or dtype =='both':
            djinn_scatter_ns = model_sca.predict(normed)
        if dtype == 'fission' or dtype == 'both':
            djinn_fission_ns = model_fis.predict(normed)
    if dtype == 'fission':
        djinn_scatter_ns = None
    if dtype == 'scatter':
        djinn_fission_ns = None
    return djinn_scatter_ns,djinn_fission_ns

def enrich_locs(layers,zones):
    full = np.zeros(np.sum(layers))
    ind = np.cumsum(layers)
    for ii in zones:
        if ii == 0:
            full[:ind[ii-1]] = 1
        else:
            full[ind[ii-2]:ind[ii-1]] = 1
    return full.astype(int)

def layer_slice(layers,half=True):
    # if half:
    #     split = int(len(layers)*0.5)
    #     layers[split:split+1] = [int(layers[split]*0.5)]*2
    bounds = np.cumsum(layers)
    bounds = np.insert(bounds,0,0)
    return [slice(bounds[ii],bounds[ii+1]) for ii in range(len(bounds)-1)]

def layer_slice_dict(layers,djinn,half=True):
    splitDic = {}
    keep = np.arange(len(layers))
    keep = keep[~np.in1d(keep,djinn)]

    keep_short = np.array(layers)[keep]
    djinn_short = np.array(layers)[djinn]
    splits = sn.layer_slice(layers)
    splits_dj = sn.layer_slice(djinn_short)
    splits_kp = sn.layer_slice(keep_short)

    splitDic['keep'] = [splits[ii] for ii in keep]
    splitDic['djinn'] = [splits[ii] for ii in djinn]
    splitDic['keep_short'] = splits_kp.copy()
    splitDic['djinn_short'] = splits_dj.copy()
    for key,val in splitDic.items(): # Not leave empty lists
        if len(val) == 0:
            splitDic[key] = [slice(0,0)]
    return splitDic

def layer_slice_dict_mixed(layers,half=True):
    splits = sn.layer_slice(layers,half)
    splitDic = {}
    keep = [0]; djinn = [1,2,3,4]
    splitDic['keep'] = [splits[ii] for ii in keep]
    splitDic['djinn'] = [splits[jj] for jj in djinn]
    return splitDic

def length(splits):
    "Length of the splice function"
    if type(splits) == slice:
        return len(range(*splits.indices(1000)))
    return sum([len(range(*jj.indices(1000))) for jj in splits])

def keff_correct(lst):
    for ii in range(len(lst)):
        if lst[ii].shape != ():
            lst[ii] = lst[ii][-1]
    return lst


def pops(full,reduce,splits,which):
    # Repopulating matrix from reduced
    for ii in range(len(splits[which])):
        full[splits[which][ii]] = reduce[splits['{}_short'.format(which)][ii]]
    return full

def pops_robust(xs,shape,keep,djinn,splits):
    """ Repopulating Phi Matrix
    Shape: shape of phi
    keep: reduced non-DJINN predicted matrix
    djinn: redcued DJINN predicted matrix
    splits: dictionary of splits    """
    full = np.zeros((shape))
    for ii in range(len(splits['{}_keep'.format(xs)])):
        full[splits['{}_keep'.format(xs)][ii]] = keep[splits['{}_keep_short'.format(xs)][ii]]
    for jj in range(len(splits['{}_djinn'.format(xs)])):
        full[splits['{}_djinn'.format(xs)][jj]] = djinn[splits['{}_djinn_short'.format(xs)][jj]]
    return full

def propagate(xs=None,G=None,I=None,N=None,L=None,dtype='total'):
    if dtype == 'total':
        return np.repeat(xs.reshape(1,G),I,axis=0)
    elif dtype == 'scatter':
        return np.tile(xs,(I,1,1,1))
    elif dtype == 'external':
        external = np.zeros((I,N,G))
        temp = np.repeat((xs).reshape(1,G),N,axis=0)
        for ii in range(I):
            external[ii] = temp
        return external
    elif dtype == 'boundary':
        return np.zeros((N,2,G))
    elif dtype == 'fission':
        return np.tile(xs,(I,1,1))
    return 'Incorrect dtype or cross section format'

def mixed_propagate(xs,layers,G=None,L=None,dtype='total'):
    ''' xs - list of cross sections
        layers - list of layer sizes
    '''
    if dtype == 'total':
        total = np.empty((0,G))
        for c,ii in enumerate(layers):
            total = np.vstack((total,np.repeat(xs[c].reshape(1,G),ii,axis=0)))
        return total
    elif dtype == 'fission':
        fission = np.empty((0,G))
        for c,ii in enumerate(layers):
            fission = np.vstack((fission,np.repeat(xs[c].reshape(1,G),ii,axis=0)))
        return fission
    elif dtype == 'fission2':
        fission = np.empty((0,G,G))
        for c,ii in enumerate(layers):
            fission = np.vstack((fission,np.repeat(xs[c].reshape(1,G,G),ii,axis=0)))
        return fission
    elif dtype == 'scatter':
        scatter = np.empty((0,L+1,G,G))
        for c,ii in enumerate(layers):
            scatter = np.vstack((scatter,np.repeat(xs[c].reshape(1,L+1,G,G),ii,axis=0)))
        return scatter
    return 'Incorrect dtype or cross section format'

def svd(A,r=None,full=True):
    u,s,v = np.linalg.svd(A)
    if full:
        sigma = np.zeros((A.shape[0],A.shape[1]))
        sigma[:s.shape[0],:s.shape[0]] = np.diag(s)
    else:
        sigma = s.copy()
    if r is not None:
        return u[:,:r],sigma[:r,:r],v[:r]
    return u,sigma,v

def wynnepsilon(lst, r):
    """ Perform Wynn Epsilon Convergence Algorithm
    Arguments:
        lst: list of values for convergence
        r: rank of system
    Returns:
        2D Array where diagonal is convergence """
    r = int(r)
    n = 2 * r + 1
    e = np.zeros(shape=(n + 1, n + 1))
    for i in range(1, n + 1):
        e[i, 1] = lst[i - 1]
    for i in range(3, n + 2):
        for j in range(3, i + 1):
            e[i - 1, j - 1] = e[i - 2, j - 3] + 1 / (e[i - 1, j - 2] - e[i - 2, j - 2])
    er = e[:, 1:n + 1:2]
    return er

def totalFissionRate(fission,phi,splits=None):
    if splits is not None:
        from discrete1.util import sn
        phi = sn.cat(phi,splits['djinn'])
        fission = sn.cat(fission,splits['djinn'])
    return np.sum(phi*np.sum(fission,axis=1),axis=1)

def totalRateShort(fission,phi,splits):
    density = np.zeros((phi.shape))
    for ii in range(len(splits)):
        density[splits[ii]] = phi[splits[ii]] * np.sum(fission[ii],axis=0)
    return np.sum(density,axis=1)


def djinnFissionRate(model,phi,fission,label=None):
    if label is not None:
        phi2 = np.hstack((label[:,None],phi))
    else:
        phi2 = phi.copy()
    temp1 = np.sum(model.predict(phi2),axis=1)
    scale = np.sum(phi*np.sum(fission,axis=1),axis=1)/temp1
    return scale*temp1