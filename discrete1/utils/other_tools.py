''' Defunct functions that will possibly be used later '''
def indexing(energyGrid,boundaries,upper=False):
    '''Takes the energyGrid of the full scattering terms and
    returns the indexes for the group boundaries, to use with
    collapse function
    Arguments:
        energyGrid: numpy array of energy levels (in eV)
        boundaries: list of new energy boundaries for a reduced group problem
        upper: If true, includes all energy levels above the highest boundary
    Returns:
        index: numpy array of energy level indicies
    '''
    import numpy as np
    if upper == True:
        return np.append([np.min(np.where(energyGrid>=i)) for i in boundaries],len(energyGrid))
    return np.array([np.min(np.where(energyGrid>=i)) for i in boundaries])

# def gridPlot(energyGrid):
#     ''' Get the averages between the energy grid sides for plotting
#     against the different energies
#     Arguments:
#     energyGrid: 1D numpy array of size L+1
#     Returns:
#     An averaged energyGrid of size L
#     '''
#     import numpy as np
#     return np.array([float(energyGrid[ii]+energyGrid[jj])/2 for ii,jj in zip(range(len(energyGrid)-1),range(1,len(energyGrid)))])

def collapse(crossSection,index):
    ''' Collapsing different cross sections into specific groups
    Arguments:
        crossSection: the 2D numpy array of the original cross sections
        index: indicies to split the cross sections into groups
    Returns:
        reduced: the reduced vector of cross sections for the specified boundaries
    '''
    import numpy as np
    return np.array([np.sum(crossSection[index[i]:index[i+1]]) for i in range(len(index)-1)])

def matrixCollapse(matrix,index):
    '''Collapsing a scattering matrix to set energy groups
    Arguments:
        matrix: the numpy 2D array that will be reduced
        index: a list of the indicies for break points
    Returns:
        A reduced numpy 2D array
    '''
    import numpy as np
    if len(matrix.shape) == 3:
        temp = []
        for jj in range(matrix.shape[0]):
            temp.append(util.matCol(matrix[jj,:,:],index))
        return np.array(temp)
    return np.array([np.sum(matrix[index[i]:index[i+1],index[j]:index[j+1]]) for i in range(len(index)-1) for j in range(len(index)-1)]).reshape(len(index)-1,len(index)-1)

def matCol(matrix,index):
    '''Workhorse behind collapsing a scattering matrix to set energy groups
    Arguments:
    matrix: the numpy 2D array that will be reduced
    index: a list of the indicies for break points
    Returns:
    A reduced numpy 2D array
    '''
    import numpy as np
    return np.array([np.sum(matrix[index[i]:index[i+1],index[j]:index[j+1]]) for i in range(len(index)-1) for j in range(len(index)-1)]).reshape(len(index)-1,len(index)-1)


