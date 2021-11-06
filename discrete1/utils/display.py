''' Tools for graphing current problems '''
def error_calc(original,predicted):
    return (abs(original-predicted)/original)

def split_graph(array):
    import numpy as np
    return np.arange(-int(array.shape[0]*0.5),int(array.shape[0]*0.5))

def percent_change(orig,new):
    import numpy as np
    return np.round(abs(orig-new)/orig,2)

def gridPlot(energyGrid=None):
    """ Get the averages between the energy grid sides for plotting
    against the different energies
    Arguments:
    energyGrid: 1D numpy array of size N+1
    Returns:
    An averaged energyGrid of size N """
    import numpy as np
    if energyGrid is None:
        energyGrid = np.load('discrete1/data/energyGrid.npy')
    return np.array([float(energyGrid[ii]+energyGrid[jj])/2 for ii,jj in
                     zip(range(len(energyGrid)-1),range(1,len(energyGrid)))])