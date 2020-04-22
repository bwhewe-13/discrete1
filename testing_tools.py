def error_calc(original,predicted):
    return (original-predicted)**2/original*100

def split_graph(array):
    import numpy as np
    return np.arange(-int(array.shape[0]*0.5),int(array.shape[0]*0.5))

def percent_change(orig,new):
    import numpy as np
    return np.round(abs(orig-new)/orig,2)

def energy_grid(energy):
    grid = []
    for ii in range(len(energy)-1):
        grid.append((energy[ii]+energy[ii+1])*0.5)
    return grid

def unnormalize(scatter,maxi,mini,ind):
    return scatter*(maxi[ind]-mini[ind])+mini[ind]

def wynnepsilon(sn, r):
    '''Perform Wynn Epsilon Convergence Algorithm
    Arguments:
        sn: list of values for convergence
        r: rank of system
    Returns:
        2D Array where diagonal is convergence
    '''
    import numpy as np
    r = int(r)
    n = 2 * r + 1
    e = np.zeros(shape=(n + 1, n + 1))

    for i in range(1, n + 1):
        e[i, 1] = sn[i - 1]

    for i in range(3, n + 2):
        for j in range(3, i + 1):
            e[i - 1, j - 1] = e[i - 2, j - 3] + 1 / (e[i - 1, j - 2] - e[i - 2, j - 2])

    er = e[:, 1:n + 1:2]
    return er

def sizer(dim,original=87**2,half=True):
    ''' Calculates the trainable parametes for autoencoder
    Arguments:
        dim: list of dimension size of each encoding layer
        original: original data size (default is 87**2)
        half: if to take half of the parameters
    Returns:
        integer for number of trainable parameters
    '''
    if original:
        dim.insert(0,original)
    if len(dim) == 2:
        total = sum(dim)
    else:
        total = sum(dim)*2-dim[0]-dim[-1]
    for ii in range(len(dim)-1):
        total += (dim[ii]*dim[ii+1]*2)
    if half:
        return 0.5*total
    return int(total)

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

def gridPlot(energyGrid):
    ''' Get the averages between the energy grid sides for plotting 
    against the different energies
    Arguments:
    energyGrid: 1D numpy array of size L+1
    Returns:
    An averaged energyGrid of size L
    '''
    import numpy as np
    return np.array([float(energyGrid[ii]+energyGrid[jj])/2 for ii,jj in zip(range(len(energyGrid)-1),range(1,len(energyGrid)))])

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

def propagate(xs=None,G=None,I=None,N=None,L=None,dtype='total'):
    import numpy as np
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
    import numpy as np
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
    elif dtype == 'scatter':
        scatter = np.empty((0,L+1,G,G))
        for c,ii in enumerate(layers):
            scatter = np.vstack((scatter,np.repeat(xs[c].reshape(1,L+1,G,G),ii,axis=0)))
        return scatter
    return 'Incorrect dtype or cross section format'


def u235_mic2mac(micro,density=18.8,molar=235.04393):
    ''' Converts microscopc cross sections to macroscopic
    Arguments:
    micro: microscopic cross section to be converted
    density: density of element (Uranium is 18.8 g/cm^3)
    molar: molar mass of element (Uranium-235 is 235.04393 g/mol)
    Returns:
    Macroscopic cross section
    '''
    return micro*1e-24*density*6.022e23/molar

def cleaning_compound(compound):
    import re
    compound = re.findall('[A-Z][^A-Z]*',compound)
    counter = []
    for ii in range(len(compound)):
        if len(re.findall('[0-9][^0-9]*',compound[ii])) == 0:
            counter.append(1)
            compound[ii] = re.sub(r'[0-9][A-Za-z]+','',compound[ii])
        else:
            if '^' in compound[ii]:
                if '_' not in compound[ii]:
                    compound[ii] = compound[ii]+'_1'
                isotope = re.findall('\^(.*)\_',compound[ii])[0]
                counter.append(int(re.findall('\_(.*)',compound[ii])[0]))
                compound[ii] = re.sub(r'[^A-Za-z]+','',compound[ii])+'-'+isotope
            else:
                counter.append(int(''.join(re.findall('[0-9][^0-9]*',compound[ii]))))
                compound[ii] = re.sub(r'[0-9]+','',compound[ii])
    return compound,counter

def total_mass(compound,counter,enrich,library):
    molar_mass = 0
    for ii,jj in zip(compound,counter):
        if ii == 'U':
            molar_mass += (enrich*library['U-235']+(1-enrich)*library['U-238'])*jj
        else:
            molar_mass += library[ii]*jj
    return molar_mass

def number_density(compound,counter,molar_mass,density,enrich,library):
    NA = 6.022E23 # Avagadros number
    density_list = []
    for ckk,kk in enumerate(compound):
        if kk == 'U':
            density_list.append(((enrich*density*NA)/library['U-235']*(enrich*library['U-235']+(1-enrich)*library['U-238'])/molar_mass)*counter[ckk])
            density_list.append((((1-enrich)*density*NA)/library['U-238']*(enrich*library['U-235']+(1-enrich)*library['U-238'])/molar_mass)*counter[ckk])
        else:
            density_list.append(((density*NA)/molar_mass)*counter[ckk])
    return density_list

def density_list(compound,density,enrich=0.0,library=None):
    import numpy as np
    if library is None:
        import json
        library = json.load(open('discrete/scripts/element_dictionary.txt'))
    
    # Formatting (compounds and count of each)
    compound,counter = cleaning_compound(compound)

    # Calculating molar mass
    molar_mass = total_mass(compound,counter,enrich,library)
    
    # number densities
    density_list = number_density(compound,counter,molar_mass,density,enrich,library)
    
    # Convert from barns
    return np.array(density_list)*1e-24

# counter = [1 if len(re.findall('[0-9][^0-9]*',compound[ii])) == 0 else int(''.join(re.findall('[0-9][^0-9]*',compound[ii]))) for ii in range(len(compound))]
# compound = [re.sub(r'[0-9]+','',ii) for ii in compound]
