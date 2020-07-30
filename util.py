class display:
    ''' Tools for graphing current problems '''
    def error_calc(original,predicted):
        return (abs(original-predicted)/original)*100

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

class sn:
    """ Tools for discrete ordinates codes (dimensional purposes/convergence) """
    def cat(lst,splits):
        import numpy as np
        return np.concatenate(([lst[ii] for ii in splits]))    

    def djinn_load(model_name,dtype):
        from djinn import djinn
        if dtype == 'both':
            model_scatter = djinn.load(model_name=model_name[0])
            model_fission = djinn.load(model_name=model_name[1])
        elif dtype == 'scatter':
            model_scatter = djinn.load(model_name=model_name)
            model_fission = None
        elif dtype == 'fission':
            model_scatter = None
            model_fission = djinn.load(model_name=model_name)
        return model_scatter,model_fission

    def enrich_list(length,enrich,splits):
        import numpy as np
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
        import numpy as np
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
        import numpy as np
        full = np.zeros(np.sum(layers))
        ind = np.cumsum(layers)
        for ii in zones:
            if ii == 0:
                full[:ind[ii-1]] = 1
            else:
                full[ind[ii-2]:ind[ii-1]] = 1
        return full.astype(int)
    
    def layer_slice(layers,half=True):
        import numpy as np
        # if half:
        #     split = int(len(layers)*0.5)
        #     layers[split:split+1] = [int(layers[split]*0.5)]*2
        bounds = np.cumsum(layers)
        bounds = np.insert(bounds,0,0)
        return [slice(bounds[ii],bounds[ii+1]) for ii in range(len(bounds)-1)]
    
    def layer_slice_dict(layers,djinn,half=True):
        import numpy as np
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
        import numpy as np
        full = np.zeros((shape))
        for ii in range(len(splits['{}_keep'.format(xs)])):
            full[splits['{}_keep'.format(xs)][ii]] = keep[splits['{}_keep_short'.format(xs)][ii]]
        for jj in range(len(splits['{}_djinn'.format(xs)])):
            full[splits['{}_djinn'.format(xs)][jj]] = djinn[splits['{}_djinn_short'.format(xs)][jj]]
        return full
    
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

    def wynnepsilon(lst, r):
        '''Perform Wynn Epsilon Convergence Algorithm
        Arguments:
            lst: list of values for convergence
            r: rank of system
        Returns:
            2D Array where diagonal is convergence
        '''
        import numpy as np
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
        import numpy as np
        if splits is not None:
            from discrete1.util import sn
            phi = sn.cat(phi,splits['djinn'])
            fission = sn.cat(fission,splits['djinn'])
        return np.sum(phi*np.sum(fission,axis=1),axis=1)
    
class nnets:
    """ Tools for autoencoders and neural networks """
    def unnormalize(scatter,maxi,mini,ind):
        return scatter*(maxi[ind]-mini[ind])+mini[ind]
    
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
    
    def djinn_metric(loc,metric='MSE',score=False,clean=False):
        ''' Gets the Best Metric for DJINN Models - Rough Comparison
        Arguments:
            loc: string location of DJINN model errors
                i.e. y_djinn_errors/model_ntrees*
        Returns None   '''
        import numpy as np
        import glob
        address = np.sort(glob.glob(loc))
        error_data = []
        # gets list of all data errors
        for ii in address:
            temp = []
            with open(ii,'r') as f:
                for ii in f.readlines():
                    broken = ii.split()
                    temp.append(np.array([broken[1],broken[5],broken[-1]],dtype=float))
            error_data.append(np.array(temp))
        # Calculate the least error from MSE, MAE, EVS
        tots = []
        for ii in range(len(address)):
            if metric == 'MSE':
                tots.append(np.sum(np.fabs(np.subtract(error_data[ii],[0,0,1])),axis=0)[0])
            elif metric == 'MAE':
                tots.append(np.sum(np.fabs(np.subtract(error_data[ii],[0,0,1])),axis=0)[1])
            elif metric == 'EVS':
                tots.append(np.sum(np.fabs(np.subtract(error_data[ii],[0,0,1])),axis=0)[2]) #explained variance score
            else:
                print('Combination of MSE, MAE, EVS')
                tots.append(np.sum(np.fabs(np.subtract(error_data[ii],[0,0,1]))))
        tots = np.array(tots)
        if score:
            print(tots[np.argmin(tots)])
        if clean:
            import re
            return ''.join(re.findall('\d{3}',address[np.argmin(tots)]))
        return address[np.argmin(tots)]

class chem:
    """ Getting number densities for different compounds """
    def mm_ss440(mm=False):
        """ Information from webelements.com and matweb.com """
        # percentage, MM
        cr50 = [0.04345, 49.9460464]
        cr52 = [0.83789, 51.9405098]
        cr53 = [0.09501, 52.9406513]
        cr54 = [0.02365, 53.9388825]
        crPer = 0.18
        cr = 1/sum([ii[0]/ii[1] for ii in [cr50,cr52,cr53,cr54]])
        fe54 = [0.05845,53.9396127]
        fe56 = [0.92036,55.9349393] # 0.91754
        fe57 = [0.02119,56.9353958]
        fePer = 0.79
        fe = 1/sum([ii[0]/ii[1] for ii in [fe54,fe56,fe57]])
        si28 = [0.922297,27.9769271]
        si29 = [0.046832,28.9764949]
        si30 = [0.030872,29.9737707]
        siPer = 0.01
        si = 1/sum([ii[0]/ii[1] for ii in [si28,si29,si30]])
        mn = 54.9380471; mnPer = 0.01
        c = 12.0116; cPer = 0.01
        mm_ss440 = 1/(fePer/fe+crPer/cr+cPer/c+siPer/si+mnPer/mn)
        if mm:
            return mm_ss440
        NA = 6.022E23; rho = 7.85 # g/cc
        Nfe = [(ii[0]*fePer*rho*NA)/ii[1]*1e-24 for ii in [fe54,fe56,fe57]]
        Ncr = [(ii[0]*crPer*rho*NA)/ii[1]*1e-24 for ii in [cr50,cr52,cr53,cr54]]
        Nsi = [(ii[0]*siPer*rho*NA)/ii[1]*1e-24 for ii in [si28,si29,si30]]
        Nmn = [(mnPer*rho*NA)/mn*1e-24]
        Nc = [(cPer*rho*NA)/c*1e-24]
        number_density = list(Nfe+Ncr+Nsi+Nmn+Nc)
        return number_density
    
    def xs_ss440(G,spec_temp):
        import numpy as np
        address_scatter = ['mydata/fe54/scatter_0{}.npy'.format(spec_temp),
                           'mydata/fe56/scatter_0{}.npy'.format(spec_temp),
                           'mydata/fe57/scatter_0{}.npy'.format(spec_temp),
                           'mydata/cr50/scatter_0{}.npy'.format(spec_temp),
                           'mydata/cr52/scatter_0{}.npy'.format(spec_temp),
                           'mydata/cr53/scatter_0{}.npy'.format(spec_temp),
                           'mydata/cr54/scatter_0{}.npy'.format(spec_temp),
                           'mydata/si28/scatter_0{}.npy'.format(spec_temp),
                           'mydata/si29/scatter_0{}.npy'.format(spec_temp),
                           'mydata/si30/scatter_0{}.npy'.format(spec_temp),
                           'mydata/mn55/scatter_0{}.npy'.format(spec_temp),
                           'mydata/cnat/scatter_0{}.npy'.format(spec_temp)]
        
        address_total = ['mydata/fe54/vecTotal.npy',
                         'mydata/fe56/vecTotal.npy',
                         'mydata/fe57/vecTotal.npy',
                         'mydata/cr50/vecTotal.npy',
                         'mydata/cr52/vecTotal.npy',
                         'mydata/cr53/vecTotal.npy',
                         'mydata/cr54/vecTotal.npy',
                         'mydata/si28/vecTotal.npy',
                         'mydata/si29/vecTotal.npy',
                         'mydata/si30/vecTotal.npy',
                         'mydata/mn55/vecTotal.npy',
                         'mydata/cnat/vecTotal.npy']
        number_density = chem.mm_ss440()
        total_xs = np.zeros((G))
        scatter_xs = np.zeros((G,G))
        for ii in range(len(number_density)):
            total_xs += number_density[ii]*np.load(address_total[ii])[eval(spec_temp)]
            scatter_xs += number_density[ii]*np.load(address_scatter[ii])[0]
        return total_xs,scatter_xs
        
    
    def micMac(xs,element):
        ''' Converts microscopc cross sections to macroscopic
        Arguments:
            xs: microscopic cross section to be converted
            element: string of element or list of element [molar mass,density]
                molar mass is g/mol, density is g/cm^3
        Returns:
            Macroscopic cross section
        '''
        if type(element) == str:
            import json
            library = json.load(open('discrete1/element_dictionary.json'))
            info = library[element]
        else:
            info = element.copy()
        return xs*1e-24*info[1]*6.022e23/info[0]

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
                molar_mass += (enrich*library['U-235'][0]+(1-enrich)*library['U-238'][0])*jj
            else:
                molar_mass += library[ii][0]*jj
        return molar_mass

    def number_density(compound,counter,molar_mass,density,enrich,library):
        NA = 6.022E23 # Avagadros number
        density_list = []
        for ckk,kk in enumerate(compound):
            if kk == 'U':
                density_list.append(((enrich*density*NA)/library['U-235'][0]*(enrich*library['U-235'][0]+(1-enrich)*library['U-238'][0])/molar_mass)*counter[ckk])
                density_list.append((((1-enrich)*density*NA)/library['U-238'][0]*(enrich*library['U-235'][0]+(1-enrich)*library['U-238'][0])/molar_mass)*counter[ckk])
            else:
                density_list.append(((density*NA)/molar_mass)*counter[ckk])
        return density_list

    def density_list(compound,density,enrich=0.0,library=None):
        import numpy as np
        if library is None:
            import json
            library = json.load(open('discrete1/element_dictionary.json'))

        # Formatting (compounds and count of each)
        compound,counter = chem.cleaning_compound(compound)

        # Calculating molar mass
        molar_mass = chem.total_mass(compound,counter,enrich,library)

        # number densities
        density_list = chem.number_density(compound,counter,molar_mass,density,enrich,library)

        # Convert from barns
        return np.array(density_list)*1e-24

    # counter = [1 if len(re.findall('[0-9][^0-9]*',compound[ii])) == 0 else int(''.join(re.findall('[0-9][^0-9]*',compound[ii]))) for ii in range(len(compound))]
    # compound = [re.sub(r'[0-9]+','',ii) for ii in compound]
    
    
    
class other_tools:
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


