""" Generating Cross Sections, etc. for Sn code and display """

from .chemistry import NumberDensity

import numpy as np
import pkg_resources

# Used for determining path to cross sections
DATA_PATH = pkg_resources.resource_filename('discrete1','xs/')
ENERGY_PATH = pkg_resources.resource_filename('discrete1','data/')

class XSGenerate087:
    """ Generating the total, fission, scatter cross sections """
    __allowed = ("enrich") # Currently only keyword
    __compounds = ("UH3","HDPE","SS440","U","C","ROD") # Materials that we can use
    __fissioning = ("UH3","U") # Materials that fission
    # Using the first temperature
    __temp = '00'

    def __init__(self,compound,**kwargs):
        assert (compound in self.__class__.__compounds), "Compound not allowed, available: UH3, HDPE, SS440, U"
        self.compound = compound
        # Kwargs
        self.enrich = 0.0; 
        # Make sure user inputs right kwargs
        for key, value in kwargs.items():
            assert (key in self.__class__.__allowed), "Attribute not allowed, available: enrich, energy" 
            setattr(self, key, value)

    def cross_section(self):
        # Dictionary of Number Densities
        nd_dict = NumberDensity(self.compound,enrich=self.enrich).run()
        # Total Cross Section List
        total = [np.load(DATA_PATH + '{}/vecTotal.npy'.format(ii))[eval(self.__class__.__temp)] for ii in nd_dict.keys()]
        total_xs = sum([total[count]*nd for count,nd in enumerate(nd_dict.values())])
        # Scatter Cross Section List
        scatter = [np.load(DATA_PATH + '{}/scatter_0{}.npy'.format(ii,self.__class__.__temp))[0] for ii in nd_dict.keys()]
        scatter_xs = sum([scatter[count]*nd for count,nd in enumerate(nd_dict.values())])
        # Check if it is a fissioning material
        fission_xs = np.zeros((scatter_xs.shape))
        if self.compound in self.__class__.__fissioning:
            fission = [np.load(DATA_PATH + '{}/nufission_0{}.npy'.format(ii,self.__class__.__temp))[0] for ii in nd_dict.keys()]
            fission_xs = sum([fission[count]*nd for count,nd in enumerate(nd_dict.values())])
        return total_xs, scatter_xs.T, fission_xs.T


class XSGenerate618:
    """ Generating the total, fission, scatter cross sections 
    The number densities do not have to be calculated  
    """
    def cross_section(enrich):
        # Scattering
        pu239_scatter = np.load(DATA_PATH + 'pu239/scatter_000.npy')
        pu240_scatter = np.load(DATA_PATH + 'pu240/scatter_000.npy')
        hdpe_scatter = np.load(DATA_PATH + 'hdpe/scatter_000.npy')

        enrich_scatter = pu239_scatter * (1 - enrich) + pu240_scatter * enrich
        scatter_xs = [hdpe_scatter,enrich_scatter,pu240_scatter]
        # Remove excess
        del pu239_scatter, pu240_scatter, hdpe_scatter, enrich_scatter
        # Fission
        pu239_fission = np.load(DATA_PATH + 'pu239/nufission_000.npy')
        pu240_fission = np.load(DATA_PATH + 'pu240/nufission_000.npy')
        hdpe_fission = np.zeros(pu239_fission.shape)
        
        enrich_fission = pu239_fission * (1 - enrich) + pu240_fission * enrich
        fission_xs = [hdpe_fission,enrich_fission,pu240_fission]

        # Remove excess
        del pu239_fission, pu240_fission, hdpe_fission, enrich_fission
        # Total
        pu239_total = np.load(DATA_PATH + 'pu239/vecTotal.npy')
        pu240_total = np.load(DATA_PATH + 'pu240/vecTotal.npy')
        hdpe_total = np.load(DATA_PATH + 'hdpe/vecTotal.npy')

        enrich_total = pu239_total * (1 - enrich) + pu240_total * enrich
        total_xs = [hdpe_total,enrich_total,pu240_total]
        
        # Remove excess
        del pu239_total, pu240_total, hdpe_total, enrich_total
        return total_xs, scatter_xs, fission_xs

    def cross_section_reduce(G,enrich):
        full_bounds = np.load(ENERGY_PATH+'energy_edges_618G.npy')
        full_diff_grid = np.diff(full_bounds)
        try:
            sm_bounds = np.load(ENERGY_PATH+'energy_edges_{}G.npy'.format(G))
            idx = []
            for count,ii in enumerate(sm_bounds):
                try:
                    temp = np.argwhere(ii == full_bounds).flatten()
                    idx.append(temp[0])
                except IndexError:
                    idx.append(np.argmin(abs(ii - full_bounds)))
            idx = np.array(idx)
            sm_diff_grid = np.diff(sm_bounds)
        except OSError:
            idx = ReduceTools.index_generator(618,G)
            sm_diff_grid = np.array([full_bounds[idx[ii+1]]-full_bounds[idx[ii]] for ii in range(G)])

        total,scatter,fission = XSGenerate618.cross_section(enrich)

        new_total = []; new_scatter = []; new_fission = []
        for material in range(len(total)):
            # Total
            temp_total = ReduceTools.vector_reduction(total[material]*full_diff_grid,idx)
            new_total.append(temp_total / sm_diff_grid)

            # Scatter
            temp_scatter = ReduceTools.matrix_reduction(scatter[material]*full_diff_grid,idx)
            new_scatter.append(temp_scatter / sm_diff_grid)

            # Fission
            temp_fission = ReduceTools.matrix_reduction(fission[material]*full_diff_grid,idx)
            new_fission.append(temp_fission / sm_diff_grid)
            del temp_total, temp_scatter, temp_fission

        return new_total,new_scatter,new_fission

    def cross_section_flux(G,enrich,flux=None):
        if flux is None:
            flux = np.load(ENERGY_PATH+'pluto_phi_{}.npy'.format(str(int(enrich*100))))
        full_bounds = np.load(ENERGY_PATH+'energy_edges_618G.npy')
        full_diff_grid = np.diff(full_bounds)
        try:
            sm_bounds = np.load(ENERGY_PATH+'energy_edges_{}G.npy'.format(G))
            idx = []
            for count,ii in enumerate(sm_bounds):
                try:
                    temp = np.argwhere(ii == full_bounds).flatten()
                    idx.append(temp[0])
                except IndexError:
                    idx.append(np.argmin(abs(ii - full_bounds)))
            idx = np.array(idx)
            sm_diff_grid = np.diff(sm_bounds)
        except OSError:
            idx = ReduceTools.index_generator(618,G)
            sm_diff_grid = np.array([full_bounds[idx[ii+1]]-full_bounds[idx[ii]] for ii in range(G)])

        total,scatter,fission = XSGenerate618.cross_section(enrich)

        layers = [500,150,350]
        total,scatter,fission = ReduceTools.populate_full_space(total,scatter,fission,layers)

        I = flux.shape[0]
        new_total = np.zeros((I,G))
        new_scatter = np.zeros((I,G,G))
        new_fission = np.zeros((I,G,G))
        # Calculate the indices while including the left-most (insert)
        inds = ReduceTools.index_generator(len(flux[0])-1,G)
        # This is for scaling the new groups properly
        for ii in range(G):
            idx = slice(inds[ii],inds[ii+1])
            new_total[:,ii] = np.sum(xs_total[:,idx] * flux[:,idx],axis=1) / np.sum(flux[:,idx],axis=1)
            for jj in range(G):
                idx2 = slice(inds[jj],inds[jj+1])
                new_scatter[:,ii,jj] = np.sum(np.einsum('ijk,ik->ij',\
                    xs_scatter[:,idx,idx2],flux[:,idx2]),axis=1) / np.sum(flux[:,idx2],axis=1)
                new_fission[:,ii,jj] = np.sum(np.einsum('ijk,ik->ij',\
                    xs_fission[:,idx,idx2],flux[:,idx2]),axis=1) / np.sum(flux[:,idx2],axis=1)

        return new_total,new_scatter,new_fission

class ReduceTools:
    def index_generator(big,small):
        """  Get the indices for resizing matrices
        Arguments:
            big: larger energy group size, int
            small: smaller energy group size, int
        Returns:
            array of indicies of length small + 1 """
        new_grid = np.ones((small)) * int(big/small)
        new_grid[np.linspace(0,small-1,big % small,dtype=int)] += 1
        assert (new_grid.sum() == big)
        return np.cumsum(np.insert(new_grid,0,0),dtype=int)

    def matrix_reduction(matrix,indices):
        """ Sum the matrix according to the indicies
        Arguments:
            matrix: the full size matrix that will be reduced
            indices: the location of which cells will be combined
        Returns:
            a matrix of size len(indices) - 1   """
        # Remove the extra grid boundary
        new_group = len(indices) - 1
        return np.array([[np.sum(matrix[indices[ii]:indices[ii+1],indices[jj]:indices[jj+1]]) \
            for jj in range(new_group)] for ii in range(new_group)])

    def vector_reduction(vector,indices):
        """ Sum the vector according to the indicies
        Arguments:
            vector: the full size matrix that will be reduced
            indices: the location of which cells will be combined
        Returns:
            a vector of size len(indices) - 1   """
        # Remove the extra grid boundary
        new_group = len(indices) - 1
        # Sum the vector
        return np.array([sum(vector[indices[ii]:indices[ii+1]]) for ii in range(new_group)])

    def populate_full_space(total,scatter,fission,layers):
        """ Populate lists into full space (I)
        total, scatter, fission: lists of cross sections of different materials
        layers: list of cell widths of each material or number of spatial cells 
            of homogeneous material  """
        if isinstance(layers,list):
            total_ = np.vstack([np.tile(total[count],(width,1)) for count,width in enumerate(layers)])
            scatter_ = np.vstack([np.tile(scatter[count],(width,1,1)) for count,width in enumerate(layers)])
            fission_ = np.vstack([np.tile(fission[count],(width,1,1)) for count,width in enumerate(layers)])
        else: # layers is number of spatial cells
            total_ = np.tile(total,(layers,1))
            scatter_ = np.tile(scatter,(layers,1,1))
            fission_ = np.tile(fission,(layers,1,1))
        return total_,scatter_,fission_
