""" 
Setting Up Multigroup problems
"""

from .generate import XSGenerate

import numpy as np
import pkg_resources


DATA_PATH = pkg_resources.resource_filename('discrete1','data/')

class Selection:
    

    def select(problem,G,N,**kwargs):
        """ Selects the right class for multigroup problems  """
        # Attributes
        boundary = 'vacuum'
        if 'boundary' in kwargs:
            boundary = kwargs['boundary']
        # Pick the correct class
        if problem == 'reeds':
            pick = Reeds(G,N)
        elif problem == 'stainless infinite':
            pick = StainlessInfinite(G,N)
        # Call for the variables
        items = list(pick.variables())
        # Change N, mu, w if reflected
        if boundary == 'reflected':
            items[2] = items[2][int(N*0.5):] # mu
            items[3] = items[3][int(N*0.5):] # w
            items[1] = int(N*0.5) # N
        return items#, kwargs

    def energy_diff(problem,Gu):
        """ Returns the width of the larger energy list """
        if problem == 'reeds':
            delta_u = [1/Gu] * Gu

        elif problem == 'stainless infinite':
            grid = np.load(DATA_PATH + 'energyGrid.npy')
            delta_u = np.diff(grid)

        return delta_u



class Reeds:
    def __init__(self,G,N):
        self.G = G
        self.N = N

    def variables(self):
        # import numpy as np
        
        L = 0; R = 16.; I = 1000
        mu,w = np.polynomial.legendre.leggauss(self.N)
        w /= np.sum(w); 

        delta = R/I

        boundaries = [slice(0,int(2/delta)),slice(int(2/delta),int(3/delta)),
            slice(int(3/delta),int(5/delta)),slice(int(5/delta),int(6/delta)),
            slice(int(6/delta),int(10/delta)),slice(int(10/delta),int(11/delta)),
            slice(int(11/delta),int(13/delta)),slice(int(13/delta),int(14/delta)),
            slice(int(14/delta),int(16/delta))]
        
        total_ = np.zeros((I,self.G)); total_vals = [10,10,0,5,50,5,0,10,10]
        scatter_ = np.zeros((I,self.G,self.G)); scatter_vals = [9.9,9.9,0,0,0,0,0,9.9,9.9]
        source_ = np.zeros((I,self.G)); source_vals = [0,1,0,0,50,0,0,1,0]

        for ii in range(len(boundaries)):
            total_[boundaries[ii]] = total_vals[ii]
            scatter_[boundaries[ii]] = np.diag(np.repeat(scatter_vals[ii],self.G))
            source_[boundaries[ii]] = source_vals[ii]*1/self.G

        fission_ = np.zeros((scatter_.shape))

        # return self.G,self.N,mu,w,total_[:,None],scatter_[:,None,None],fission_[:,None,None],source_[:,None],I,1/delta
        return self.G,self.N,mu,w,total_,scatter_,fission_,source_,I,1/delta

class FourGroup:
    def __init__(self,G,N):
        self.G = G
        self.N = N

    def variables(self):
        # import numpy as np

        L = 0; R = 5.; I = 1000
        mu,w = np.polynomial.legendre.leggauss(self.N)
        w /= np.sum(w); 

        delta = R/I
        
        sigma_a = np.array([0.00490, 0.00280, 0.03050, 0.12100])
        sigma_ds = np.array([0.08310,0.05850,0.06510])
        D_g = np.array([2.16200,1.08700,0.63200,0.35400])
            
        total_ = np.tile(1/(3*D_g),(I,1))
        down_scat = np.array([(1/(3*D_g[ii]) - sigma_a[ii]) - sigma_ds[ii] for ii in range(self.G-1)])

        scatter_vals = np.diag(down_scat,-1)
        np.fill_diagonal(scatter_vals,1/(3*D_g) - sigma_a)
        scatter_ = np.tile(scatter_vals,(I,1,1))

        source_vals = [1e12,0,0,0]
        source_ = np.tile(source_vals,(I,1))

        fission_ = np.zeros((scatter_.shape))

        return self.G,self.N,mu,w,total_,scatter_,fission_,source_,I,1/delta



class StainlessInfinite:
    def __init__(self,G,N):
        self.G = G
        self.N = N

    def variables(self):
        
        reduced = False
        if self.G != 87:
            reduced = True

        L = 0; R = 10000.; I = 1000
        mu,w = np.polynomial.legendre.leggauss(self.N)
        w /= np.sum(w); 

        delta = R/I

        total,scatter,fission = XSGenerate('SS440').cross_section()
        # Create source in 14.1 MeV group
        energy_grid = np.load(DATA_PATH + 'energyGrid.npy')
        g = np.argmin(abs(energy_grid-14.1E6))
        source = np.zeros((I,self.G))
        source[:,g] = 1

        if reduced:
            total,scatter,fission = Tools.group_reduction(self.G,energy_grid,total,scatter.T,fission.T)

        # Progagate for all groups
        scatter_ = np.tile(scatter,(I,1,1))
        fission_ = np.tile(fission,(I,1,1))
        total_ = np.tile(total,(I,1))
        
        return self.G,self.N,mu,w,total_,scatter_,fission_,source,I,delta

    def scatter_fission(self):
        _,_,_,_,_,scatter,fission,_,_,_ = StainlessInfinite.variables(self)
        return scatter,fission

class Tools:

    def group_reduction(new_group,grid,total,scatter,fission):
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
        new_grid[np.linspace(0,new_group-1,rmdr,dtype=int)] += 1
        assert (new_grid.sum() == old_group)

        # Calculate the indices while including the left-most (insert)
        inds = np.cumsum(np.insert(new_grid,0,0),dtype=int)

        # This is for scaling the new groups properly
        # Calculate the change in energy for each of the new boundaries (of size new_group)
        new_diff_grid = np.array([grid[inds[ii+1]]-grid[inds[ii]] for ii in range(new_group)])
        # Repeated for the matrix (fission and scatter)
        new_diff_matrix = new_diff_grid[:,None] @ new_diff_grid[None,:]
        # Calculate the change in energy for each of the old boundaries
        old_diff_grid = np.diff(grid)
        # Repeated for the matrix (fission and scatter)
        old_diff_matrix = old_diff_grid[:,None] @ old_diff_grid[None,:]
        
        # Total Cross Section
        total *= old_diff_grid
        new_total = Tools.vector_reduction(total,inds)
        new_total /= new_diff_grid

        # Scatter Cross Section
        scatter *= old_diff_grid
        new_scatter = Tools.matrix_reduction(scatter,inds)
        new_scatter /= new_diff_grid

        # Fission Cross Section
        fission *= old_diff_grid
        new_fission = Tools.matrix_reduction(fission,inds)
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
