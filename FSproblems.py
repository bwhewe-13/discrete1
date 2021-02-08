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
        boundary = 'vacuum'; enrich = 0.007
        if 'boundary' in kwargs:
            boundary = kwargs['boundary']

        # Pick the correct class
        if problem == 'reeds':
            pick = Reeds(G,N,boundary=boundary)
        elif problem == 'stainless infinite':
            pick = StainlessInfinite(G,N)
        elif problem == 'uranium infinite':
            pick = UraniumInfinite(G,N,enrich=enrich)
        # Call for the variables
        items = list(pick.variables())
        
        # Change N, mu, w if reflected
        if boundary == 'reflected':
            items[2] = items[2][int(N*0.5):] # mu
            items[3] = items[3][int(N*0.5):] # w
            items[1] = int(N*0.5) # N
        return items, kwargs

    def energy_diff(problem,Gu):
        """ Returns the width of the larger energy list """
        if problem == 'reeds':
            delta_u = [1/Gu] * Gu

        elif problem in ['stainless infinite','uranium infinite']:
            grid = np.load(DATA_PATH + 'energyGrid.npy')
            delta_u = np.diff(grid)

        return delta_u

    def speed_calc(problem,Gu):
        if problem == 'reeds':
            v = np.ones((Gu))

        elif problem in ['stainless infinite', 'uranium infinite']:
            grid = np.load(DATA_PATH + 'energyGrid.npy')
            v = Tools.relative_speed(grid,Gu)
            # v = Tools.classical_speed(grid,Gu)

        return v


class Reeds:
    __allowed = ("boundary")

    def __init__(self,G,N,**kwargs):
        self.G = G
        self.N = N
        self.boundary = 'vacuum'; 
        for key, value in kwargs.items():
            assert (key in self.__class__.__allowed), "Attribute not allowed, available: boundary, time" 
            setattr(self, key, value)

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

        if self.boundary == 'reflected':
            R = 8.; delta = R/I
            boundaries = [slice(0,int(2/delta)),slice(int(2/delta),int(3/delta)),
                slice(int(3/delta),int(5/delta)),slice(int(5/delta),int(6/delta)),
                slice(int(6/delta),int(8/delta))]

            total_vals = total_vals[:5].copy()
            scatter_vals = scatter_vals[:5].copy()
            source_vals = source_vals[:5].copy()

        for ii in range(len(boundaries)):
            total_[boundaries[ii]] = total_vals[ii]
            scatter_[boundaries[ii]] = np.diag(np.repeat(scatter_vals[ii],self.G))
            source_[boundaries[ii]] = source_vals[ii]*1/self.G

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

        L = 0; R = 1000.; I = 1000
        mu,w = np.polynomial.legendre.leggauss(self.N)
        w /= np.sum(w); 

        delta = R/I
        total,scatter,fission = XSGenerate('SS440').cross_section()

        # Create source in 14.1 MeV group
        energy_grid = np.load(DATA_PATH + 'energyGrid.npy')
        g = np.argmin(abs(energy_grid-14.1E6))
        source = np.zeros((len(energy_grid)-1))
        source[g] = 1 # all spatial cells in this group

        if reduced:
            total,scatter,fission = Tools.group_reduction(self.G,energy_grid,total,scatter,fission)
            source = Tools.group_reduction(self.G,energy_grid,source,None,None)

        # Progagate for all groups
        scatter_ = np.tile(scatter,(I,1,1))
        fission_ = np.tile(fission,(I,1,1))
        total_ = np.tile(total,(I,1))
        source = np.tile(source,(I,1))

        return self.G,self.N,mu,w,total_,scatter_,fission_,source,I,delta

    def scatter_fission(self):
        _,_,_,_,_,scatter,fission,_,_,_ = StainlessInfinite.variables(self)
        return scatter,fission

class UraniumInfinite:
    def __init__(self,G,N,enrich=0.0):
        self.G = G
        self.N = N
        self.enrich = enrich

    def variables(self):

        reduced = False
        if self.G != 87:
            reduced = True

        L = 0; R = 1000.; I = 1000
        mu,w = np.polynomial.legendre.leggauss(self.N)
        w /= np.sum(w); 

        delta = R/I

        total,scatter,fission = XSGenerate('U',enrich=self.enrich).cross_section()

        # Create source in 14.1 MeV group
        energy_grid = np.load(DATA_PATH + 'energyGrid.npy')
        g = np.argmin(abs(energy_grid-14.1E6))
        source = np.zeros((len(energy_grid)-1))
        source[g] = 1 # all spatial cells in this group

        if reduced:
            total,scatter,fission = Tools.group_reduction(self.G,energy_grid,total,scatter,fission)
            source = Tools.group_reduction(self.G,energy_grid,source,None,None)

        # Progagate for all groups
        scatter_ = np.tile(scatter,(I,1,1))
        fission_ = np.tile(fission,(I,1,1))
        total_ = np.tile(total,(I,1))
        source = np.tile(source,(I,1))

        # source[0,g] = 1

        return self.G,self.N,mu,w,total_,scatter_,fission_,source,I,delta

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

        # For source problem
        if scatter is None and fission is None:
            return new_total

        # Scatter Cross Section
        scatter *= old_diff_grid
        new_scatter = Tools.matrix_reduction(scatter,inds)
        new_scatter /= new_diff_grid
        # new_scatter /= new_diff_matrix

        # Fission Cross Section
        fission *= old_diff_grid
        new_fission = Tools.matrix_reduction(fission,inds)
        new_fission /= new_diff_grid
        # new_fission /= new_diff_matrix

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

    def classical_speed(grid,G):
        """ Convert energy edges to speed at cell centers, Classical Physics
        Arguments:
            grid: energy edges
            G: number of groups to collapse 
        Returns:
            speeds at cell centers (cm/s)
        """
        mass_neutron = 1.67493E-27 # kg
        eV_J = 1.60218E-19 # J

        if len(grid) - 1 == G:
            centers = np.array([float(grid[ii]+grid[jj])*0.5 for ii,jj in zip(range(len(grid)-1),range(1,len(grid)))])
        else:
            old_group = len(grid) - 1            
            # Create array showing the number of groups combined 
            new_grid = np.ones((G)) * int(old_group/G)
            # Add the remainder groups to the first x number 
            new_grid[np.linspace(0,G-1,old_group % G,dtype=int)] += 1

            assert (new_grid.sum() == old_group)
            # Calculate the indices while including the left-most (insert)
            inds = np.cumsum(np.insert(new_grid,0,0),dtype=int)
            centers = np.array([float(grid[ii]+grid[jj])*0.5 for ii,jj in zip(inds[:len(grid)-1],inds[1:])])

        v = np.sqrt((2 * eV_J * centers)/mass_neutron) * 100
        return v

    def relative_speed(grid,G):
        """ Convert energy edges to speed at cell centers, Relative Physics
        Arguments:
            grid: energy edges
            G: number of groups to collapse 
        Returns:
            speeds at cell centers (cm/s)
        """
        mass_neutron = 1.67493E-27 # kg
        eV_J = 1.60218E-19 # J
        light = 2.9979246E8 # m/s

        if len(grid) - 1 == G:
            centers = np.array([float(grid[ii]+grid[jj])*0.5 for ii,jj in zip(range(len(grid)-1),range(1,len(grid)))])
        else:
            old_group = len(grid) - 1            
            # Create array showing the number of groups combined 
            new_grid = np.ones((G)) * int(old_group/G)
            # Add the remainder groups to the first x number 
            new_grid[np.linspace(0,G-1,old_group % G,dtype=int)] += 1

            assert (new_grid.sum() == old_group)
            # Calculate the indices while including the left-most (insert)
            inds = np.cumsum(np.insert(new_grid,0,0),dtype=int)
            centers = np.array([float(grid[ii]+grid[jj])*0.5 for ii,jj in zip(inds[:len(grid)-1],inds[1:])])

        gamma = (eV_J * centers)/(mass_neutron * light**2) + 1
        v = light/gamma * np.sqrt(gamma**2 - 1) * 100
        return v



