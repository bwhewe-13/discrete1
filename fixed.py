""" 
Setting Up Multigroup problems for source.py and hybrid.py
"""

from .generate import XSGenerate

import numpy as np
import pkg_resources
import os


DATA_PATH = pkg_resources.resource_filename('discrete1','data/')

# Problems with same energy grid
group87 = ['stainless infinite','stainless','uranium infinite', 'uranium stainless']

class FixedSource:
    # Keyword Arguments allowed currently
    __allowed = ("T","dt","boundary","enrich","hybrid","edges")

    def __init__(self,ptype,G,N,**kwargs):
        """ Deals with picking the correct variables needed for the function
        (steady state, time dependent, hybrid)
        Attributes:
            ptype: the preset problem, str
                Options: Reeds, Stainless, StainlessInfinite, UraniumInfinite, UraniumStainless
            G: Number of energy groups, int
            N: Number of discrete angles, int
        kwargs:
            T: Length of the time period (for time dependent problems), float
            dt: Width of the time step (for time dependent problems), float
            v: Speed of the neutrons in cm/s, list of length G
            boundary: str (default vacuum), determine RHS of problem
                options: 'vacuum', 'reflected'
            enrich: enrichment percentage of U235 in uranium problems, float
            hybrid: Will return uncollided delta if hybrid == G, collided and splits if hybrid > G
        """
        # Attributes
        self.ptype = ptype
        self.G = G
        self.N = N
        # kwargs
        self.T = None; self.dt = None; # self.v = None
        self.boundary = 'vacuum'; self.enrich = None; self.hybrid = None; self.edges = None
        for key, value in kwargs.items():
            assert (key in self.__class__.__allowed), "Attribute not allowed, available: T, dt, boundary, enrich, hybrid, edges" 
            setattr(self, key, value)

    @classmethod
    def initialize(cls,ptype,G,N,**kwargs):
        problem = cls(ptype,G,N,**kwargs)
        td_vars = {}; hy_vars = {}
        # calling steady state variables
        if problem.enrich:
            ss_vars = list(eval(ptype).steady(G,N,problem.boundary,problem.enrich,problem.edges))
        else:
            ss_vars = list(eval(ptype).steady(G,N,problem.boundary,problem.edges))

        # Check if time dependent problem
        if problem.T:
            temp = list(eval(ptype).timed(G,problem.T,problem.dt))
            keys = ['T','dt','v']
            for ii in range(len(keys)):
                td_vars[keys[ii]] = temp[ii]

        # Check if hybrid problem
        if problem.hybrid:
            temp = list(eval(ptype).hybrid(G,problem.hybrid))
            keys = ['delta_e','splits'] # collided
            for ii in range(len(keys)):
                hy_vars[keys[ii]] = temp[ii]

        return ss_vars, {**td_vars,**hy_vars}

    @classmethod
    def dictionary(cls,ptype,G,N,**kwargs):
        problem = cls(ptype,G,N,**kwargs)
        ss_vars = {}; td_vars = {}; hy_vars = {}
        # calling steady state variables
        if problem.enrich:
            temp = list(eval(ptype).steady(G,N,problem.boundary,problem.enrich))
        else:
            temp = list(eval(ptype).steady(G,N,problem.boundary))
        # Send to dictionary
        keys = ['G','N','mu','w','total','scatter','fission','source','I','delta','LHS']
        for ii in range(len(keys)):
            ss_vars[keys[ii]] = temp[ii]
        del temp
        # Check if time dependent problem
        if problem.T:
            temp = list(eval(ptype).timed(G,problem.T,problem.dt))
            keys = ['T','dt','v']
            for ii in range(len(keys)):
                td_vars[keys[ii]] = temp[ii]

        # Check if hybrid problem
        if problem.hybrid:
            temp = list(eval(ptype).hybrid(G,problem.hybrid))
            keys = ['delta_e','splits'] 
            for ii in range(len(keys)):
                hy_vars[keys[ii]] = temp[ii]

        return {**ss_vars, **td_vars,**hy_vars}

class Reeds:
    
    def steady(G,N,boundary='vacuum'):
        
        L = 0; R = 16.; I = 1000
        mu,w = np.polynomial.legendre.leggauss(N)
        w /= np.sum(w); 

        delta = R/I

        boundaries = [slice(0,int(2/delta)),slice(int(2/delta),int(3/delta)),
            slice(int(3/delta),int(5/delta)),slice(int(5/delta),int(6/delta)),
            slice(int(6/delta),int(10/delta)),slice(int(10/delta),int(11/delta)),
            slice(int(11/delta),int(13/delta)),slice(int(13/delta),int(14/delta)),
            slice(int(14/delta),int(16/delta))]
        
        total_ = np.zeros((I,G)); total_vals = [10,10,0,5,50,5,0,10,10]
        scatter_ = np.zeros((I,G,G)); scatter_vals = [9.9,9.9,0,0,0,0,0,9.9,9.9]
        source_ = np.zeros((I,G)); source_vals = [0,1,0,0,50,0,0,1,0]
        lhs = np.zeros(G)

        if boundary == 'reflected':
            R = 8.; delta = R/I
            boundaries = [slice(0,int(2/delta)),slice(int(2/delta),int(3/delta)),
                slice(int(3/delta),int(5/delta)),slice(int(5/delta),int(6/delta)),
                slice(int(6/delta),int(8/delta))]

            total_vals = total_vals[:5].copy()
            scatter_vals = scatter_vals[:5].copy()
            source_vals = source_vals[:5].copy()

            N = int(0.5 * N)
            mu = mu[N:]; w = w[N:]

        for ii in range(len(boundaries)):
            total_[boundaries[ii]] = total_vals[ii]
            scatter_[boundaries[ii]] = np.diag(np.repeat(scatter_vals[ii],G))
            source_[boundaries[ii]] = source_vals[ii]*1/G

        fission_ = np.zeros((scatter_.shape))

        Tools.recompile(I)

        return G,N,mu,w,total_,scatter_,fission_,source_,I,delta,lhs

    def timed(G,T,dt):
        v = np.ones((G))
        return T, dt, v

    def hybrid(G,hy_g):
        # Will return 
        if hy_g == G:
            delta_u = [1/G] * G
            return delta_u,None
        # Will return uncollided delta_e
        elif hy_g > G:
            splits = Tools.energy_distribution(hy_g,G)
            delta_u = [1/hy_g] * hy_g
            delta_c = [sum(delta_u[ii]) for ii in splits]
            return delta_c, splits


class StainlessInfinite:

    def steady(G,N,boundary='vacuum'):
        reduced = False
        if G != 87:
            reduced = True

        L = 0; R = 1000.; I = 1000
        mu,w = np.polynomial.legendre.leggauss(N)
        w /= np.sum(w); 

        if boundary == 'reflected':
            N = int(0.5 * N)
            mu = mu[N:]; w = w[N:]

        delta = R/I
        total,scatter,fission = XSGenerate('SS440').cross_section()

        # Create source in 14.1 MeV group
        energy_grid = np.load(DATA_PATH + 'energyGrid.npy')
        g = np.argmin(abs(energy_grid-14.1E6))
        source = np.zeros((len(energy_grid)-1))
        source[g] = 1 # all spatial cells in this group
        lhs = np.zeros((len(energy_grid)-1))

        if reduced:
            total,scatter,fission = Tools.group_reduction(G,energy_grid,total,scatter,fission)
            source = Tools.source_reduction(87,G,source)
            lhs = Tools.source_reduction(87,G,lhs)

        scatter_ = np.tile(scatter,(I,1,1))
        fission_ = np.tile(fission,(I,1,1))
        total_ = np.tile(total,(I,1))
        source_ = np.tile(source,(I,1))

        Tools.recompile(I)

        return G,N,mu,w,total_,scatter_,fission_,source_,I,delta,lhs

    def timed(G,T,dt):
        grid = np.load(DATA_PATH + 'energyGrid.npy')
        v = Tools.relative_speed(grid,G)
        return T, dt, v

    def hybrid(G,hy_g):
        grid = np.load(DATA_PATH + 'energyGrid.npy')
        if hy_g == G:
            delta_u = np.diff(grid)
            return delta_u,None
        # Will return uncollided delta_e
        elif hy_g > G:
            splits = Tools.energy_distribution(hy_g,G)
            delta_u = np.diff(grid)
            delta_c = [sum(delta_u[ii]) for ii in splits]
            return delta_c, splits


class Stainless:

    def steady(G,N,boundary='vacuum',edges=None):
        reduced = False
        if G != 87:
            reduced = True

        L = 0; R = 10.; I = 1000
        mu,w = np.polynomial.legendre.leggauss(N)
        w /= np.sum(w); 

        if boundary == 'reflected':
            N = int(0.5 * N)
            mu = mu[N:]; w = w[N:]

        delta = R/I
        total,scatter,fission = XSGenerate('SS440').cross_section()

        # Create source in 14.1 MeV group
        energy_grid = np.load(DATA_PATH + 'energyGrid.npy')
        g = np.argmin(abs(energy_grid-14.1E6))
        source = np.zeros((len(energy_grid)-1))
        lhs = np.zeros((len(energy_grid)-1))
        lhs[g] = 1 # all spatial cells in this group

        if reduced:
            total,scatter,fission = Tools.group_reduction(G,energy_grid,total,scatter,fission,edges)
            source = Tools.source_reduction(87,G,source,edges)
            lhs = Tools.source_reduction(87,G,lhs,edges)


        scatter_ = np.tile(scatter,(I,1,1))
        fission_ = np.tile(fission,(I,1,1))
        total_ = np.tile(total,(I,1))
        source_ = np.tile(source,(I,1))

        # source_[1:] *= 0 # Entering from LHS        
        Tools.recompile(I)

        return G,N,mu,w,total_,scatter_,fission_,source_,I,delta,lhs

    def timed(G,T,dt):
        grid = np.load(DATA_PATH + 'energyGrid.npy')
        v = Tools.relative_speed(grid,G)
        return T, dt, v

    def hybrid(G,hy_g):
        grid = np.load(DATA_PATH + 'energyGrid.npy')
        if hy_g == G:
            delta_u = np.diff(grid)
            return delta_u,None
        # Will return uncollided delta_e
        elif hy_g > G:
            splits = Tools.energy_distribution(hy_g,G)
            delta_u = np.diff(grid)
            delta_c = [sum(delta_u[ii]) for ii in splits]
            return delta_c, splits


class UraniumInfinite:

    def steady(G,N,boundary='vacuum',enrich=0.007):

        reduced = False
        if G != 87:
            reduced = True

        L = 0; R = 1000.; I = 1000
        mu,w = np.polynomial.legendre.leggauss(N)
        w /= np.sum(w); 

        delta = R/I

        total,scatter,fission = XSGenerate('U',enrich=enrich).cross_section()

        # Create source in 14.1 MeV group
        energy_grid = np.load(DATA_PATH + 'energyGrid.npy')
        g = np.argmin(abs(energy_grid-14.1E6))
        source = np.zeros((len(energy_grid)-1))
        source[g] = 1 # all spatial cells in this group
        lhs = np.zeros((len(enegyr_grid)-1))

        if reduced:
            total,scatter,fission = Tools.group_reduction(G,energy_grid,total,scatter,fission)
            source = Tools.source_reduction(87,G,source)
            lhs = Tools.source_reduction(87,G,lhs)

        # Progagate for all groups
        scatter_ = np.tile(scatter,(I,1,1))
        fission_ = np.tile(fission,(I,1,1))
        total_ = np.tile(total,(I,1))
        source_ = np.tile(source,(I,1))

        Tools.recompile(I)

        return G,N,mu,w,total_,scatter_,fission_,source_,I,delta,lhs

    def timed(G,T,dt):
        grid = np.load(DATA_PATH + 'energyGrid.npy')
        v = Tools.relative_speed(grid,G)
        return T, dt, v

    def hybrid(G,hy_g):
        grid = np.load(DATA_PATH + 'energyGrid.npy')
        if hy_g == G:
            delta_u = np.diff(grid)
            return delta_u,None
        # Will return uncollided delta_e
        elif hy_g > G:
            splits = Tools.energy_distribution(hy_g,G)
            delta_u = np.diff(grid)
            delta_c = [sum(delta_u[ii]) for ii in splits]
            return delta_c, splits


class UraniumStainless:

    def steady(G,N,boundary='vacuum',enrich=0.2,edges=None):
        shape = [4,2,4]
        materials = ['ss440',['u',enrich],'ss440']

        reduced = False
        if G != 87:
            reduced = True

        L = 0; R = sum(shape); I = 1000
        mu,w = np.polynomial.legendre.leggauss(N)
        w /= np.sum(w); 

        delta = R/I

        # Get list of xs for each material
        xs_total,xs_scatter,xs_fission = Tools.populate_xs_list(materials)
        # Get sizes of each material
        layers = [int(ii/delta) for ii in shape]

        # Source term
        energy_grid = np.load(DATA_PATH + 'energyGrid.npy')
        g = np.argmin(abs(energy_grid-14.1E6))
        source = np.zeros((len(energy_grid)-1))
        lhs = np.zeros((len(energy_grid)-1))
        lhs[g] = 1 # all spatial cells in this group

        if reduced:
            xs_total,xs_scatter,xs_fission = Tools.group_reduction(G,energy_grid,xs_total,xs_scatter,xs_fission,edges)
            source = Tools.source_reduction(87,G,source,edges)
            lhs = Tools.source_reduction(87,G,lhs,edges)

        # Propogate onto full space 
        total_,scatter_,fission_ = Tools.populate_full_space(xs_total,xs_scatter,xs_fission,layers)

        source_ = np.tile(source,(I,1))
        # source_[1:] *= 0

        Tools.recompile(I)

        return G,N,mu,w,total_,scatter_,fission_,source_,I,delta,lhs

    def timed(G,T,dt):
        grid = np.load(DATA_PATH + 'energyGrid.npy')
        v = Tools.relative_speed(grid,G)
        return T, dt, v

    def hybrid(G,hy_g):
        grid = np.load(DATA_PATH + 'energyGrid.npy')
        if hy_g == G:
            delta_u = np.diff(grid)
            return delta_u, None
        # Will return uncollided delta_e
        elif hy_g > G:
            splits = Tools.energy_distribution(hy_g,G)
            delta_u = np.diff(grid)
            delta_c = [sum(delta_u[ii]) for ii in splits]
            return delta_c, splits


class Tools:

    def energy_distribution(big,small):
        """ List of slices for different energy sizes
        Arguments:
            big: uncollided energy groups, int
            small: collided energy groups, int
        Returns:
            list of slices   """
        inds = Tools.index_generator(big,small)

        splits = [slice(ii,jj) for ii,jj in zip(inds[:small],inds[1:])]
        return splits

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

        inds = np.cumsum(np.insert(new_grid,0,0),dtype=int)

        return inds

    def index_generator_cherry(big,small,total):
        # # Get the difference for the total cross section
        # difference = abs(np.diff(total))
        # # Find locations of change below threshold 
        # min_change = np.argwhere(difference < np.sort(difference)[big-small-1]).flatten()
        # # Combine consecutive terms
        # inds = np.array([ii for ii in range(big) if ii not in min_change])
        # inds[-1] = 87

        new_grid = np.ones((small)) * int(big/small)
        new_grid[np.linspace(0,small-1,big % small,dtype=int)] += 1

        new_grid = np.sort(new_grid)
        inds = np.cumsum(np.insert(new_grid,0,0),dtype=int)

        return inds


    def source_reduction(big,small,source,inds=None):
        """ Multiplication factors not used with source 
        Arguments:
            big: the correct size of the matrix, int
            small: the reduced size of the matrix, int
            source: source size (of size (G x 1))
        Returns:
            the reduced vector """
        if inds is None:
            inds = Tools.index_generator(big,small)
        orig_size = np.sum(source)

        new_source = Tools.vector_reduction(source,inds)
        assert (orig_size == np.sum(new_source))

        return new_source

    def group_reduction(new_group,grid,total,scatter,fission,inds=None):
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

        # Calculate the indices while including the left-most (insert)
        if inds is None:
            inds = Tools.index_generator(len(grid)-1,new_group)
        
        # inds = Tools.index_generator_cherry(len(grid)-1,new_group,total)

        # nums = np.arange(1,87)
        # np.random.shuffle(nums); np.random.shuffle(nums)
        # inds = np.sort(nums[:new_group-1])

        # inds = np.insert(inds,0,0); 
        # inds = np.append(inds,87)

        # np.save('temporary_indexing',inds)

        # This is for scaling the new groups properly
        # Calculate the change in energy for each of the new boundaries (of size new_group)
        new_diff_grid = np.array([grid[inds[ii+1]]-grid[inds[ii]] for ii in range(new_group)])
        # Calculate the change in energy for each of the old boundaries
        old_diff_grid = np.diff(grid)

        # Check for list of materials
        if type(total) == list:
            new_total = []; new_scatter = []; new_fission = []
            for mat in range(len(total)):
                # Total
                temp_total = Tools.vector_reduction(total[mat]*old_diff_grid,inds)
                new_total.append(temp_total / new_diff_grid)
                # Scatter
                temp_scatter = Tools.matrix_reduction(scatter[mat]*old_diff_grid,inds)
                new_scatter.append(temp_scatter / new_diff_grid)
                # Fission
                temp_fission = Tools.matrix_reduction(fission[mat]*old_diff_grid,inds)
                new_fission.append(temp_fission / new_diff_grid)
                del temp_total, temp_scatter, temp_fission

            return new_total,new_scatter,new_fission
        
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

    def recompile(I):
        # Recompile cSource
        command = 'gcc -fPIC -shared -o {}cSource.so {}cSource.c -DLENGTH={}'.format(DATA_PATH,DATA_PATH,I)
        os.system(command)
        # Recompile cHybrid
        command = 'gcc -fPIC -shared -o {}cHybrid.so {}cHybrid.c -DLENGTH={}'.format(DATA_PATH,DATA_PATH,I)
        os.system(command)

    def populate_xs_list(materials):
        """ Populate list with cross sections of different materials """
        xs_total = []; xs_scatter = []; xs_fission = []
        # Iterate through materials list
        for mat in materials:
            # Check for Enrichment
            if type(mat) is list:
                iso = mat[0].upper()
                total_,scatter_,fission_ = XSGenerate(iso,enrich=mat[1]).cross_section()
            else:
                total_,scatter_,fission_ = XSGenerate(mat.upper()).cross_section()
            xs_total.append(total_); xs_scatter.append(scatter_); xs_fission.append(fission_)
            del total_, scatter_, fission_
        return xs_total, xs_scatter, xs_fission

    def populate_full_space(total,scatter,fission,layers):
        """ Populate lists into full space (I)
        total, scatter, fission: lists of cross sections of different materials
        layers: list of cell widths of each material
        """
        total_ = np.vstack([np.tile(total[count],(width,1)) for count,width in enumerate(layers)])
        scatter_ = np.vstack([np.tile(scatter[count],(width,1,1)) for count,width in enumerate(layers)])
        fission_ = np.vstack([np.tile(fission[count],(width,1,1)) for count,width in enumerate(layers)])

        return total_,scatter_,fission_

   