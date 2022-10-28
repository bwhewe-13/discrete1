""" Setting Up Multigroup problems for source.py and hybrid.py """

from .generate import XSGenerate087

import numpy as np
import pkg_resources
import os

DATA_PATH = pkg_resources.resource_filename('discrete1','data/')
XS_PATH = pkg_resources.resource_filename('discrete1','xs/')
C_PATH = pkg_resources.resource_filename('discrete1','c/')

# Problems with same energy grid
group87 = ['stainless','uranium', 'uranium stainless']

class FixedSource:
    # Keyword Arguments allowed currently
    __allowed = ("T","dt","boundary","enrich","hybrid","edges","geometry","td", "xsr","vr")

    def __init__(self,ptype,G,N,**kwargs):
        """ Deals with picking the correct variables needed for the function
        (steady state, time dependent, hybrid)
        Attributes:
            ptype: the preset problem, str
                Options: Reeds, Stainless, Uranium, UraniumStainless
            G: Number of energy groups, int
            N: Number of discrete angles, int
        kwargs:
            T: Length of the time period (for time dependent problems), float
            dt: Width of the time step (for time dependent problems), float
            v: Speed of the neutrons in cm/s, list of length G
            boundary: str (default vacuum), determine RHS of problem
                options: 'vacuum', 'reflected'
            enrich: enrichment percentage of U235 in uranium problems, float
            hybrid: Will return uncollided delta if hybrid == G, collided and 
                splits if hybrid > G
        """
        # Attributes
        self.ptype = ptype
        self.G = G
        self.N = N
        # kwargs
        self.T = None; self.dt = None; # self.v = None
        self.boundary = 'vacuum'; self.enrich = None; self.hybrid = None; 
        self.edges = None; 
        # Testing
        self.xsr = 0; self.vr = 0
        for key, value in kwargs.items():
            assert (key in self.__class__.__allowed), "Attribute not allowed, \
            available: T, dt, boundary, enrich, hybrid, edges" 
            setattr(self, key, value)

    @classmethod
    def initialize(cls,ptype,G,N,**kwargs):
        problem = cls(ptype,G,N,**kwargs)
        td_vars = {}; hy_vars = {}
        # calling steady state variables
        if problem.enrich:
            ss_vars = list(eval(ptype).steady(G,N,problem.boundary,problem.enrich,\
                problem.edges))
        else:
            ss_vars = list(eval(ptype).steady(G,N,problem.boundary,problem.edges))
        # Check if time dependent problem
        if problem.T:
            if ptype in ['Reeds']:
                temp = list(eval(ptype).timed(G,problem.T,problem.dt))
            else: # 87 Group problems
                temp = list(Standard.timed(G,problem.T,problem.dt))
            keys = ['T','dt','v']
            if len(temp[2].shape) == 1:
                spatial_cells = ss_vars[8]
                # temp[2] = np.tile(temp[2],(spatial_cells,1))
            # print('Velocity Shape',temp[2].shape)
            for ii in range(len(keys)):
                td_vars[keys[ii]] = temp[ii]
        # Check if hybrid problem
        if problem.hybrid:
            if ptype in ['Reeds']:
                temp = list(eval(ptype).hybrid(G, problem.hybrid))
            else: # 87 Group problems
                temp = list(Standard.hybrid(G, problem.hybrid))
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
            temp = list(eval(ptype).steady(G,N,problem.boundary,problem.enrich,\
                problem.edges))
        else:
            temp = list(eval(ptype).steady(G,N,problem.boundary,problem.edges))
        # Send to dictionary
        keys = ['G','N','mu','w','total','scatter','fission','source','I', \
            'delta','LHS']
        for ii in range(len(keys)):
            ss_vars[keys[ii]] = temp[ii]
        del temp
        # Check if time dependent problem
        if problem.T:
            if ptype in ['Reeds']:
                temp = list(eval(ptype).timed(G,problem.T,problem.dt))
            else: # 87 Group problems
                temp = list(Standard.timed(G,problem.T,problem.dt))
            keys = ['T','dt','v']
            for ii in range(len(keys)):
                td_vars[keys[ii]] = temp[ii]
        # Check if hybrid problem
        if problem.hybrid:
            if ptype in ['Reeds']:
                temp = list(eval(ptype).hybrid(G,problem.hybrid))
            else: # 87 Group problems
                temp = list(Standard.hybrid(G,problem.hybrid))
            keys = ['delta_e','splits'] 
            for ii in range(len(keys)):
                hy_vars[keys[ii]] = temp[ii]
        return {**ss_vars, **td_vars,**hy_vars}


class Reeds:
    
    def steady(G,N,boundary='vacuum',edges=None):
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
            return delta_u, None
        # Will return uncollided delta_e
        elif hy_g > G:
            splits = Tools.energy_distribution(hy_g,G)
            delta_u = [1/hy_g] * hy_g
            delta_c = [sum(delta_u[ii]) for ii in splits]
            return delta_c, splits


class Standard:
    # Steady is a work in progress
    def steady(G,N,boundary='vacuum',**kwargs):
        # Checking keyword arguments
        edges = kwargs['edges'] if 'edges' in kwargs else None
        enrich = kwargs['enrich'] if 'enrich' in kwargs else None
        # Checking for energy group collapse
        reduced = True if G != 87 else False
        mu,w = np.polynomial.legendre.leggauss(N)
        w /= np.sum(w); 
        if boundary == 'reflected':
            N = int(0.5 * N)
            mu = mu[N:]; w = w[N:]
        energy_grid = np.load(DATA_PATH + 'energy_edges_087G.npy')
        g = np.argmin(abs(energy_grid-14.1E6))
        source = np.zeros((len(energy_grid)-1))
        # source[g] = 1 
        lhs = np.zeros((len(energy_grid)-1))
        print('Not usable yet')

    def timed(G, T, dt):
        try:
            # grid = np.load(DATA_PATH + 'energy_edges_{}G.npy'.format(str(G).zfill(3)))
            grid = np.load(DATA_PATH + "energy_edges_087G.npy")
            idx = Tools.index_generator(87, G)
        except FileNotFoundError:
            grid = np.load(DATA_PATH + 'energy_edges_361G.npy')
            idx = np.load(DATA_PATH + 'group_indices_361G.npz')[str(G).zfill(3)]
        v = Tools.relative_speed(G, grid, idx=idx)
        return T, dt, v

    def hybrid(G, hy_g):
        try:
            # grid = np.load(DATA_PATH + 'energy_edges_{}G.npy'.format(str(G).zfill(3)))
            grid = np.load(DATA_PATH + "energy_edges_087G.npy")
            idx = Tools.index_generator(87, G)
        except FileNotFoundError:
            grid = np.load(DATA_PATH + 'energy_edges_361G.npy')
            idx = np.load(DATA_PATH + 'group_indices_361G.npz')[str(G).zfill(3)]
        if hy_g == G:
            delta_u = np.diff(grid)
            return delta_u, None
        # Will return uncollided delta_e

        elif hy_g > G:
            splits = Tools.energy_distribution(hy_g, G, idx)
            delta_u = np.diff(grid)
            delta_c = [sum(delta_u[ii]) for ii in splits]
            return delta_c, splits


class UraniumStainless: # Slab problem
    _original_groups = 87

    def steady(G,N,boundary='vacuum',enrich=0.2,edges=None):
        """ 
        xsr (cross section reduced)
            xsr = 0: original (no flux, static)
            xsr = 1: flux of 87 group solution at each time step
        vr (velocity reduction)
            vr = 0: original (G vector)
            vr = 1: change velocity at each time step and spatial cell
        """
        # Checking for energy group collapse
        reduced = True if G != UraniumStainless._original_groups else False
        # Setting up shape problem
        shape = [4,2,4]
        materials = ['ss440',['u',enrich],'ss440']
        # Angles
        I = 1000
        delta = sum(shape)/I
        mu,w = np.polynomial.legendre.leggauss(N)
        w /= np.sum(w); 
        # Get list of xs for each material
        xs_total, xs_scatter, xs_fission = Tools.populate_xs_list(materials)
        # Get sizes of each material
        layers = [int(ii/delta) for ii in shape]
        # Source term
        energy_grid = np.load(DATA_PATH + 'energy_edges_087G.npy')
        g = np.argmin(abs(energy_grid-14.1E6))
        external_source = np.zeros((UraniumStainless._original_groups))
        point_source = np.zeros((UraniumStainless._original_groups))
        point_source[g] = 1 # all spatial cells in this group
        if reduced:
            xs_total,xs_scatter,xs_fission = Tools.group_reduction(G,energy_grid,\
                xs_total,xs_scatter,xs_fission,edges)
            external_source = Tools.source_reduction(UraniumStainless._original_groups, \
                                                     G, external_source, edges)
            point_source = Tools.source_reduction(UraniumStainless._original_groups, \
                G, point_source, edges)
        # Propogate onto full space 
        total_,scatter_,fission_ = Tools.populate_full_space(xs_total, xs_scatter, \
                                                             xs_fission, layers)
        point_source_loc = 0
        point_source = [point_source_loc, point_source]
        external_source_ = np.tile(external_source,(I,1))
        Tools.recompile(I)
        return G, N, mu, w, total_, scatter_, fission_, external_source_, I, \
               delta, point_source


class StainlessUranium: # Sphere problem
    _original_groups = 87

    def steady(G,N,boundary='vacuum',enrich=0.2,edges=None):
        # Checking for energy group collapse
        reduced = True if G != StainlessUranium._original_groups else False
        # Setting up shape problem
        # shape = [4,2,4]
        # materials = [['u',enrich],['u',0],'ss440']
        shape = [6, 4, 4, 6]
        materials = [["u", 0.5], ["u", 0.25], ["u", 0.0], "ss440"]
        # Angles
        I = 1000
        delta = sum(shape)/I
        mu,w = np.polynomial.legendre.leggauss(N)
        w /= np.sum(w); 
        # Get list of xs for each material
        xs_total, xs_scatter, xs_fission = Tools.populate_xs_list(materials)
        # Get sizes of each material
        layers = [int(ii/delta) for ii in shape]
        print("layers", layers, np.sum(layers))
        # Source term
        energy_grid = np.load(DATA_PATH + 'energy_edges_087G.npy')
        g = np.argmin(abs(energy_grid-14.1E6))
        external_source = np.zeros((StainlessUranium._original_groups))
        point_source = np.zeros((StainlessUranium._original_groups))
        point_source[g] = 1 # all spatial cells in this group
        if reduced:
            xs_total, xs_scatter, xs_fission = Tools.group_reduction(G, energy_grid,\
                xs_total, xs_scatter, xs_fission, edges)
            external_source = Tools.source_reduction(StainlessUranium._original_groups, \
                                                     G, external_source, edges)
            point_source = Tools.source_reduction(87,G,point_source,edges)
        # Propogate onto full space 
        total_, scatter_, fission_ = Tools.populate_full_space(xs_total, \
                                                xs_scatter, xs_fission, layers)
        point_source_loc = 1000
        point_source = [point_source_loc, point_source]
        external_source_ = np.tile(external_source,(I,1))
        Tools.recompile(I)
        return G, N, mu, w, total_, scatter_, fission_, external_source_, I, \
               delta, point_source


class ControlRod: # Slab problem
    _original_groups = 87

    def steady(G,N,boundary='vacuum',enrich=0.20,edges=None):
        # Checking for energy group collapse
        reduced = True if G != ControlRod._original_groups else False
        # Setting up shape problem
        shape = [10,4,12,4,10]
        materials = ['c',['u',enrich],'c',['u',enrich],'c']
        # Angles
        I = 1000
        delta = sum(shape)/I
        mu,w = np.polynomial.legendre.leggauss(N)
        w /= np.sum(w); 
        # Get list of xs for each material
        xs_total,xs_scatter,xs_fission = Tools.populate_xs_list(materials)
        # Get sizes of each material
        layers = [int(ii/delta) for ii in shape]
        # External and Point Source
        energy_grid = np.load(DATA_PATH + 'energy_edges_087G.npy')
        g = np.argmin(abs(energy_grid-14.1E6))
        external_source = np.zeros((ControlRod._original_groups))
        point_source = np.zeros((ControlRod._original_groups))
        point_source_loc = 0
        if reduced:
            xs_total,xs_scatter,xs_fission = Tools.group_reduction(G,energy_grid,\
                xs_total,xs_scatter,xs_fission,edges)
            # source_ = Tools.source_reduction(87,G,source_.T,edges).T
            source = Tools.source_reduction(ControlRod._original_groups, G, external_source)
            point_source = Tools.source_reduction(ControlRod._original_groups, G, \
                point_source, edges)
        external_source_ = np.tile(external_source,(I,1))
        point_source = [point_source_loc, point_source]
        # Propogate onto full space
        total_, scatter_, fission_ = Tools.populate_full_space(xs_total, xs_scatter, \
            xs_fission, layers)
        Tools.recompile(I)
        return G, N, mu, w, total_, scatter_, fission_, external_source_, I, \
            delta, point_source

    def xs_update(G,enrich,switch,edges=None):
        reduced = True if G != 87 else False
        # Setting up shape problem
        shape = [10,4,12,4,10]
        materials = ['c',['u',enrich],['rod',switch],['u',enrich],'c']
        # Angles
        R = sum(shape); I = 1000
        delta = R/I
        # Get list of xs for each material
        xs_total,xs_scatter,xs_fission = Tools.populate_xs_list(materials)
        # Get sizes of each material
        layers = [int(ii/delta) for ii in shape]
        energy_grid = np.load(DATA_PATH + 'energy_edges_087G.npy')
        if reduced:
            xs_total,xs_scatter,xs_fission = Tools.group_reduction(G,energy_grid,\
                xs_total,xs_scatter,xs_fission,edges)
        # Propogate onto full space 
        total_,scatter_,fission_ = Tools.populate_full_space(xs_total,xs_scatter,xs_fission,layers)
        return total_,scatter_,fission_


class SHEM: # Slab problem
    _original_groups = 361

    def steady(G,N,boundary='vacuum',enrich=0.2,edges=None):
        # Checking for energy group collapse
        reduced = True if G != SHEM._original_groups else False
        # Spatial
        R = 38.364868961274624 # Critical
        R = 38.36485*2 # Critical
        I = 1000
        delta = R/I
        # Angles
        mu,w = np.polynomial.legendre.leggauss(N)
        w /= np.sum(w); 
        # Cross Sections        
        xs_total = np.load(XS_PATH + 'shem/vecTotal.npy')
        xs_scatter = np.load(XS_PATH + 'shem/scatter_000.npy')
        xs_fission = np.load(XS_PATH + 'shem/nufission_000.npy')
        # Point and External Source
        edges_idx = np.load(DATA_PATH + 'group_indices_361G.npz')[str(G).zfill(3)]
        point_source = SHEM.AmBe_source(G, edges_idx)
        point_source_loc = int(0.5 * I)
        # point_source = [point_source_loc, point_source]
        energy_grid = np.load(DATA_PATH + 'energy_edges_361G.npy')
        external_source = np.zeros((SHEM._original_groups))
        if reduced:
            xs_total,xs_scatter,xs_fission = Tools.group_reduction(G, energy_grid,\
                xs_total, xs_scatter, xs_fission, edges_idx)
            external_source = Tools.source_reduction(SHEM._original_groups, G, \
                external_source,edges_idx)
            point_source = Tools.source_reduction(SHEM._original_groups, G, \
                point_source,edges_idx)
        # Propogate onto full space 
        total_ = np.tile(xs_total,(I, 1))
        scatter_ = np.tile(xs_scatter,(I, 1, 1))
        fission_ = np.tile(xs_fission,(I, 1, 1))
        external_source_ = np.tile(external_source, (I, 1))
        external_source_[499] = 0.5 / delta * point_source
        external_source_[500] = 0.5 / delta * point_source
        point_source = [point_source_loc, point_source * 0]
        Tools.recompile(I)
        return G, N, mu, w, total_, scatter_, fission_, external_source_, I, \
                delta, point_source

    def AmBe_source(G, edges_idx):
        AmBe = np.load(DATA_PATH + 'AmBe_source_050G.npz')
        # Make sure to convert energy to MeV
        energy_edges = np.load(DATA_PATH + 'energy_edges_361G.npy') * 1E-6
        # energy_edges = energy_edges[edges_idx].copy()
        energy_centers = 0.5 * (energy_edges[1:] + energy_edges[:-1])
        locs = lambda xmin, xmax: np.argwhere((energy_centers > xmin) & \
                                (energy_centers <= xmax)).flatten()
        source = np.zeros((SHEM._original_groups))
        for center in range(len(AmBe['magnitude'])):
            idx = locs(AmBe['edges'][center], AmBe['edges'][center+1])
            source[idx] = AmBe['magnitude'][center]
        return source


class Tools:

    def energy_distribution(big, small, idx=None):
        """ List of slices for different energy sizes
        Arguments:
            big: uncollided energy groups, int
            small: collided energy groups, int
        Returns:
            list of slices   """
        if idx is None:
            idx = Tools.index_generator(big,small)
        return [slice(ii,jj) for ii,jj in zip(idx[:small],idx[1:])]

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

    def source_reduction(big,small,source,inds=None):
        """ Multiplication factors not used with source 
        Arguments:
            big: the correct size of the matrix, int
            small: the reduced size of the matrix, int
            source: source size (of size (G x 1))
        Returns:
            the reduced vector """
        inds = Tools.index_generator(big,small) if inds is None else inds
        orig_size = np.sum(source)
        new_source = Tools.vector_reduction(source,inds)
        # assert (orig_size == np.sum(new_source))
        return new_source

    def group_reduction_flux(new_group,flux,grid,xs_total,xs_scatter,xs_fission,inds=None):
        """ Used to reduce the number of groups for cross sections and energy levels 
        Arguments:
            new_group: the reduction in the number of groups from the original
            grid: the original energy grid
            energy: True/False to return the new energy grid, default is False
            kwargs: 
                xs_total: total cross section of one spatial cell
                xs_scatter: scatter cross section of one spatial cell
                xs_fission: fission cross section of one spatial cell
        Returns:
            The reduced matrices and vectors of the cross sections or energy
            (Specified in the kwargs)   """
        I = xs_total.shape[0]
        inds = Tools.index_generator(len(flux[0])-1,new_group) if inds is None else inds
        diff_grid = np.diff(grid)
        
        temp_total = Tools.vector_reduction(xs_total.copy()*diff_grid*flux,inds)
        new_total = (temp_total / (Tools.vector_reduction(flux*diff_grid,inds))).T

        # new_total = Tools.vector_reduction(xs_total.copy()*diff_grid*flux,inds).T
        xs_scatter = np.einsum('ijk,ik->ijk',xs_scatter.copy(),flux*diff_grid)
        xs_fission = np.einsum('ijk,ik->ijk',xs_fission.copy(),flux*diff_grid)

        new_scatter = np.zeros((I,new_group,new_group))
        new_fission = np.zeros((I,new_group,new_group))
        # Calculate the indices while including the left-most (insert)
        for ii in range(new_group):
            idx1 = slice(inds[ii],inds[ii+1])
            for jj in range(new_group):
                idx2 = slice(inds[jj],inds[jj+1])
                new_scatter[:,ii,jj] = np.sum(xs_scatter[:,idx1,idx2],axis=(2,1)) / np.sum(flux[:,idx2]*diff_grid[idx2],axis=1)
                new_fission[:,ii,jj] = np.sum(xs_fission[:,idx1,idx2],axis=(2,1)) / np.sum(flux[:,idx2]*diff_grid[idx2],axis=1)
        return new_total, new_scatter, new_fission

    def group_reduction(new_group,grid,xs_total,xs_scatter,xs_fission,inds=None):
        """ Used to reduce the number of groups for cross sections and energy levels 
        Arguments:
            new_group: the reduction in the number of groups from the original
            grid: the original energy grid
            energy: True/False to return the new energy grid, default is False
            kwargs: 
                xs_total: total cross section of one spatial cell
                xs_scatter: scatter cross section of one spatial cell
                xs_fission: fission cross section of one spatial cell
        Returns:
            The reduced matrices and vectors of the cross sections or energy
            (Specified in the kwargs)   """
        total = xs_total.copy()
        scatter = xs_scatter.copy()
        fission = xs_fission.copy()
        # Calculate the indices while including the left-most (insert)
        inds = Tools.index_generator(len(grid)-1,new_group) if inds is None else inds
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
        if len(vector.shape) == 2:
            return np.array([np.sum(vector[:,indices[ii]:indices[ii+1]],axis=1) for ii in range(new_group)])
        return np.array([sum(vector[indices[ii]:indices[ii+1]]) for ii in range(new_group)])

    def classical_speed(grid,G):
        """ Convert energy edges to speed at cell centers, Classical Physics
        Arguments:
            grid: energy edges
            G: number of groups to collapse 
        Returns:
            speeds at cell centers (cm/s)   """
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

    def relative_speed(G, grid=None, idx=None):
        """ Convert energy edges to speed at cell centers, Relative Physics
        Arguments:
            grid: energy edges
            G: number of groups to collapse 
        Returns:
            speeds at cell centers (cm/s)   """
        # Constants
        mass_neutron = 1.67493E-27 # kg
        eV_J = 1.60218E-19 # J
        light = 2.9979246E8 # m/s
        if grid is None:
            grid = np.load(DATA_PATH + 'energy_edges_087G.npy')
        # Calculate velocity for 87 group
        centers = 0.5 * (grid[1:] + grid[:-1])
        if idx is None:
            idx = Tools.index_generator(len(grid)-1, G)
        # This is for new centers (previously used)
        # centers = np.array([float(grid[ii]+grid[jj])*0.5 for ii,jj in zip(idx[:len(grid)-1],idx[1:])])
        gamma = (eV_J * centers)/(mass_neutron * light**2) + 1
        velocity = light/gamma * np.sqrt(gamma**2 - 1) * 100
        # Take mean of collapsed groups
        # velocity *= np.diff(grid)
        # new_velocity = Tools.vector_reduction(velocity, idx)
        # new_velocity /= np.array([grid[idx[ii+1]]-grid[idx[ii]] for ii in range(G)])
        velocity = np.array([np.mean(velocity[idx[ii]:idx[ii+1]]) for ii in range(len(idx)-1)])
        return velocity

    def recompile(I):
        # Recompile cSource
        command = f'gcc -fPIC -shared -o {C_PATH}cSource.so {C_PATH}cSource.c -DLENGTH={I}'
        os.system(command)
        # Recompile cHybrid
        command = f'gcc -fPIC -shared -o {C_PATH}cHybrid.so {C_PATH}cHybrid.c -DLENGTH={I}'
        os.system(command)
        # cSource sphere
        command = f'gcc -fPIC -shared -o {C_PATH}cSourceSP.so {C_PATH}cSourceSP.c -DLENGTH={I}'
        os.system(command)
        # cHybrid sphere
        command = f'gcc -fPIC -shared -o {C_PATH}cHybridSP.so {C_PATH}cHybridSP.c -DLENGTH={I}'
        os.system(command)

    def populate_xs_list(materials):
        """ Populate list with cross sections of different materials """
        xs_total = []; xs_scatter = []; xs_fission = []
        # Iterate through materials list
        for mat in materials:
            # Check for Enrichment
            if type(mat) is list:
                iso = mat[0].upper()
                total_,scatter_,fission_ = XSGenerate087(iso,enrich=mat[1]).cross_section()
            else:
                total_,scatter_,fission_ = XSGenerate087(mat.upper()).cross_section()
            xs_total.append(total_); xs_scatter.append(scatter_); xs_fission.append(fission_)
            del total_, scatter_, fission_
        return xs_total, xs_scatter, xs_fission

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
