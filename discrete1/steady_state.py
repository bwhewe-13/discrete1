################################################################################
#
# Steady State
#
#
#
################################################################################

from discrete1.utils import transport_tools

import numpy as np
import ctypes
import pkg_resources

C_PATH = pkg_resources.resource_filename('discrete1','c/')

INNER_TOLERANCE = 1E-12
OUTER_TOLERANCE = 1E-8
MAX_ITERATIONS = 100

class Dime:

    def __init__(self, groups, cells, angles, total, scatter, fission, source, \
                    boundary, cell_width, material_map, geometry):
        """ Class for running one-dimensional steady-state multigroup problem.

        .. class: Tide inherits this class.
        .. class: Keg inherits this class.

        Args:
            groups (int): Number of energy groups.
            cells (int): Number of spatial cells.
            angles (int): Number of discrete angles.
            total (array of size (materials x groups)): Total cross section for
                each material.
            scatter (array of size (materials x groups x groups)): Scattering
                cross section matrix for each material.
            fission (array of size (materials x groups x groups)): Fission 
                cross section matrix for each material.
            source (array of size (cells x groups)): Source for each spatial
                cell and energy level.
            boundary (array of size (groups x 2)): [left, right] boundary 
                conditions for each energy level.
            cell_width (float): width of each spatial cell.
            material_map (array of size (cells x 1)): Location of each of the 
                materials in the problem.
            geometry (str): Determines the coordinates of the one-dimensional
                sweep. Either 'slab' or 'sphere'.

        Kwargs:
            track: bool (default False), if track flux change with iteration

        """
        self.groups = groups
        self.cells = cells
        self.angles = angles
        self.total = total
        self.scatter = scatter
        self.fission = fission
        self.source = source
        self.boundary = boundary
        self.cell_width = 1/cell_width
        self.material_map = material_map
        self.geometry = geometry
               
    def slab(self,total_g,scatter_gg,source_g,boundary_g,scalar_flux_guess,\
                angular_flux_last=None):
        """ Function for running mono-energetic transport sweeps in one-
        dimensional slabs.

        Args:
            total_g (array of size (materials x 1)): Total cross section for 
                each material at group g
            scatter_gg (array of size (materials x 1)): Self-scattering cross 
                section for each material at group g
            source_g (array of size (materials x 1)): Summation of non self-
                scattering, fission, and external sources for group g
            boundary_g (list of size 2): boundary conditions for the (left, right)
                boundaries for group g
            scalar_flux_guess (array of size (cells x 1)): Initial guess of the
                scalar flux for group g
            angular_flux_last (array of size (cells x angles)): Angular flux
                of the previous time step, used in time_dependent.Tide

        Returns:
            scalar_flux_g: (array of size (cells x 1)): Calculated Scalar flux
                for group g

        """
        clibrary = ctypes.cdll.LoadLibrary(C_PATH + 'sweep1d.so')
        sweep = clibrary.reflected if np.sum(boundary_g) > 0 else clibrary.vacuum
        if np.sum(boundary_g) > 0:
            sweep = clibrary.reflected
            # print('Reflected')
        else:
            sweep = clibrary.vacuum
            # print(boundary_g,np.sum(boundary_g))
            # print('Vacuum')

        scalar_flux_old = scalar_flux_guess.copy()

        if angular_flux_last is not None:
            angular_flux_next = np.zeros(angular_flux_last.shape,dtype='float64')

        converged = 0
        count = 1
        while not (converged):
            scalar_flux_g = np.zeros((self.cells),dtype='float64')
            for n in range(self.corrected_angles):
                direction = ctypes.c_int(int(np.sign(self.mu[n])))
                weight = np.sign(self.mu[n]) * self.mu[n] * self.cell_width

                angular_flux_g = np.zeros((self.cells),dtype='float64')
                angular_flux_ptr = ctypes.c_void_p(angular_flux_g.ctypes.data)

                if angular_flux_last is None:
                    top_mult = (weight - 0.5 * total_g).astype('float64')
                    bottom_mult = (1/(weight + 0.5 * total_g)).astype('float64')
                    rhs = source_g.astype('float64')
                else:
                    top_mult = (weight - 0.5 * total_g - self.time_const * self.speed).astype('float64')
                    bottom_mult = (1/(weight + 0.5 * total_g + self.time_const * self.speed)).astype('float64')
                    rhs = (source_g + angular_flux_last[:,n] * self.speed).astype('float64')

                top_ptr = ctypes.c_void_p(top_mult.ctypes.data)
                bot_ptr = ctypes.c_void_p(bottom_mult.ctypes.data)
                rhs_ptr = ctypes.c_void_p(rhs.ctypes.data)

                temp_scat = (scatter_gg * scalar_flux_old).astype('float64')
                ts_ptr = ctypes.c_void_p(temp_scat.ctypes.data)
                
                scalar_flux_ptr = ctypes.c_void_p(scalar_flux_g.ctypes.data)
                
                sweep(scalar_flux_ptr, angular_flux_ptr, ts_ptr, rhs_ptr, top_ptr, \
                    bot_ptr, ctypes.c_double(self.w[n]), direction)

                if angular_flux_last is not None:
                    angular_flux_next[:,n] = angular_flux_g.copy()

            change = np.linalg.norm((scalar_flux_g - scalar_flux_old) \
                                     /scalar_flux_g/(self.cells))
            # if np.isnan(change) or np.isinf(change):
            #     change = 0.
            converged = (change < INNER_TOLERANCE) or (count >= MAX_ITERATIONS) 
            count += 1
            scalar_flux_old = scalar_flux_g.copy()
        if angular_flux_last is not None:
            return scalar_flux_g, angular_flux_next
        return scalar_flux_g

    def sphere(self,total_g,scatter_gg,source_g,boundary_g,scalar_flux_guess,\
                angular_flux_last=None):
        """ Function for running mono-energetic transport sweeps in one-
        dimensional spheres.

        Args:
            total_g (arrar of size (materials x 1)): Total cross section for
                each material at group g
            scatter_gg (array of size (materials x 1)): Self-scattering cross
                section for each material at group g
            source_g (array of size (materials x 1)): Summation of non self-
                scattering, fission, and external sources for group g
            boundary_g (list of size 2): boundary conditions for the (left, right)
                boundaries for group g
            scalar_flux_guess (array of size (cells x 1): Initial guess of the
                scalar flux for group g

        Returns:
            scalar_flux_g: (array of size (cells x 1)): Calculated Scalar flux
                for group g

        """
        clibrary = ctypes.cdll.LoadLibrary(C_PATH+'sweep1d.so')

        edges = np.cumsum(np.insert(np.ones((self.cells))*1/self.cell_width,0,0))
        # Positive surface area
        SA_plus = transport_tools.surface_area_calc(edges[1:]).astype('float64')
        SA_plus_ptr = ctypes.c_void_p(SA_plus.ctypes.data)
        # Negative surface area
        SA_minus = transport_tools.surface_area_calc(edges[:self.cells]).astype('float64')
        SA_minus_ptr = ctypes.c_void_p(SA_minus.ctypes.data)
        # Volume and total
        V = transport_tools.volume_calc(edges[1:],edges[:self.cells])
        if angular_flux_last is None:
            volume_total_g = (V * total_g).astype('float64')
        else:
            volume_total_g = (V * total_g + self.speed).astype('float64')
        volume_total_ptr = ctypes.c_void_p(volume_total_g.ctypes.data)

        scalar_flux_old = scalar_flux_guess.copy()

        if angular_flux_last is not None:
            angular_flux_next = np.zeros(angular_flux_last.shape,dtype='float64')

        psi_centers = np.zeros((self.corrected_angles),dtype='float64')

        converged = 0
        count = 1
        while not (converged):
            mu_minus = -1
            angular_flux = np.zeros((self.cells),dtype='float64')
            angular_flux_ptr = ctypes.c_void_p(angular_flux.ctypes.data)
            scalar_flux_g = np.zeros((self.cells),dtype='float64')
            for n in range(self.corrected_angles):
                mu_plus = mu_minus + 2 * self.w[n]
                tau = (self.mu[n] - mu_minus) / (mu_plus - mu_minus)
                if n == 0:
                    alpha_minus = ctypes.c_double(0.)
                    psi_nhalf = (transport_tools.half_angle(boundary,total_g, \
                        1/self.cell_width, source_g + scatter_gg * scalar_flux_old)).astype('float64')
                if n == self.corrected_angles - 1:
                    alpha_plus = ctypes.c_double(0.)
                else:
                    alpha_plus = ctypes.c_double(alpha_minus - self.mu[n] * self.w[n])

                if self.mu[n] > 0:
                    # psi_ihalf = ctypes.c_double(psi_centers[self.N-n-1])
                    psi_ihalf = ctypes.c_double(psi_nhalf[0])
                elif self.mu[n] < 0:
                    psi_ihalf = ctypes.c_double(boundary)
                if angular_flux_last is None:
                    Q = (V * (source_g + scatter_gg * scalar_flux_old)).astype('float64')
                else:
                    Q = (V * (source_g + scatter_gg * scalar_flux_old) + \
                        angular_flux_last[:,n] * self.speed).astype('float64')
                q_ptr = ctypes.c_void_p(Q.ctypes.data)

                psi_ptr = ctypes.c_void_p(psi_nhalf.ctypes.data)
                scalar_flux_ptr = ctypes.c_void_p(scalar_flux.ctypes.data)

                clibrary.sweep(angular_flux_ptr, scalar_flux_ptr, psi_ptr, \
                                q_ptr, v_ptr, SAp_ptr, SAm_ptr, \
                                ctypes.c_double(self.w[n]), ctypes.c_double(self.mu[n]), \
                                alpha_plus, alpha_minus, psi_ihalf, ctypes.c_double(tau))
                # Update angular center corrections
                psi_centers[n] = angular_flux[0]
                angular_flux_next[:,n] = angular_flux.copy()
                # Update angular difference coefficients
                alpha_minus = alpha_plus
                mu_minus = mu_plus
                
            change = np.linalg.norm((scalar_flux_g - scalar_flux_old) \
                                     /scalar_flux/(self.cells))
            if np.isnan(change) or np.isinf(change):
                change = 0.
            converged = (change < INNER_TOLERANCE) or (count >= MAX_ITERATIONS)
            count += 1
            scalar_flux_old = scalar_flux_g.copy()
        if angular_flux_last is not None:
            return scalar_flux_g, angular_flux
        return scalar_flux_g

    def multigroup(self, time_const=0.5, angular_flux_last=None, \
                    scalar_flux_old=None, _problem='steady-state'):
        """ Run multi group steady state problem
        Returns:
            scalar_flux: scalar flux, numpy array of size (I x G) """
        self.corrected_angles, self.mu, self.w = transport_tools.creating_weights( \
                                                    self.angles, self.boundary)

        if angular_flux_last is not None:
            angular_flux_next = np.zeros(angular_flux_last.shape)
            self.time_const = time_const
        elif scalar_flux_old is None:
            scalar_flux_old = np.zeros((self.cells, self.groups))
        # else:
        #     scalar_flux_old = np.zeros((self.cells, self.groups))

        geo = getattr(Dime,self.geometry)  # Get the specific sweep

        converged = 0
        count = 1
        while not (converged):
            scalar_flux = np.zeros(scalar_flux_old.shape)
            for g in range(self.groups):
                q_tilde = self.source[:,g] + transport_tools.update_q(self.scatter,\
                                             scalar_flux_old, g+1, self.groups, g)
                if g != 0:
                    q_tilde += transport_tools.update_q(self.scatter, scalar_flux, 0, g, g) 
                if _problem in ['steady-state','time-dependent']:
                    q_tilde += transport_tools.update_q(self.fission, \
                                            scalar_flux_old, g+1, self.groups, g)
                if g != 0 and _problem in ['steady-state', 'time-dependent']:
                    q_tilde += transport_tools.update_q(self.fission, scalar_flux, 0, g, g)

                # q_tilde = self.source[:,g] + transport_tools.update_q(self.scatter,scalar_flux_old,g+1,self.groups,g) 
                #     + transport_tools.update_q(self.fission,scalar_flux_old,g+1,self.groups,g)
                # if g != 0:
                #     q_tilde += transport_tools.update_q(self.scatter,scalar_flux,0,g,g) + transport_tools.update_q(self.fission,scalar_flux,0,g,g)


                if _problem == 'steady-state':
                    scalar_flux[:,g] = geo(self, self.total[:,g], \
                        self.scatter[:,g,g] + self.fission[:,g,g], q_tilde, \
                        self.boundary[g], scalar_flux_old[:,g])
                elif _problem == 'time-dependent':
                    scalar_flux[:,g], angular_flux_next[:,:,g] = geo(self, self.total[:,g], \
                        self.scatter[:,g,g] + self.fission[:,g,g], q_tilde, self.boundary[g], \
                        scalar_flux_old[:,g], angular_flux_last[:,:,g])
                elif _problem == 'k-eigenvalue':
                    scalar_flux[:,g] = geo(self, self.total[:,g], \
                        self.scatter[:,g,g], q_tilde, \
                        self.boundary[g], scalar_flux_old[:,g])

                # if angular_flux_last is not None:
                #     scalar_flux[:,g], angular_flux_next[:,:,g] = geo(self, self.total[:,g], \
                #         self.scatter[:,g,g] + self.fission[:,g,g], q_tilde, self.boundary[g], \
                #         scalar_flux_old[:,g], angular_flux_last[:,:,g])
                # else:
                #     # scalar_flux[:,g] = geo(self, self.total[:,g], \
                #     #     self.scatter[:,g,g] + self.fission[:,g,g], q_tilde, \
                #     #     self.boundary[g], scalar_flux_old[:,g])
                #     scalar_flux[:,g] = geo(self, self.total[:,g], \
                #         self.scatter[:,g,g], q_tilde, \
                #         self.boundary[g], scalar_flux_old[:,g])
            change = np.linalg.norm((scalar_flux - scalar_flux_old)/scalar_flux/(self.cells))
            # if np.isnan(change) or np.isinf(change):
            #     change = 0.5
            count += 1
            converged = (change < OUTER_TOLERANCE) or (count >= MAX_ITERATIONS) 
            scalar_flux_old = scalar_flux.copy()
            # print(np.sum(scalar_flux), np.sum(scalar_flux_old))
        if angular_flux_last is not None:
            return scalar_flux, angular_flux_next
        return scalar_flux


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    G = 1 # Energy Groups
    N = 8 # Angles and weights
    mu,w = np.polynomial.legendre.leggauss(N)
    w /= np.sum(w);
    # Geometry
    R = 16.; I = 1000
    delta = R/I
    boundaries = [slice(0,int(2/delta)),slice(int(2/delta),int(3/delta)),
        slice(int(3/delta),int(5/delta)),slice(int(5/delta),int(6/delta)),
        slice(int(6/delta),int(10/delta)),slice(int(10/delta),int(11/delta)),
        slice(int(11/delta),int(13/delta)),slice(int(13/delta),int(14/delta)),
        slice(int(14/delta),int(16/delta))]
    total_ = np.zeros((I,G)); total_vals = [10,10,0,5,50,5,0,10,10]
    scatter_ = np.zeros((I,G,G)); scatter_vals = [9.9,9.9,0,0,0,0,0,9.9,9.9]
    source_ = np.zeros((I,G)); source_vals = [0,1,0,0,50,0,0,1,0]
    fission_ = np.zeros((scatter_.shape))
    for ii in range(len(boundaries)):
        total_[boundaries[ii]] = total_vals[ii]
        scatter_[boundaries[ii]] = np.diag(np.repeat(scatter_vals[ii],G))
        source_[boundaries[ii]] = source_vals[ii]*1/G
    
    # Time variable
    v = np.ones((G))
    T = 100
    dt = 1
    # Time Problem
    # time_problem = Multigroup(G,N,mu,w,total_,scatter_,fission_,source_,I,delta, \
    #                         v=v,T=T,dt=dt)
    # final_phi,phi_timesteps = time_problem.backward_euler()

    reflected = np.array([[0,1]])
    vacuum = np.array([[0,0]])
    # Steady State
    steady_problem = Dime(G,I,N,total_,scatter_,fission_,source_,reflected,delta,None,'slab')
    steady_phi = steady_problem.multigroup()

    vac_ = Dime(G,I,N,total_,scatter_,fission_,source_,vacuum,delta,None,'slab')
    vac_phi = vac_.multigroup()
    
    plt.figure()
    xspace = np.linspace(0,16,1000,endpoint=False)
    plt.plot(xspace,vac_phi,label='Vac Phi',c='k',ls='--')
    plt.plot(xspace,steady_phi,label='Reflect Phi',c='r',alpha=0.6)
    plt.legend(loc=0); plt.grid()
    plt.xlabel('Distance (cm)'); plt.ylabel('Flux')
    plt.show()
