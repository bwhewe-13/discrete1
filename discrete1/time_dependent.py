################################################################################
#
# 
#
#
#
################################################################################

from discrete1.utils import transport_tools
from discrete1.steady_state import Dime

import numpy as np

class Tide(Dime):

    def __init__(self, groups, cells, angles, total, scatter, fission, source, \
                    boundary, cell_width, material_map, geometry, time_length, \
                    time_steps, time_width, velocity):
        """ Class for running one-dimensional time-dependent multigroup problem.

        .. Parent Class of Dime

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
            time_length (float): The length of the time required to run the
                transport problem.
            time_steps (int): Number of time steps taken in the elapsed time.
            time_width (float): Width of each of the time steps.
            velocty (array of size (groups x 1)): Neutron velocities for each of
                the energy levels.

        """
        self.time_length = time_length
        self.time_steps = time_steps
        self.time_width = time_width
        self.velocity = velocity
        super(Tide, self).__init__(groups, cells, angles, total, scatter, \
                                    fission, source, boundary, cell_width, \
                                    material_map, geometry)

    def bdf_one(self, display=True):
        # Initialize flux and list of fluxes
        scalar_flux_old = np.zeros((self.cells, self.groups))
        scalar_flux_collection = []
        # Initialize angular flux
        angular_flux_last = np.zeros((self.cells, self.angles, self.groups))
        # Initialize Speed
        self.speed = 1/(self.velocity * self.time_width)
        # For calculating the number of time steps (computer rounding error)
        for step in range(self.time_steps):
            # Run the multigroup problem
            scalar_flux, angular_flux_next = Dime.multigroup(self, time_const=0.5, \
                angular_flux_last=angular_flux_last, scalar_flux_old=scalar_flux_old, \
                _problem='time-dependent')
            if display:
                print('Time Step',step,'Flux',np.sum(scalar_flux),\
                    '\n===================================')
            # Update scalar/angular flux
            angular_flux_last = angular_flux_next.copy()
            scalar_flux_collection.append(scalar_flux)
            scalar_flux_old = scalar_flux.copy()
        return scalar_flux, scalar_flux_collection

    def bdf_two(self, display=True):
        # Initialize flux and list of fluxes
        scalar_flux_old = np.zeros((self.cells, self.groups))
        scalar_flux_collection = []
        # Initialize angular flux
        angular_flux_zero = np.zeros((self.cells, self.angles, self.groups))
        angular_flux_one = angular_flux_zero.copy()
        # angular_flux_last = np.zeros((self.cells, self.angles, self.groups))
        # Initialize speed
        self.speed = 1/(self.velocity * self.time_width)
        for step in range(self.time_steps):
            if step == 0:
                angular_flux_last = angular_flux_one.copy()
            else:
                angular_flux_last = 2 * angular_flux_one - 0.5 * angular_flux_zero
            scalar_flux, angular_flux_next = Dime.multigroup(self, time_const=0.75, \
                angular_flux_last=angular_flux_last, scalar_flux_old=scalar_flux_old, \
                _problem='time-dependent')
            if display:
                print('Time Step {} Flux {}\n{}'.format(step, np.sum(scalar_flux), '='*35))
            # Update scalar/angular flux
            angular_flux_zero = angular_flux_one.copy()
            angular_flux_one = angular_flux_next.copy()
            scalar_flux_collection.append(scalar_flux)
            scalar_flux_old = scalar_flux.copy()
        return scalar_flux, scalar_flux_collection


if __name__ == '__main__':
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

    steady_problem = Dime(G,I,N,total_,scatter_,fission_,source_,[0,0],delta,None,'slab')
    steady_phi = steady_problem.multigroup()
    # Time variable
    v = np.ones((G))
    T = 100
    dt = 1
    steps = 100
    time_dependent = Tide(G,I,N,total_,scatter_,fission_,source_,[0,0],delta,None,'slab',T,steps,dt,v)
    bdf_one_phi, time_phi = time_dependent.bdf_one()
    bdf_two_phi, time_phi = time_dependent.bdf_two()

    # print('\n{}\n'.format(np.sum(np.fabs(steady_phi - final_phi))))

    plt.figure()
    xspace = np.linspace(0,16,1000,endpoint=False)
    plt.plot(xspace,bdf_one_phi,label='Backward Euler - Time Dependent', c='r', alpha=0.6)
    plt.plot(xspace,bdf_two_phi,label='BDF2 - Time Dependent', c='b', alpha=0.6)
    plt.plot(xspace,steady_phi,label='Time Independent',c='k',ls='--')
    plt.legend(loc=0); plt.grid()
    plt.xlabel('Distance (cm)'); plt.ylabel('Flux')
    plt.show()
