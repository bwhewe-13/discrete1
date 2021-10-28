################################################################################
#
# K-eigenvalue
#
#
#
################################################################################

from utils import transport_tools
from steady_state import Dime

import numpy as np

OUTER_TOLERANCE = 1E-12
MAX_ITERATIONS = 100

class Keg(Dime):

    def __init__(self, groups, cells, angles, total, scatter, fission, source, \
                    boundary, cell_width, material_map, geometry):
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

        """
        super(Keg, self).__init__(groups, cells, angles, total, scatter, \
                                    fission, source, boundary, cell_width, \
                                    material_map, geometry)

    def keffective(self):
        scalar_flux_old = np.random.rand(self.cells, self.groups)
        scalar_flux_old /= np.linalg.norm(scalar_flux_old)

        converged = 0
        count = 1
        while not (converged):
            self.source = np.einsum('ijk,ik->ij', self.fission, scalar_flux_old)
            scalar_flux = Dime.multigroup(self, scalar_flux_old=scalar_flux_old)
            keff = np.linalg.norm(scalar_flux)
            scalar_flux /= keff
            change = np.linalg.norm((scalar_flux - scalar_flux_old)/scalar_flux/self.cells )
            print('Power Iteration {}\n{}\nChange {} Keff {}'.format(count, \
                     '='*35, change, keff))
            converged = (change < OUTER_TOLERANCE) or (count >= MAX_ITERATIONS)
            count += 1
            scalar_flux_old = scalar_flux.copy()
        return scalar_flux, keff

if __name__ == "__main__":

    groups = 1
    angles = 32
    cells = 1000

    rad = 1.853722
    delta = rad / cells

    fission = np.array([2.84*0.0816])
    fission = np.tile(fission, (cells, groups, groups))

    total = np.array([0.32640])
    total = np.tile(total, (cells, groups))

    scatter = np.array([0.225216])
    scatter = np.tile(scatter, (cells, groups, groups))

    reflected = np.array([[0,1]])
    vacuum = np.array([[0,0]])
    problem = Keg(groups, cells, angles, total, scatter, fission, None, \
                reflected,delta,None,'slab')

    phi, keff = problem.keffective()
    print('\nKeff',keff)