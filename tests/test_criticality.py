################################################################################
#
# Test Problems for the Neutron Transport Equation for k-effective benchmark 
# problems. All benchmarks come from ``Analytical Benchmark Test Set for
# Criticality Code Verification'' (Los Alamos National Laboratory). Referred
# to as the `Benchmark Book' in the comments.
#
################################################################################


from discrete1 import criticality as K

import numpy as np
import pytest

@pytest.mark.smoke
def test_one_group_plutonium_slab_keff():
    """ 
    One-Energy Group Isotropic Cross section
    Case of Pu-239 with c = 1.50.
    Page 16 of Benchmark Book.
    """
    groups = 1
    angles = 20
    cells = 1000
    fission = np.array([3.24 * 0.0816])
    fission = np.tile(fission, (cells, groups, groups))
    total = np.array([0.32640])
    total = np.tile(total, (cells, groups))
    scatter = np.array([0.225216])
    scatter = np.tile(scatter, (cells, groups, groups))
    # Reflected Boundary Conditions
    reflected = np.array([[1,0]])
    rad = 1.853722
    delta = rad / cells
    problem = K.Keg(groups, cells, angles, total, scatter, fission, None, \
                reflected, delta, None, 'slab')
    scalar_flux, keff = problem.keffective(display=False)
    assert( abs(keff - 1) < 1.0e-3), 'Reflected Boundary, K-effective'
    # Vacuum Boundary Conditions    
    vacuum = np.array([[0,0]])
    rad *= 2
    delta = rad / cells
    problem = K.Keg(groups, cells, angles, total, scatter, fission, None, \
                vacuum, delta, None, 'slab')
    scalar_flux, keff = problem.keffective(False)
    assert( abs(keff - 1) < 1.0e-3), 'Vacuum Boundary, K-effective'

def test_one_group_plutonium_slab_keff_flux():
    """ 
    One-Energy Group Isotropic Cross section
    Case of Pu-239 with c = 1.40.
    Page 16 of Benchmark Book.
    """
    groups = 1
    angles = 20
    cells = 1000
    fission = np.array([2.84 * 0.0816])
    fission = np.tile(fission, (cells, groups, groups))
    total = np.array([0.32640])
    total = np.tile(total, (cells, groups))
    scatter = np.array([0.225216])
    scatter = np.tile(scatter, (cells, groups, groups))
    # Reflected Boundary Conditions
    reflected = np.array([[1,0]])
    rad = 2.256751 
    delta = rad / cells
    problem = K.Keg(groups, cells, angles, total, scatter, fission, None, \
                reflected, delta, None, 'slab')
    scalar_flux, keff = problem.keffective(display=False)
    assert( abs(keff - 1) < 1.0e-3), 'Reflected Boundary, K-effective'
    scalar_flux /= scalar_flux[-1]
    assert( abs(scalar_flux[750] - 0.9701734) < 1.e-3), 'Reflected Boundary, Flux 25%'
    assert( abs(scalar_flux[500] - 0.8810540) < 1.e-3), 'Reflected Boundary, Flux 50%'
    assert( abs(scalar_flux[250] - 0.7318131) < 1.e-3), 'Reflected Boundary, Flux 75%'
    assert( abs(scalar_flux[0] - 0.4902592) < 1.e-3), 'Reflected Boundary, Flux 100%'
    # Vacuum Boundary Conditions
    rad *= 2
    delta = rad / cells
    vacuum = np.array([[0,0]])
    problem = K.Keg(groups, cells, angles, total, scatter, fission, None, \
                vacuum, delta, None, 'slab')
    scalar_flux, keff = problem.keffective(display=False)
    assert( abs(keff - 1) < 1.0e-3), 'Vacuum Boundary, K-effective'
    scalar_flux /= scalar_flux[-1]
    assert( abs(scalar_flux[750] - 0.9701734) < 1.e-3), 'Vacuum Boundary, Flux 25%'
    assert( abs(scalar_flux[500] - 0.8810540) < 1.e-3), 'Vacuum Boundary, Flux 50%'
    assert( abs(scalar_flux[250] - 0.7318131) < 1.e-3), 'Vacuum Boundary, Flux 75%'
    assert( abs(scalar_flux[0] - 0.4902592) < 1.e-3), 'Vacuum Boundary, Flux 100%'

