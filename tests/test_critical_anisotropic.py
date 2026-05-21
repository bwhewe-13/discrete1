"""Anisotropic-scattering criticality (1-D) benchmark tests for discrete1.

These tests run the 1-D power-iteration solvers across a set of
benchmark configurations with anisotropic scattering and verify
multiplication factors against reference values.
"""

import numpy as np
import pytest

import discrete1
from discrete1.critical1d import power_iteration

################################################################################
# One Group Anisotropic Scattering
################################################################################


@pytest.mark.smoke
@pytest.mark.slab
@pytest.mark.anisotropic
@pytest.mark.power_iteration
@pytest.mark.parametrize(("bc_x"), [[0, 0], [0, 1], [1, 0]])
def test_one_group_aniso_slab_plutonium_01a(bc_x: list[int]):
    cells_x = 75
    angles = 16
    angle_x, angle_w = discrete1.angular_x(angles, bc_x)
    xs_total = np.array([[1.0]])
    xs_scatter = np.array([[[[0.733333, 0.2, 0.075]]]])
    xs_fission = np.array([[[2.5 * 0.266667]]])
    cells_x = 150 if np.sum(bc_x) == 0 else 75
    length = 0.76378 * 2 if np.sum(bc_x) == 0 else 0.76378
    medium_map = np.zeros((cells_x), dtype=np.int32)
    delta_x = np.repeat(length / cells_x, cells_x)
    flux, keff = power_iteration(
        xs_total,
        xs_scatter,
        xs_fission,
        medium_map,
        delta_x,
        angle_x,
        angle_w,
        bc_x,
        geometry=1,
    )
    assert abs(keff - 1.0) < 2e-3, str(keff) + " not critical"


@pytest.mark.smoke
@pytest.mark.slab
@pytest.mark.anisotropic
@pytest.mark.power_iteration
@pytest.mark.parametrize(("bc_x"), [[0, 0], [0, 1], [1, 0]])
def test_one_group_aniso_slab_plutonium_01a_chi(bc_x: list[int]):
    cells_x = 75
    angles = 16
    angle_x, angle_w = discrete1.angular_x(angles, bc_x)
    xs_total = np.array([[1.0]])
    xs_scatter = np.array([[[[0.733333, 0.2, 0.075]]]])
    chi = np.array([[1.0]])
    nusigf = np.array([[2.5 * 0.266667]])
    cells_x = 150 if np.sum(bc_x) == 0 else 75
    length = 0.76378 * 2 if np.sum(bc_x) == 0 else 0.76378
    medium_map = np.zeros((cells_x), dtype=np.int32)
    delta_x = np.repeat(length / cells_x, cells_x)
    flux, keff = power_iteration(
        xs_total,
        xs_scatter,
        nusigf,
        medium_map,
        delta_x,
        angle_x,
        angle_w,
        bc_x,
        chi=chi,
        geometry=1,
    )
    assert abs(keff - 1.0) < 2e-3, str(keff) + " not critical"


@pytest.mark.smoke
@pytest.mark.slab
@pytest.mark.anisotropic
@pytest.mark.power_iteration
@pytest.mark.parametrize(("bc_x"), [[0, 0], [0, 1], [1, 0]])
def test_one_group_aniso_slab_plutonium_01a_p1(bc_x: list[int]):
    cells_x = 75
    angles = 16
    angle_x, angle_w = discrete1.angular_x(angles, bc_x)
    xs_total = np.array([[1.0]])
    xs_scatter = np.array([[[[0.733333, 0.2]]]])
    xs_fission = np.array([[[2.5 * 0.266667]]])
    cells_x = 150 if np.sum(bc_x) == 0 else 75
    length = 0.77032 * 2 if np.sum(bc_x) == 0 else 0.77032
    medium_map = np.zeros((cells_x), dtype=np.int32)
    delta_x = np.repeat(length / cells_x, cells_x)
    flux, keff = power_iteration(
        xs_total,
        xs_scatter,
        xs_fission,
        medium_map,
        delta_x,
        angle_x,
        angle_w,
        bc_x,
        geometry=1,
    )
    assert abs(keff - 1.0) < 2e-3, str(keff) + " not critical"


@pytest.mark.smoke
@pytest.mark.slab
@pytest.mark.anisotropic
@pytest.mark.power_iteration
@pytest.mark.parametrize(("bc_x"), [[0, 0], [0, 1], [1, 0]])
def test_one_group_aniso_slab_plutonium_01a_p1_chi(bc_x: list[int]):
    cells_x = 75
    angles = 16
    angle_x, angle_w = discrete1.angular_x(angles, bc_x)
    xs_total = np.array([[1.0]])
    xs_scatter = np.array([[[[0.733333, 0.2]]]])
    chi = np.array([[1.0]])
    nusigf = np.array([[2.5 * 0.266667]])
    cells_x = 150 if np.sum(bc_x) == 0 else 75
    length = 0.77032 * 2 if np.sum(bc_x) == 0 else 0.77032
    medium_map = np.zeros((cells_x), dtype=np.int32)
    delta_x = np.repeat(length / cells_x, cells_x)
    flux, keff = power_iteration(
        xs_total,
        xs_scatter,
        nusigf,
        medium_map,
        delta_x,
        angle_x,
        angle_w,
        bc_x,
        chi=chi,
        geometry=1,
    )
    assert abs(keff - 1.0) < 2e-3, str(keff) + " not critical"


@pytest.mark.smoke
@pytest.mark.slab
@pytest.mark.anisotropic
@pytest.mark.power_iteration
@pytest.mark.parametrize(("bc_x"), [[0, 0], [0, 1], [1, 0]])
def test_one_group_aniso_slab_plutonium_01b(bc_x: list[int]):
    cells_x = 75
    angles = 16
    angle_x, angle_w = discrete1.angular_x(angles, bc_x)
    xs_total = np.array([[1.0]])
    xs_scatter = np.array([[[[0.733333, 0.333333, 0.125]]]])
    xs_fission = np.array([[[2.5 * 0.266667]]])
    cells_x = 150 if np.sum(bc_x) == 0 else 75
    length = 0.78396 * 2 if np.sum(bc_x) == 0 else 0.78396
    medium_map = np.zeros((cells_x), dtype=np.int32)
    delta_x = np.repeat(length / cells_x, cells_x)
    flux, keff = power_iteration(
        xs_total,
        xs_scatter,
        xs_fission,
        medium_map,
        delta_x,
        angle_x,
        angle_w,
        bc_x,
        geometry=1,
    )
    assert abs(keff - 1.0) < 2e-3, str(keff) + " not critical"


@pytest.mark.smoke
@pytest.mark.slab
@pytest.mark.anisotropic
@pytest.mark.power_iteration
@pytest.mark.parametrize(("bc_x"), [[0, 0], [0, 1], [1, 0]])
def test_one_group_aniso_slab_plutonium_01b_chi(bc_x: list[int]):
    cells_x = 75
    angles = 16
    angle_x, angle_w = discrete1.angular_x(angles, bc_x)
    xs_total = np.array([[1.0]])
    xs_scatter = np.array([[[[0.733333, 0.333333, 0.125]]]])
    chi = np.array([[1.0]])
    nusigf = np.array([[2.5 * 0.266667]])
    cells_x = 150 if np.sum(bc_x) == 0 else 75
    length = 0.78396 * 2 if np.sum(bc_x) == 0 else 0.78396
    medium_map = np.zeros((cells_x), dtype=np.int32)
    delta_x = np.repeat(length / cells_x, cells_x)
    flux, keff = power_iteration(
        xs_total,
        xs_scatter,
        nusigf,
        medium_map,
        delta_x,
        angle_x,
        angle_w,
        bc_x,
        chi=chi,
        geometry=1,
    )
    assert abs(keff - 1.0) < 2e-3, str(keff) + " not critical"


@pytest.mark.smoke
@pytest.mark.slab
@pytest.mark.anisotropic
@pytest.mark.power_iteration
@pytest.mark.parametrize(("bc_x"), [[0, 0], [0, 1], [1, 0]])
def test_one_group_aniso_slab_plutonium_01b_p1(bc_x: list[int]):
    # Report warns of negative scattering for |mu| > 1/3 (PUb-1-1-SL, Problem 34)
    cells_x = 75
    angles = 16
    angle_x, angle_w = discrete1.angular_x(angles, bc_x)
    xs_total = np.array([[1.0]])
    xs_scatter = np.array([[[[0.733333, 0.333333]]]])
    xs_fission = np.array([[[2.5 * 0.266667]]])
    cells_x = 150 if np.sum(bc_x) == 0 else 75
    length = 0.79606 * 2 if np.sum(bc_x) == 0 else 0.79606
    medium_map = np.zeros((cells_x), dtype=np.int32)
    delta_x = np.repeat(length / cells_x, cells_x)
    flux, keff = power_iteration(
        xs_total,
        xs_scatter,
        xs_fission,
        medium_map,
        delta_x,
        angle_x,
        angle_w,
        bc_x,
        geometry=1,
    )
    assert abs(keff - 1.0) < 2e-3, str(keff) + " not critical"


@pytest.mark.smoke
@pytest.mark.slab
@pytest.mark.anisotropic
@pytest.mark.power_iteration
@pytest.mark.parametrize(("bc_x"), [[0, 0], [0, 1], [1, 0]])
def test_one_group_aniso_slab_plutonium_01b_p1_chi(bc_x: list[int]):
    # Report warns of negative scattering for |mu| > 1/3 (PUb-1-1-SL, Problem 34)
    cells_x = 75
    angles = 16
    angle_x, angle_w = discrete1.angular_x(angles, bc_x)
    xs_total = np.array([[1.0]])
    xs_scatter = np.array([[[[0.733333, 0.333333]]]])
    chi = np.array([[1.0]])
    nusigf = np.array([[2.5 * 0.266667]])
    cells_x = 150 if np.sum(bc_x) == 0 else 75
    length = 0.79606 * 2 if np.sum(bc_x) == 0 else 0.79606
    medium_map = np.zeros((cells_x), dtype=np.int32)
    delta_x = np.repeat(length / cells_x, cells_x)
    flux, keff = power_iteration(
        xs_total,
        xs_scatter,
        nusigf,
        medium_map,
        delta_x,
        angle_x,
        angle_w,
        bc_x,
        chi=chi,
        geometry=1,
    )
    assert abs(keff - 1.0) < 2e-3, str(keff) + " not critical"


################################################################################
# One Group Anisotropic Scattering -UD20 Spheres
################################################################################


@pytest.mark.smoke
@pytest.mark.sphere
@pytest.mark.anisotropic
@pytest.mark.power_iteration
def test_one_group_aniso_sphere_heavy_water_01a():
    # UD20(a)-1-1-SP (Problem 39): P1 anisotropic sphere, rc = 18.30563081 cm
    cells_x = 200
    angles = 16
    bc_x = [1, 0]
    angle_x, angle_w = discrete1.angular_x(angles, bc_x)
    xs_total = np.array([[0.54628]])
    xs_scatter = np.array([[[[0.464338, 0.056312624]]]])
    xs_fission = np.array([[[1.808381 * 0.054628]]])
    length = 18.30563081
    delta_x = np.repeat(length / cells_x, cells_x)
    medium_map = np.zeros((cells_x), dtype=np.int32)
    flux, keff = power_iteration(
        xs_total,
        xs_scatter,
        xs_fission,
        medium_map,
        delta_x,
        angle_x,
        angle_w,
        bc_x,
        geometry=2,
    )
    assert abs(keff - 1.0) < 2e-3, str(keff) + " not critical"


@pytest.mark.smoke
@pytest.mark.sphere
@pytest.mark.anisotropic
@pytest.mark.power_iteration
def test_one_group_aniso_sphere_heavy_water_01a_chi():
    cells_x = 200
    angles = 16
    bc_x = [1, 0]
    angle_x, angle_w = discrete1.angular_x(angles, bc_x)
    xs_total = np.array([[0.54628]])
    xs_scatter = np.array([[[[0.464338, 0.056312624]]]])
    chi = np.array([[1.0]])
    nusigf = np.array([[1.808381 * 0.054628]])
    length = 18.30563081
    delta_x = np.repeat(length / cells_x, cells_x)
    medium_map = np.zeros((cells_x), dtype=np.int32)
    flux, keff = power_iteration(
        xs_total,
        xs_scatter,
        nusigf,
        medium_map,
        delta_x,
        angle_x,
        angle_w,
        bc_x,
        chi=chi,
        geometry=2,
    )
    assert abs(keff - 1.0) < 2e-3, str(keff) + " not critical"


@pytest.mark.smoke
@pytest.mark.sphere
@pytest.mark.anisotropic
@pytest.mark.power_iteration
def test_one_group_aniso_sphere_heavy_water_01b():
    # UD20(b)-1-1-SP (Problem 41): P1 anisotropic sphere, rc = 18.30563081 cm
    cells_x = 200
    angles = 16
    bc_x = [1, 0]
    angle_x, angle_w = discrete1.angular_x(angles, bc_x)
    xs_total = np.array([[0.54628]])
    xs_scatter = np.array([[[[0.464338, 0.112982569]]]])
    xs_fission = np.array([[[1.841086 * 0.054628]]])
    length = 18.30563081
    delta_x = np.repeat(length / cells_x, cells_x)
    medium_map = np.zeros((cells_x), dtype=np.int32)
    flux, keff = power_iteration(
        xs_total,
        xs_scatter,
        xs_fission,
        medium_map,
        delta_x,
        angle_x,
        angle_w,
        bc_x,
        geometry=2,
    )
    assert abs(keff - 1.0) < 2e-3, str(keff) + " not critical"


@pytest.mark.smoke
@pytest.mark.sphere
@pytest.mark.anisotropic
@pytest.mark.power_iteration
def test_one_group_aniso_sphere_heavy_water_01b_chi():
    cells_x = 200
    angles = 16
    bc_x = [1, 0]
    angle_x, angle_w = discrete1.angular_x(angles, bc_x)
    xs_total = np.array([[0.54628]])
    xs_scatter = np.array([[[[0.464338, 0.112982569]]]])
    chi = np.array([[1.0]])
    nusigf = np.array([[1.841086 * 0.054628]])
    length = 18.30563081
    delta_x = np.repeat(length / cells_x, cells_x)
    medium_map = np.zeros((cells_x), dtype=np.int32)
    flux, keff = power_iteration(
        xs_total,
        xs_scatter,
        nusigf,
        medium_map,
        delta_x,
        angle_x,
        angle_w,
        bc_x,
        chi=chi,
        geometry=2,
    )
    assert abs(keff - 1.0) < 2e-3, str(keff) + " not critical"


@pytest.mark.smoke
@pytest.mark.sphere
@pytest.mark.anisotropic
@pytest.mark.power_iteration
def test_one_group_aniso_sphere_heavy_water_01c():
    # UD20(c)-1-1-SP (Problem 43): P1 anisotropic sphere, rc = 18.30563081 cm
    # Negative Sigma_s1 -report warns of negative scattering for |mu| near -1
    cells_x = 200
    angles = 16
    bc_x = [1, 0]
    angle_x, angle_w = discrete1.angular_x(angles, bc_x)
    xs_total = np.array([[0.54628]])
    xs_scatter = np.array([[[[0.464338, -0.27850447]]]])
    xs_fission = np.array([[[1.6964 * 0.054628]]])
    length = 18.30563081
    delta_x = np.repeat(length / cells_x, cells_x)
    medium_map = np.zeros((cells_x), dtype=np.int32)
    flux, keff = power_iteration(
        xs_total,
        xs_scatter,
        xs_fission,
        medium_map,
        delta_x,
        angle_x,
        angle_w,
        bc_x,
        geometry=2,
    )
    assert abs(keff - 1.0) < 2e-3, str(keff) + " not critical"


@pytest.mark.smoke
@pytest.mark.sphere
@pytest.mark.anisotropic
@pytest.mark.power_iteration
def test_one_group_aniso_sphere_heavy_water_01c_chi():
    cells_x = 200
    angles = 16
    bc_x = [1, 0]
    angle_x, angle_w = discrete1.angular_x(angles, bc_x)
    xs_total = np.array([[0.54628]])
    xs_scatter = np.array([[[[0.464338, -0.27850447]]]])
    chi = np.array([[1.0]])
    nusigf = np.array([[1.6964 * 0.054628]])
    length = 18.30563081
    delta_x = np.repeat(length / cells_x, cells_x)
    medium_map = np.zeros((cells_x), dtype=np.int32)
    flux, keff = power_iteration(
        xs_total,
        xs_scatter,
        nusigf,
        medium_map,
        delta_x,
        angle_x,
        angle_w,
        bc_x,
        chi=chi,
        geometry=2,
    )
    assert abs(keff - 1.0) < 2e-3, str(keff) + " not critical"


################################################################################
# Two Group Anisotropic Scattering
################################################################################


@pytest.mark.smoke
@pytest.mark.slab
@pytest.mark.anisotropic
@pytest.mark.power_iteration
@pytest.mark.parametrize(("bc_x"), [[0, 0], [0, 1], [1, 0]])
def test_two_group_aniso_slab_uranium_reactor_01(bc_x: list[int]):
    # URRa-2-1-SL (Problem 71): 2-group P1 anisotropic, rc = 9.491600 cm
    # Report warns of negative scattering for |mu| near -1
    cells_x = 200
    angles = 20
    angle_x, angle_w = discrete1.angular_x(angles, bc_x)
    xs_total = np.array([[2.52025, 0.65696]])
    xs_scatter = np.array(
        [
            np.stack(
                [
                    np.array([[2.44383, 0.0], [0.029227, 0.62568]]).T,  # L0
                    np.array([[0.83318, 0.0], [0.0075737, 0.27459]]).T,  # L1
                ],
                axis=-1,
            )
        ]
    )
    chi = np.array([[0.0], [1.0]])
    nu = np.array([[2.5, 2.5]])
    sigmaf = np.array([[0.050632, 0.0010484]])
    xs_fission = np.array([chi @ (nu * sigmaf)])
    cells_x = 200 if np.sum(bc_x) == 0 else 100
    length = 9.491600 * 2 if np.sum(bc_x) == 0 else 9.491600
    delta_x = np.repeat(length / cells_x, cells_x)
    medium_map = np.zeros((cells_x), dtype=np.int32)
    flux, keff = power_iteration(
        xs_total,
        xs_scatter,
        xs_fission,
        medium_map,
        delta_x,
        angle_x,
        angle_w,
        bc_x,
        geometry=1,
    )
    assert abs(keff - 1.0) < 2e-3, str(keff) + " not critical"


@pytest.mark.smoke
@pytest.mark.slab
@pytest.mark.anisotropic
@pytest.mark.power_iteration
@pytest.mark.parametrize(("bc_x"), [[0, 0], [0, 1], [1, 0]])
def test_two_group_aniso_slab_uranium_reactor_01_chi(bc_x: list[int]):
    # URRa-2-1-SL (Problem 71): 2-group P1 anisotropic, rc = 9.491600 cm
    # Report warns of negative scattering for |mu| near -1
    cells_x = 200
    angles = 20
    angle_x, angle_w = discrete1.angular_x(angles, bc_x)
    xs_total = np.array([[2.52025, 0.65696]])
    xs_scatter = np.array(
        [
            np.stack(
                [
                    np.array([[2.44383, 0.0], [0.029227, 0.62568]]).T,  # L0
                    np.array([[0.83318, 0.0], [0.0075737, 0.27459]]).T,  # L1
                ],
                axis=-1,
            )
        ]
    )
    chi = np.array([[0.0, 1.0]])
    nu = np.array([[2.5, 2.5]])
    sigmaf = np.array([[0.050632, 0.0010484]])
    nusigf = nu * sigmaf
    cells_x = 200 if np.sum(bc_x) == 0 else 100
    length = 9.491600 * 2 if np.sum(bc_x) == 0 else 9.491600
    delta_x = np.repeat(length / cells_x, cells_x)
    medium_map = np.zeros((cells_x), dtype=np.int32)
    flux, keff = power_iteration(
        xs_total,
        xs_scatter,
        nusigf,
        medium_map,
        delta_x,
        angle_x,
        angle_w,
        bc_x,
        chi=chi,
        geometry=1,
    )
    assert abs(keff - 1.0) < 2e-3, str(keff) + " not critical"
