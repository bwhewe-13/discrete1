# Criticality benchmark problems

import pytest
import numpy as np

import discrete1
from discrete1.critical0d import power_iteration


@pytest.mark.infinite
@pytest.mark.power_iteration
def test_one_group_plutonium_01a():
    angles = 16
    xs_total = np.array([0.32640])
    xs_scatter = np.array([[0.225216]])
    xs_fission = np.array([[3.24 * 0.0816]])
    flux, keff = power_iteration(angles, xs_total, xs_scatter, xs_fission)
    assert abs(keff - 2.612903) < 1e-4, str(keff) + " not infinite value"


@pytest.mark.smoke
@pytest.mark.infinite
@pytest.mark.power_iteration
def test_one_group_plutonium_01b():
    angles = 16
    xs_total = np.array([0.32640])
    xs_scatter = np.array([[0.225216]])
    xs_fission = np.array([[2.84 * 0.0816]])
    flux, keff = power_iteration(angles, xs_total, xs_scatter, xs_fission)
    assert abs(keff - 2.290323) < 1e-4, str(keff) + " not infinite value"


@pytest.mark.infinite
@pytest.mark.power_iteration
def test_one_group_uranium_01a():
    angles = 16
    xs_total = np.array([0.32640])
    xs_scatter = np.array([[0.248064]])
    xs_fission = np.array([[2.70 * 0.065280]])
    flux, keff = power_iteration(angles, xs_total, xs_scatter, xs_fission)
    assert abs(keff - 2.25) < 1e-4, str(keff) + " not infinite value"


@pytest.mark.infinite
@pytest.mark.power_iteration
def test_one_group_uranium_01b():
    angles = 16
    xs_total = np.array([0.32640])
    xs_scatter = np.array([[0.248064]])
    xs_fission = np.array([[2.797101 * 0.065280]])
    flux, keff = power_iteration(angles, xs_total, xs_scatter, xs_fission)
    assert abs(keff - 2.330917) < 1e-4, str(keff) + " not infinite value"


@pytest.mark.infinite
@pytest.mark.power_iteration
def test_one_group_uranium_01c():
    angles = 16
    xs_total = np.array([0.32640])
    xs_scatter = np.array([[0.248064]])
    xs_fission = np.array([[2.707308 * 0.065280]])
    flux, keff = power_iteration(angles, xs_total, xs_scatter, xs_fission)
    assert abs(keff - 2.256083) < 1e-4, str(keff) + " not infinite value"


@pytest.mark.infinite
@pytest.mark.power_iteration
def test_one_group_uranium_01d():
    angles = 16
    xs_total = np.array([0.32640])
    xs_scatter = np.array([[0.248064]])
    xs_fission = np.array([[2.679198 * 0.065280]])
    flux, keff = power_iteration(angles, xs_total, xs_scatter, xs_fission)
    assert abs(keff - 2.232667) < 1e-4, str(keff) + " not infinite value"


@pytest.mark.infinite
@pytest.mark.power_iteration
def test_one_group_heavy_water_01a():
    angles = 16
    xs_total = np.array([0.54628])
    xs_scatter = np.array([[0.464338]])
    xs_fission = np.array([[1.70 * 0.054628]])
    flux, keff = power_iteration(angles, xs_total, xs_scatter, xs_fission)
    assert abs(keff - 1.133333) < 1e-4, str(keff) + " not infinite value"


@pytest.mark.infinite
@pytest.mark.power_iteration
def test_one_group_uranium_reactor_01a():
    angles = 16
    xs_total = np.array([0.407407])
    xs_scatter = np.array([[0.328042]])
    xs_fission = np.array([[2.50 * 0.06922744]])
    flux, keff = power_iteration(angles, xs_total, xs_scatter, xs_fission)
    assert abs(keff - 2.1806667) < 1e-4, str(keff) + " not infinite value"


@pytest.mark.smoke
@pytest.mark.infinite
@pytest.mark.power_iteration
def test_two_group_plutonium_01():
    angles = 20
    xs_total = np.array([0.3360, 0.2208])
    xs_scatter = np.array([[0.23616, 0.0], [0.0432, 0.0792]]).T
    chi = np.array([[0.425], [0.575]])
    nu = np.array([[2.93, 3.10]])
    sigmaf = np.array([[0.08544, 0.0936]])
    xs_fission = chi @ (nu * sigmaf)
    flux, keff = power_iteration(angles, xs_total, xs_scatter, xs_fission)
    assert abs(keff - 2.683767) < 1e-4, str(keff) + " not infinite value"


@pytest.mark.skip(reason="Incorrect answer from Benchmarks")
@pytest.mark.infinite
@pytest.mark.power_iteration
def test_two_group_uranium_01():
    angles = 20
    xs_total = np.array([0.3456, 0.2160])
    xs_scatter = np.array([[0.26304, 0.0], [0.0720, 0.078240]]).T
    chi = np.array([[0.425], [0.575]])
    nu = np.array([[2.50, 2.70]])
    sigmaf = np.array([[0.06912, 0.06912]])
    xs_fission = chi @ (nu * sigmaf)
    flux, keff = power_iteration(angles, xs_total, xs_scatter, xs_fission)
    assert abs(keff - 2.216349) < 1e-4, str(keff) + " not infinite value"


@pytest.mark.infinite
@pytest.mark.power_iteration
def test_two_group_uranium_aluminum():
    angles = 20
    xs_total = np.array([1.27698, 0.26817])
    xs_scatter = np.array([[1.21313, 0.0], [0.020432, 0.247516]]).T
    chi = np.array([[0.0], [1.0]])
    nu = np.array([[2.83, 0.0]])
    sigmaf = np.array([[0.06070636042, 0.0]])
    xs_fission = chi @ (nu * sigmaf)
    flux, keff = power_iteration(angles, xs_total, xs_scatter, xs_fission)
    assert abs(keff - 2.661745) < 1e-4, str(keff) + " not infinite value"


@pytest.mark.infinite
@pytest.mark.power_iteration
def test_two_group_uranium_reactor_01a():
    angles = 20
    xs_total = np.array([2.52025, 0.65696])
    xs_scatter = np.array([[2.44383, 0.0], [0.029227, 0.62568]]).T
    chi = np.array([[0.0], [1.0]])
    nu = np.array([[2.5, 2.5]])
    sigmaf = np.array([[0.050632, 0.0010484]])
    xs_fission = chi @ (nu * sigmaf)
    flux, keff = power_iteration(angles, xs_total, xs_scatter, xs_fission)
    assert abs(keff - 1.631452) < 1e-4, str(keff) + " not infinite value"


@pytest.mark.infinite
@pytest.mark.power_iteration
def test_two_group_uranium_reactor_01b():
    angles = 20
    xs_total = np.array([2.9727, 0.88721])
    xs_scatter = np.array([[2.9183, 0.000767], [0.04635, 0.83892]]).T
    chi = np.array([[0.0], [1.0]])
    nu = np.array([[2.5, 2.5]])
    sigmaf = np.array([[0.029564, 0.000836]])
    xs_fission = chi @ (nu * sigmaf)
    flux, keff = power_iteration(angles, xs_total, xs_scatter, xs_fission)
    assert abs(keff - 1.365821) < 1e-4, str(keff) + " not infinite value"


@pytest.mark.infinite
@pytest.mark.power_iteration
def test_two_group_uranium_reactor_01c():
    angles = 20
    xs_total = np.array([2.9628, 0.88655])
    xs_scatter = np.array([[2.8751, 0.00116], [0.04536, 0.83807]]).T
    chi = np.array([[0.0], [1.0]])
    nu = np.array([[2.5, 2.5]])
    sigmaf = np.array([[0.057296, 0.001648]])
    xs_fission = chi @ (nu * sigmaf)
    flux, keff = power_iteration(angles, xs_total, xs_scatter, xs_fission)
    assert abs(keff - 1.633380) < 1e-4, str(keff) + " not infinite value"
