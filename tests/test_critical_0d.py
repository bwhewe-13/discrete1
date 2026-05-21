"""Criticality (0-D) benchmark tests.

These tests exercise the scalar (0-D) criticality solvers and compare
computed multiplication factors to benchmark references.
"""

import numpy as np
import pytest

from discrete1.critical0d import power_iteration

################################################################################
# One Group Isotropic Scattering
################################################################################


@pytest.mark.infinite
@pytest.mark.power_iteration
def test_one_group_plutonium_01a():
    angles = 16
    xs_total = np.array([0.32640])
    xs_scatter = np.array([[0.225216]])
    xs_fission = np.array([[3.24 * 0.0816]])
    _, keff = power_iteration(angles, xs_total, xs_scatter, xs_fission)
    assert abs(keff - 2.612903) < 1e-4, str(keff) + " not infinite value"


@pytest.mark.infinite
@pytest.mark.power_iteration
def test_one_group_plutonium_01a_chi():
    angles = 16
    xs_total = np.array([0.32640])
    xs_scatter = np.array([[0.225216]])
    chi = np.array([[1.0]])
    nusigf = np.array([[3.24 * 0.0816]]).flatten()
    _, keff = power_iteration(angles, xs_total, xs_scatter, nusigf, chi.flatten())
    assert abs(keff - 2.612903) < 1e-4, str(keff) + " not infinite value"


@pytest.mark.smoke
@pytest.mark.infinite
@pytest.mark.power_iteration
def test_one_group_plutonium_01b():
    angles = 16
    xs_total = np.array([0.32640])
    xs_scatter = np.array([[0.225216]])
    xs_fission = np.array([[2.84 * 0.0816]])
    _, keff = power_iteration(angles, xs_total, xs_scatter, xs_fission)
    assert abs(keff - 2.290323) < 1e-4, str(keff) + " not infinite value"


@pytest.mark.smoke
@pytest.mark.infinite
@pytest.mark.power_iteration
def test_one_group_plutonium_01b_chi():
    angles = 16
    xs_total = np.array([0.32640])
    xs_scatter = np.array([[0.225216]])
    chi = np.array([[1.0]])
    nusigf = np.array([[2.84 * 0.0816]]).flatten()
    _, keff = power_iteration(angles, xs_total, xs_scatter, nusigf, chi.flatten())
    assert abs(keff - 2.290323) < 1e-4, str(keff) + " not infinite value"


@pytest.mark.infinite
@pytest.mark.power_iteration
def test_one_group_uranium_01a():
    angles = 16
    xs_total = np.array([0.32640])
    xs_scatter = np.array([[0.248064]])
    xs_fission = np.array([[2.70 * 0.065280]])
    _, keff = power_iteration(angles, xs_total, xs_scatter, xs_fission)
    assert abs(keff - 2.25) < 1e-4, str(keff) + " not infinite value"


@pytest.mark.infinite
@pytest.mark.power_iteration
def test_one_group_uranium_01a_chi():
    angles = 16
    xs_total = np.array([0.32640])
    xs_scatter = np.array([[0.248064]])
    chi = np.array([[1.0]])
    nusigf = np.array([[2.70 * 0.065280]]).flatten()
    _, keff = power_iteration(angles, xs_total, xs_scatter, nusigf, chi.flatten())
    assert abs(keff - 2.25) < 1e-4, str(keff) + " not infinite value"


@pytest.mark.infinite
@pytest.mark.power_iteration
def test_one_group_uranium_01b():
    angles = 16
    xs_total = np.array([0.32640])
    xs_scatter = np.array([[0.248064]])
    xs_fission = np.array([[2.797101 * 0.065280]])
    _, keff = power_iteration(angles, xs_total, xs_scatter, xs_fission)
    assert abs(keff - 2.330917) < 1e-4, str(keff) + " not infinite value"


@pytest.mark.infinite
@pytest.mark.power_iteration
def test_one_group_uranium_01b_chi():
    angles = 16
    xs_total = np.array([0.32640])
    xs_scatter = np.array([[0.248064]])
    chi = np.array([[1.0]])
    nusigf = np.array([[2.797101 * 0.065280]]).flatten()
    _, keff = power_iteration(angles, xs_total, xs_scatter, nusigf, chi.flatten())
    assert abs(keff - 2.330917) < 1e-4, str(keff) + " not infinite value"


@pytest.mark.infinite
@pytest.mark.power_iteration
def test_one_group_uranium_01c():
    angles = 16
    xs_total = np.array([0.32640])
    xs_scatter = np.array([[0.248064]])
    xs_fission = np.array([[2.707308 * 0.065280]])
    _, keff = power_iteration(angles, xs_total, xs_scatter, xs_fission)
    assert abs(keff - 2.256083) < 1e-4, str(keff) + " not infinite value"


@pytest.mark.infinite
@pytest.mark.power_iteration
def test_one_group_uranium_01_chi():
    angles = 16
    xs_total = np.array([0.32640])
    xs_scatter = np.array([[0.248064]])
    chi = np.array([[1.0]])
    nusigf = np.array([[2.707308 * 0.065280]]).flatten()
    _, keff = power_iteration(angles, xs_total, xs_scatter, nusigf, chi.flatten())
    assert abs(keff - 2.256083) < 1e-4, str(keff) + " not infinite value"


@pytest.mark.infinite
@pytest.mark.power_iteration
def test_one_group_uranium_01d():
    angles = 16
    xs_total = np.array([0.32640])
    xs_scatter = np.array([[0.248064]])
    xs_fission = np.array([[2.679198 * 0.065280]])
    _, keff = power_iteration(angles, xs_total, xs_scatter, xs_fission)
    assert abs(keff - 2.232667) < 1e-4, str(keff) + " not infinite value"


@pytest.mark.infinite
@pytest.mark.power_iteration
def test_one_group_uranium_01d_chi():
    angles = 16
    xs_total = np.array([0.32640])
    xs_scatter = np.array([[0.248064]])
    chi = np.array([[1.0]])
    nusigf = np.array([[2.679198 * 0.065280]]).flatten()
    _, keff = power_iteration(angles, xs_total, xs_scatter, nusigf, chi.flatten())
    assert abs(keff - 2.232667) < 1e-4, str(keff) + " not infinite value"


@pytest.mark.infinite
@pytest.mark.power_iteration
def test_one_group_heavy_water_01a():
    angles = 16
    xs_total = np.array([0.54628])
    xs_scatter = np.array([[0.464338]])
    xs_fission = np.array([[1.70 * 0.054628]])
    _, keff = power_iteration(angles, xs_total, xs_scatter, xs_fission)
    assert abs(keff - 1.133333) < 1e-4, str(keff) + " not infinite value"


@pytest.mark.infinite
@pytest.mark.power_iteration
def test_one_group_heavy_water_01a_chi():
    angles = 16
    xs_total = np.array([0.54628])
    xs_scatter = np.array([[0.464338]])
    chi = np.array([[1.0]])
    nusigf = np.array([[1.70 * 0.054628]]).flatten()
    _, keff = power_iteration(angles, xs_total, xs_scatter, nusigf, chi.flatten())
    assert abs(keff - 1.133333) < 1e-4, str(keff) + " not infinite value"


@pytest.mark.infinite
@pytest.mark.power_iteration
def test_one_group_uranium_reactor_01a():
    angles = 16
    xs_total = np.array([0.407407])
    xs_scatter = np.array([[0.328042]])
    xs_fission = np.array([[2.50 * 0.06922744]])
    _, keff = power_iteration(angles, xs_total, xs_scatter, xs_fission)
    assert abs(keff - 2.1806667) < 1e-4, str(keff) + " not infinite value"


@pytest.mark.infinite
@pytest.mark.power_iteration
def test_one_group_uranium_reactor_01a_chi():
    angles = 16
    xs_total = np.array([0.407407])
    xs_scatter = np.array([[0.328042]])
    chi = np.array([1.0])
    nusigf = np.array([2.50 * 0.06922744])
    _, keff = power_iteration(angles, xs_total, xs_scatter, nusigf, chi)
    assert abs(keff - 2.1806667) < 1e-4, str(keff) + " not infinite value"


################################################################################
# One Group Anisotropic Scattering
################################################################################
@pytest.mark.anisotropic
@pytest.mark.infinite
@pytest.mark.power_iteration
def test_one_group_aniso_plutonium_01a():
    angles = 16
    xs_total = np.array([1.0])
    xs_scatter = np.array([[[0.733333, 0.2, 0.075]]])
    xs_fission = np.array([[2.5 * 0.266667]])
    _, keff = power_iteration(angles, xs_total, xs_scatter, xs_fission)
    assert abs(keff - 2.5) < 1e-4, str(keff) + " not infinite value"


@pytest.mark.anisotropic
@pytest.mark.infinite
@pytest.mark.power_iteration
def test_one_group_aniso_plutonium_01a_chi():
    angles = 16
    xs_total = np.array([1.0])
    xs_scatter = np.array([[[0.733333, 0.2, 0.075]]])
    chi = np.array([1.0])
    nusigf = np.array([2.5 * 0.266667])
    _, keff = power_iteration(angles, xs_total, xs_scatter, nusigf, chi=chi)
    assert abs(keff - 2.5) < 1e-4, str(keff) + " not infinite value"


@pytest.mark.anisotropic
@pytest.mark.infinite
@pytest.mark.power_iteration
def test_one_group_aniso_plutonium_01b():
    angles = 16
    xs_total = np.array([1.0])
    xs_scatter = np.array([[[0.733333, 0.333333, 0.125]]])
    xs_fission = np.array([[2.5 * 0.266667]])
    _, keff = power_iteration(angles, xs_total, xs_scatter, xs_fission)
    assert abs(keff - 2.5) < 1e-4, str(keff) + " not infinite value"


@pytest.mark.anisotropic
@pytest.mark.infinite
@pytest.mark.power_iteration
def test_one_group_aniso_plutonium_01b_chi():
    angles = 16
    xs_total = np.array([1.0])
    xs_scatter = np.array([[[0.733333, 0.333333, 0.125]]])
    chi = np.array([1.0])
    nusigf = np.array([2.5 * 0.266667])
    _, keff = power_iteration(angles, xs_total, xs_scatter, nusigf, chi=chi)
    assert abs(keff - 2.5) < 1e-4, str(keff) + " not infinite value"


@pytest.mark.anisotropic
@pytest.mark.infinite
@pytest.mark.power_iteration
def test_one_group_aniso_heavy_water_01a():
    # UD20(a)-1-1-IN (Problem 38): k_inf = 1.205587
    angles = 16
    xs_total = np.array([0.54628])
    xs_scatter = np.array([[[0.464338, 0.056312624]]])
    xs_fission = np.array([[1.808381 * 0.054628]])
    _, keff = power_iteration(angles, xs_total, xs_scatter, xs_fission)
    assert abs(keff - 1.205587) < 1e-4, str(keff) + " not infinite value"


@pytest.mark.anisotropic
@pytest.mark.infinite
@pytest.mark.power_iteration
def test_one_group_aniso_heavy_water_01a_chi():
    angles = 16
    xs_total = np.array([0.54628])
    xs_scatter = np.array([[[0.464338, 0.056312624]]])
    chi = np.array([1.0])
    nusigf = np.array([1.808381 * 0.054628])
    _, keff = power_iteration(angles, xs_total, xs_scatter, nusigf, chi=chi)
    assert abs(keff - 1.205587) < 1e-4, str(keff) + " not infinite value"


@pytest.mark.anisotropic
@pytest.mark.infinite
@pytest.mark.power_iteration
def test_one_group_aniso_heavy_water_01b():
    # UD20(b)-1-1-IN (Problem 40): k_inf = 1.227391
    angles = 16
    xs_total = np.array([0.54628])
    xs_scatter = np.array([[[0.464338, 0.112982569]]])
    xs_fission = np.array([[1.841086 * 0.054628]])
    _, keff = power_iteration(angles, xs_total, xs_scatter, xs_fission)
    assert abs(keff - 1.227391) < 1e-4, str(keff) + " not infinite value"


@pytest.mark.anisotropic
@pytest.mark.infinite
@pytest.mark.power_iteration
def test_one_group_aniso_heavy_water_01b_chi():
    angles = 16
    xs_total = np.array([0.54628])
    xs_scatter = np.array([[[0.464338, 0.112982569]]])
    chi = np.array([1.0])
    nusigf = np.array([1.841086 * 0.054628])
    _, keff = power_iteration(angles, xs_total, xs_scatter, nusigf, chi=chi)
    assert abs(keff - 1.227391) < 1e-4, str(keff) + " not infinite value"


@pytest.mark.anisotropic
@pytest.mark.infinite
@pytest.mark.power_iteration
def test_one_group_aniso_heavy_water_01c():
    # UD20(c)-1-1-IN (Problem 42): k_inf = 1.130933
    # Negative Sigma_s1 - report warns of negative scattering for |mu| near -1
    angles = 16
    xs_total = np.array([0.54628])
    xs_scatter = np.array([[[0.464338, -0.27850447]]])
    xs_fission = np.array([[1.6964 * 0.054628]])
    _, keff = power_iteration(angles, xs_total, xs_scatter, xs_fission)
    assert abs(keff - 1.130933) < 1e-4, str(keff) + " not infinite value"


@pytest.mark.anisotropic
@pytest.mark.infinite
@pytest.mark.power_iteration
def test_one_group_aniso_heavy_water_01c_chi():
    angles = 16
    xs_total = np.array([0.54628])
    xs_scatter = np.array([[[0.464338, -0.27850447]]])
    chi = np.array([1.0])
    nusigf = np.array([1.6964 * 0.054628])
    _, keff = power_iteration(angles, xs_total, xs_scatter, nusigf, chi=chi)
    assert abs(keff - 1.130933) < 1e-4, str(keff) + " not infinite value"


################################################################################
# Two Group Isotropic Scattering
################################################################################


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
    _, keff = power_iteration(angles, xs_total, xs_scatter, xs_fission)
    assert abs(keff - 2.683767) < 1e-4, str(keff) + " not infinite value"


@pytest.mark.smoke
@pytest.mark.infinite
@pytest.mark.power_iteration
def test_two_group_plutonium_01_chi():
    angles = 20
    xs_total = np.array([0.3360, 0.2208])
    xs_scatter = np.array([[0.23616, 0.0], [0.0432, 0.0792]]).T
    chi = np.array([[0.425, 0.575]])
    nu = np.array([[2.93, 3.10]])
    sigmaf = np.array([[0.08544, 0.0936]])
    nusigf = (nu * sigmaf).flatten()
    _, keff = power_iteration(angles, xs_total, xs_scatter, nusigf, chi.flatten())
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
    _, keff = power_iteration(angles, xs_total, xs_scatter, xs_fission)
    assert abs(keff - 2.216349) < 1e-4, str(keff) + " not infinite value"


@pytest.mark.skip(reason="Incorrect answer from Benchmarks")
@pytest.mark.infinite
@pytest.mark.power_iteration
def test_two_group_uranium_01_chi():
    angles = 20
    xs_total = np.array([0.3456, 0.2160])
    xs_scatter = np.array([[0.26304, 0.0], [0.0720, 0.078240]]).T
    chi = np.array([[0.425, 0.575]])
    nu = np.array([[2.50, 2.70]])
    sigmaf = np.array([[0.06912, 0.06912]])
    nusigf = (nu * sigmaf).flatten()
    _, keff = power_iteration(angles, xs_total, xs_scatter, nusigf, chi.flatten())
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
    _, keff = power_iteration(angles, xs_total, xs_scatter, xs_fission)
    assert abs(keff - 2.661745) < 1e-4, str(keff) + " not infinite value"


@pytest.mark.infinite
@pytest.mark.power_iteration
def test_two_group_uranium_aluminum_chi():
    angles = 20
    xs_total = np.array([1.27698, 0.26817])
    xs_scatter = np.array([[1.21313, 0.0], [0.020432, 0.247516]]).T
    chi = np.array([[0.0, 1.0]])
    nu = np.array([[2.83, 0.0]])
    sigmaf = np.array([[0.06070636042, 0.0]])
    nusigf = (nu * sigmaf).flatten()
    _, keff = power_iteration(angles, xs_total, xs_scatter, nusigf, chi.flatten())
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
    _, keff = power_iteration(angles, xs_total, xs_scatter, xs_fission)
    assert abs(keff - 1.631452) < 1e-4, str(keff) + " not infinite value"


@pytest.mark.infinite
@pytest.mark.power_iteration
def test_two_group_uranium_reactor_01a_chi():
    angles = 20
    xs_total = np.array([2.52025, 0.65696])
    xs_scatter = np.array([[2.44383, 0.0], [0.029227, 0.62568]]).T
    chi = np.array([[0.0, 1.0]])
    nu = np.array([[2.5, 2.5]])
    sigmaf = np.array([[0.050632, 0.0010484]])
    nusigf = (nu * sigmaf).flatten()
    _, keff = power_iteration(angles, xs_total, xs_scatter, nusigf, chi.flatten())
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
    _, keff = power_iteration(angles, xs_total, xs_scatter, xs_fission)
    assert abs(keff - 1.365821) < 1e-4, str(keff) + " not infinite value"


@pytest.mark.infinite
@pytest.mark.power_iteration
def test_two_group_uranium_reactor_01b_chi():
    angles = 20
    xs_total = np.array([2.9727, 0.88721])
    xs_scatter = np.array([[2.9183, 0.000767], [0.04635, 0.83892]]).T
    chi = np.array([[0.0, 1.0]])
    nu = np.array([[2.5, 2.5]])
    sigmaf = np.array([[0.029564, 0.000836]])
    nusigf = (nu * sigmaf).flatten()
    _, keff = power_iteration(angles, xs_total, xs_scatter, nusigf, chi.flatten())
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
    _, keff = power_iteration(angles, xs_total, xs_scatter, xs_fission)
    assert abs(keff - 1.633380) < 1e-4, str(keff) + " not infinite value"


@pytest.mark.infinite
@pytest.mark.power_iteration
def test_two_group_uranium_reactor_01c_chi():
    angles = 20
    xs_total = np.array([2.9628, 0.88655])
    xs_scatter = np.array([[2.8751, 0.00116], [0.04536, 0.83807]]).T
    chi = np.array([[0.0, 1.0]])
    nu = np.array([[2.5, 2.5]])
    sigmaf = np.array([[0.057296, 0.001648]])
    nusigf = (nu * sigmaf).flatten()
    _, keff = power_iteration(angles, xs_total, xs_scatter, nusigf, chi.flatten())
    assert abs(keff - 1.633380) < 1e-4, str(keff) + " not infinite value"


@pytest.mark.infinite
@pytest.mark.power_iteration
def test_two_group_uranium_reactor_01d():
    # URRd-2-0-IN (Problem 62): k_inf = 1.034970
    # Note: nu_fast = 1.004 is slightly unphysical per report
    angles = 20
    xs_total = np.array([2.13800, 0.650917])
    xs_scatter = np.array([[2.06880, 0.0], [0.0342008, 0.0]]).T
    chi = np.array([[0.0], [1.0]])
    nu = np.array([[2.50, 1.004]])
    sigmaf = np.array([[0.045704, 0.61475]])
    xs_fission = chi @ (nu * sigmaf)
    _, keff = power_iteration(angles, xs_total, xs_scatter, xs_fission)
    assert abs(keff - 1.034970) < 1e-4, str(keff) + " not infinite value"


@pytest.mark.infinite
@pytest.mark.power_iteration
def test_two_group_uranium_reactor_01d_chi():
    angles = 20
    xs_total = np.array([2.13800, 0.650917])
    xs_scatter = np.array([[2.06880, 0.0], [0.0342008, 0.0]]).T
    chi = np.array([[0.0, 1.0]])
    nu = np.array([[2.50, 1.004]])
    sigmaf = np.array([[0.045704, 0.61475]])
    nusigf = (nu * sigmaf).flatten()
    _, keff = power_iteration(angles, xs_total, xs_scatter, nusigf, chi.flatten())
    assert abs(keff - 1.034970) < 1e-4, str(keff) + " not infinite value"


@pytest.mark.infinite
@pytest.mark.power_iteration
def test_two_group_heavy_water_01():
    # UD20-2-0-IN (Problem 67): k_inf = 1.000196
    angles = 20
    xs_total = np.array([0.54628, 0.33588])
    xs_scatter = np.array([[0.42410, 0.0], [0.004555, 0.31980]]).T
    chi = np.array([[0.0], [1.0]])
    nu = np.array([[2.50, 2.50]])
    sigmaf = np.array([[0.097, 0.002817]])
    xs_fission = chi @ (nu * sigmaf)
    _, keff = power_iteration(angles, xs_total, xs_scatter, xs_fission)
    assert abs(keff - 1.000196) < 1e-4, str(keff) + " not infinite value"


@pytest.mark.infinite
@pytest.mark.power_iteration
def test_two_group_heavy_water_01_chi():
    angles = 20
    xs_total = np.array([0.54628, 0.33588])
    xs_scatter = np.array([[0.42410, 0.0], [0.004555, 0.31980]]).T
    chi = np.array([[0.0, 1.0]])
    nu = np.array([[2.50, 2.50]])
    sigmaf = np.array([[0.097, 0.002817]])
    nusigf = (nu * sigmaf).flatten()
    _, keff = power_iteration(angles, xs_total, xs_scatter, nusigf, chi.flatten())
    assert abs(keff - 1.000196) < 1e-4, str(keff) + " not infinite value"


################################################################################
# Two Group Anisotropic Scattering
################################################################################


@pytest.mark.anisotropic
@pytest.mark.infinite
@pytest.mark.power_iteration
def test_two_group_aniso_uranium_reactor_01a():
    # URRa-2-1-IN (Problem 70): k_inf = 1.631452
    # Same as Problem 53 (URRa-2-0-IN); k_inf is invariant to anisotropic moments
    # in infinite medium. L1 moments from the same cross-section set as Problem 71.
    angles = 20
    xs_total = np.array([2.52025, 0.65696])
    xs_scatter = np.stack(
        [
            np.array([[2.44383, 0.0], [0.029227, 0.62568]]).T,  # L0
            np.array([[0.83318, 0.0], [0.0075737, 0.27459]]).T,  # L1
        ],
        axis=-1,
    )
    chi = np.array([[0.0], [1.0]])
    nu = np.array([[2.5, 2.5]])
    sigmaf = np.array([[0.050632, 0.0010484]])
    xs_fission = chi @ (nu * sigmaf)
    _, keff = power_iteration(angles, xs_total, xs_scatter, xs_fission)
    assert abs(keff - 1.631452) < 1e-4, str(keff) + " not infinite value"


@pytest.mark.anisotropic
@pytest.mark.infinite
@pytest.mark.power_iteration
def test_two_group_aniso_uranium_reactor_01a_chi():
    angles = 20
    xs_total = np.array([2.52025, 0.65696])
    xs_scatter = np.stack(
        [
            np.array([[2.44383, 0.0], [0.029227, 0.62568]]).T,
            np.array([[0.83318, 0.0], [0.0075737, 0.27459]]).T,
        ],
        axis=-1,
    )
    chi = np.array([[0.0, 1.0]])
    nu = np.array([[2.5, 2.5]])
    sigmaf = np.array([[0.050632, 0.0010484]])
    nusigf = (nu * sigmaf).flatten()
    _, keff = power_iteration(angles, xs_total, xs_scatter, nusigf, chi.flatten())
    assert abs(keff - 1.631452) < 1e-4, str(keff) + " not infinite value"


@pytest.mark.anisotropic
@pytest.mark.infinite
@pytest.mark.power_iteration
def test_two_group_aniso_heavy_water_01():
    # UD20-2-1-IN (Problem 72): k_inf = 1.000196
    # Same as Problem 67 (UD20-2-0-IN); k_inf is invariant to anisotropic moments
    # in infinite medium. L1 moments from Tables 56/57 of LA-13511.
    angles = 20
    xs_total = np.array([0.54628, 0.33588])
    xs_scatter = np.stack(
        [
            np.array([[0.42410, 0.0], [0.004555, 0.31980]]).T,  # L0
            np.array([[0.06694, -0.0003972], [0.0, 0.05439]]).T,  # L1
        ],
        axis=-1,
    )
    chi = np.array([[0.0], [1.0]])
    nu = np.array([[2.50, 2.50]])
    sigmaf = np.array([[0.097, 0.002817]])
    xs_fission = chi @ (nu * sigmaf)
    _, keff = power_iteration(angles, xs_total, xs_scatter, xs_fission)
    assert abs(keff - 1.000196) < 1e-4, str(keff) + " not infinite value"


@pytest.mark.anisotropic
@pytest.mark.infinite
@pytest.mark.power_iteration
def test_two_group_aniso_heavy_water_01_chi():
    angles = 20
    xs_total = np.array([0.54628, 0.33588])
    xs_scatter = np.stack(
        [
            np.array([[0.42410, 0.0], [0.004555, 0.31980]]).T,
            np.array([[0.06694, -0.0003972], [0.0, 0.05439]]).T,
        ],
        axis=-1,
    )
    chi = np.array([[0.0, 1.0]])
    nu = np.array([[2.50, 2.50]])
    sigmaf = np.array([[0.097, 0.002817]])
    nusigf = (nu * sigmaf).flatten()
    _, keff = power_iteration(angles, xs_total, xs_scatter, nusigf, chi.flatten())
    assert abs(keff - 1.000196) < 1e-4, str(keff) + " not infinite value"
