"""Tests for the energy-dependent (GxG) fission spectrum ``chi``.

Everywhere ``chi`` can be a rank-1 vector shared across incident groups, it
can now instead be a matrix indexed ``[..., g_in, g_out]`` (matching PyCNiC's
``Material.chi_matrix`` convention). These tests check that:

- a degenerate matrix (every incident-group row identical) reproduces the
  existing rank-1 path exactly, at both the numba-kernel level and through
  the ``critical0d``/``critical1d`` drivers, and
- a genuinely energy-dependent spectrum produces the hand-computable
  fission source / keff update.
"""

import numpy as np
import pytest

from discrete1 import tools
from discrete1.critical0d import power_iteration as power_iteration_0d


def test_transfer_matrix_degenerate_chi_matches_vector():
    xs_scatter = np.array([[[0.1, 0.02], [0.03, 0.2]]])
    nusigf = np.array([[0.5, 1.5]])
    chi_vec = np.array([[0.4, 0.6]])
    chi_mat = np.broadcast_to(chi_vec[:, None, :], (1, 2, 2)).copy()

    m_vec = tools.transfer_matrix(xs_scatter, nusigf, chi_vec)
    m_mat = tools.transfer_matrix(xs_scatter, nusigf, chi_mat)
    assert np.allclose(m_vec, m_mat)


def test_fission_vec_prod_degenerate_chi_matches_vector():
    flux = np.array([[1.0, 2.0], [0.5, 1.5]])
    medium_map = np.array([0, 0], dtype=np.int32)
    nusigf = np.array([[0.5, 1.5]])
    chi_vec = np.array([[0.4, 0.6]])
    chi_mat = np.broadcast_to(chi_vec[:, None, :], (1, 2, 2)).copy()
    keff = 1.1

    source_vec = np.zeros((2, 1, 2))
    tools.fission_vec_prod(flux, chi_vec, nusigf, source_vec, medium_map, keff)
    source_mat = np.zeros((2, 1, 2))
    tools.fission_vec_prod_echi(flux, chi_mat, nusigf, source_mat, medium_map, keff)
    assert np.allclose(source_vec, source_mat)


def test_update_keff_vec_degenerate_chi_matches_vector():
    flux = np.array([[1.0, 2.0], [0.5, 1.5]])
    flux_old = np.array([[0.9, 1.8], [0.4, 1.3]])
    medium_map = np.array([0, 0], dtype=np.int32)
    nusigf = np.array([[0.5, 1.5]])
    chi_vec = np.array([[0.4, 0.6]])
    chi_mat = np.broadcast_to(chi_vec[:, None, :], (1, 2, 2)).copy()
    keff = 1.1

    k_vec = tools._update_keff_vec(flux, flux_old, chi_vec, nusigf, medium_map, keff)
    k_mat = tools._update_keff_vec_echi(
        flux, flux_old, chi_mat, nusigf, medium_map, keff
    )
    assert k_vec == pytest.approx(k_mat)


def test_fission_source_energy_dependent_chi_hand_computed():
    # Fast fission (g_in=0) mostly reproduces a fast neutron; thermal fission
    # (g_in=1) is softer -- a genuinely energy-dependent spectrum.
    chi = np.array([[[0.9, 0.1], [0.2, 0.8]]])
    nusigf = np.array([[0.5, 1.5]])
    flux = np.array([[2.0, 3.0]])
    medium_map = np.array([0], dtype=np.int32)
    keff = 1.0

    source = np.zeros((1, 1, 2))
    tools.fission_vec_prod_echi(flux, chi, nusigf, source, medium_map, keff)

    expected_og0 = (
        flux[0, 0] * chi[0, 0, 0] * nusigf[0, 0]
        + flux[0, 1] * chi[0, 1, 0] * nusigf[0, 1]
    )
    expected_og1 = (
        flux[0, 0] * chi[0, 0, 1] * nusigf[0, 0]
        + flux[0, 1] * chi[0, 1, 1] * nusigf[0, 1]
    )
    assert source[0, 0, 0] == pytest.approx(expected_og0)
    assert source[0, 0, 1] == pytest.approx(expected_og1)


@pytest.mark.infinite
@pytest.mark.power_iteration
def test_two_group_plutonium_01_chi_matrix_0d():
    # Same benchmark as test_two_group_plutonium_01_chi in
    # test_critical_0d.py, but chi is (groups, groups) with both
    # incident-group rows equal to the rank-1 spectrum -- degenerate, so it
    # must reproduce the same keff.
    angles = 20
    xs_total = np.array([0.3360, 0.2208])
    xs_scatter = np.array([[0.23616, 0.0], [0.0432, 0.0792]]).T
    chi_vec = np.array([0.425, 0.575])
    chi = np.array([chi_vec, chi_vec])
    nu = np.array([2.93, 3.10])
    sigmaf = np.array([0.08544, 0.0936])
    nusigf = nu * sigmaf
    _, keff = power_iteration_0d(angles, xs_total, xs_scatter, nusigf, chi=chi)
    assert abs(keff - 2.683767) < 1e-4, str(keff) + " not infinite value"
