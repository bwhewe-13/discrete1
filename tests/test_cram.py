"""Unit tests for the CRAM matrix-exponential solver.

These tests check the Chebyshev Rational Approximation Method against known
analytic Bateman solutions and against a dense reference matrix exponential.
"""

import numpy as np
import pytest
from scipy.linalg import expm

from discrete1.cram import cram_expm

pytestmark = pytest.mark.depletion


def _decay_chain_matrix(lambdas):
    """Build a linear decay-chain matrix A[i+1, i] = lam_i, A[i, i] = -lam_i."""
    m = len(lambdas)
    A = np.zeros((m, m))
    for i, lam in enumerate(lambdas):
        A[i, i] -= lam
        if i + 1 < m:
            A[i + 1, i] += lam
    return A


@pytest.mark.parametrize("order", [16, 48])
@pytest.mark.parametrize("dt", [1e-2, 1.0, 1e3, 1e7])
def test_single_decay(order, dt):
    # N(t) = N0 exp(-lambda t)
    lam = 1.0e-3
    A = np.array([[-lam]])
    n0 = np.array([7.0])
    got = cram_expm(A, n0, dt, order=order)
    assert got[0] == pytest.approx(n0[0] * np.exp(-lam * dt), rel=1e-10)


@pytest.mark.parametrize("order", [16, 48])
def test_two_step_chain_analytic(order):
    # A -> B -> C (C stable). Analytic Bateman for B and C.
    la, lb = 0.05, 0.02
    A = _decay_chain_matrix([la, lb, 0.0])
    n0 = np.array([1.0, 0.0, 0.0])
    t = 12.0
    got = cram_expm(A, n0, t, order=order)

    na = np.exp(-la * t)
    nb = la / (lb - la) * (np.exp(-la * t) - np.exp(-lb * t))
    nc = 1.0 - na - nb
    assert got[0] == pytest.approx(na, rel=1e-9)
    assert got[1] == pytest.approx(nb, rel=1e-9)
    assert got[2] == pytest.approx(nc, rel=1e-9)


@pytest.mark.parametrize("order", [16, 48])
def test_matches_dense_expm_stiff(order):
    # Stiff chain spanning many decay-constant decades vs dense expm.
    lambdas = [1e4, 1e1, 1e-2, 5.0, 0.0, 3e2]
    A = _decay_chain_matrix(lambdas)
    n0 = np.zeros(len(lambdas))
    n0[0] = 1.0
    for dt in [1e-3, 1.0, 1e4]:
        ref = expm(A * dt) @ n0
        got = cram_expm(A, n0, dt, order=order)
        assert np.allclose(got, ref, atol=1e-12, rtol=1e-9)


def test_conserves_mass_closed_chain():
    # A pure decay chain conserves total atom count.
    A = _decay_chain_matrix([0.1, 0.3, 0.7, 0.0])
    n0 = np.array([1.0, 0.0, 0.0, 0.0])
    got = cram_expm(A, n0, 25.0, order=48)
    assert got.sum() == pytest.approx(1.0, rel=1e-12)
    assert np.all(got >= -1e-12)


def test_substeps_match_single_step():
    A = _decay_chain_matrix([1e3, 1e-1, 1e-5, 2e2])
    n0 = np.zeros(4)
    n0[0] = 1.0
    single = cram_expm(A, n0, 1e5, order=48, substeps=1)
    many = cram_expm(A, n0, 1e5, order=48, substeps=20)
    assert np.allclose(single, many, atol=1e-12, rtol=1e-9)


def test_invalid_order_raises():
    A = np.array([[-1.0]])
    with pytest.raises(ValueError):
        cram_expm(A, np.array([1.0]), 1.0, order=32)


def test_invalid_substeps_raises():
    A = np.array([[-1.0]])
    with pytest.raises(ValueError):
        cram_expm(A, np.array([1.0]), 1.0, order=48, substeps=0)
