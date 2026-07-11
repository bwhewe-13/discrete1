"""Unit tests for burnup-matrix assembly and single-region depletion."""

import numpy as np
import pytest

import discrete1.constants as const
from discrete1 import depletion
from discrete1.nuclides import synthetic_library

pytestmark = pytest.mark.depletion


def test_reaction_rate_1g():
    # rate = sum_g sigma_g phi_g * barn-to-cm^2
    xs = np.array([[2.0, 3.0], [0.0, 1.0]])  # barns, (M=2, G=2)
    flux = np.array([1e14, 2e14])
    rate = depletion.reaction_rate_1g(xs, flux)
    expected = np.array([2.0 * 1e14 + 3.0 * 2e14, 1.0 * 2e14]) * const.BARNS_TO_CM2
    assert np.allclose(rate, expected)


def test_burnup_matrix_decay_only():
    # Zero flux: only the fp -> fp_stable decay is active.
    lib = synthetic_library(groups=1)
    A = depletion.build_burnup_matrix(lib, np.zeros(1)).toarray()
    i = lib.index
    lam = np.log(2.0) / lib.half_life[i["fp"]]
    assert A[i["fp_stable"], i["fp"]] == pytest.approx(lam)
    assert A[i["fp"], i["fp"]] == pytest.approx(-lam)
    # Stable nuclides have no loss term
    assert A[i["fuel"], i["fuel"]] == pytest.approx(0.0)
    # Closed decay conserves atoms -> every column sums to zero
    assert np.allclose(A.sum(axis=0), 0.0)


def test_burnup_matrix_with_flux():
    lib = synthetic_library(groups=1)
    flux = np.array([1e14])
    A = depletion.build_burnup_matrix(lib, flux).toarray()
    i = lib.index

    cap = lib.reaction_xs["(n,gamma)"][i["fuel"], 0] * flux[0] * const.BARNS_TO_CM2
    fis = lib.fission_xs[i["fuel"], 0] * flux[0] * const.BARNS_TO_CM2
    lam = np.log(2.0) / lib.half_life[i["fp"]]

    # Capture moves fuel -> fuel2
    assert A[i["fuel2"], i["fuel"]] == pytest.approx(cap)
    # Fission produces 2 fp per fission of fuel
    assert A[i["fp"], i["fuel"]] == pytest.approx(2.0 * fis)
    # Fuel diagonal loss = capture + fission (fuel is stable)
    assert A[i["fuel"], i["fuel"]] == pytest.approx(-(cap + fis))
    # Net fuel column gain equals fission rate (two products per one absorption)
    assert A[:, i["fuel"]].sum() == pytest.approx(fis)
    # fp column: produced from fuel, lost to decay
    assert A[i["fp"], i["fp"]] == pytest.approx(-lam)


def test_burnup_matrix_skips_untracked_products():
    # decay_to / fy_product of -1 means the product is not tracked: the parent
    # still loses atoms, but nothing may be credited elsewhere (a negative
    # index would silently feed the *last* nuclide).
    lib = synthetic_library(groups=1)
    i = lib.index
    lib.decay_to = np.array([-1], dtype=np.int64)  # fp daughter untracked
    lib.fy_product = np.array([-1], dtype=np.int64)  # fission product untracked
    lib.validate()

    flux = np.array([1e14])
    A = depletion.build_burnup_matrix(lib, flux).toarray()
    lam = np.log(2.0) / lib.half_life[i["fp"]]
    fis = lib.fission_xs[i["fuel"], 0] * flux[0] * const.BARNS_TO_CM2

    # Losses unchanged
    assert A[i["fp"], i["fp"]] == pytest.approx(-lam)
    assert A[i["fuel"], i["fuel"]] < 0.0
    # Nothing lands on the last nuclide (index -1 target of both couplings)
    last = lib.n_nuclides - 1
    assert A[last, i["fp"]] == 0.0
    assert A[last, i["fuel"]] == 0.0
    # Fission still removes fuel even though its product is untracked
    assert -A[i["fuel"], i["fuel"]] >= fis


def test_deplete_decay_conserves_atoms():
    # No flux: fp decays to fp_stable, total atoms conserved.
    lib = synthetic_library(groups=1)
    n0 = np.array([1.0, 0.0, 0.5, 0.0])  # fuel, fuel2, fp, fp_stable
    n1 = depletion.deplete(lib, n0, np.zeros(1), dt=3.0e4, order=48)
    assert n1.sum() == pytest.approx(n0.sum(), rel=1e-10)
    assert n1[lib.index["fp"]] < n0[lib.index["fp"]]
    assert n1[lib.index["fp_stable"]] > 0.0


def test_deplete_burns_fuel_under_flux():
    lib = synthetic_library(groups=1)
    n0 = np.array([0.04, 0.0, 0.0, 0.0])
    flux = np.array([5e14])
    n1 = depletion.deplete(lib, n0, flux, dt=2.6e6, order=48)
    # Fuel is consumed; fuel2 (capture) and fission products appear
    assert n1[lib.index["fuel"]] < n0[lib.index["fuel"]]
    assert n1[lib.index["fuel2"]] > 0.0
    assert n1[lib.index["fp"]] + n1[lib.index["fp_stable"]] > 0.0
    assert np.all(n1 >= -1e-15)
