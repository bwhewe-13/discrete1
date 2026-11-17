"""Tests for hybrid_power_iteration and ml_power_iteration in critical1d.

Verifies:
  - ml_power_iteration with no models falls back to standard power_iteration
  - hybrid_power_iteration with groups_c == groups_u and angles_c == angles_u
    converges to the same keff as standard power_iteration
  - ml_power_iteration with exact-operator models reproduces power_iteration
    for both joint_keff estimates
"""

import numpy as np
import pytest

import discrete1
from discrete1 import multi_group as mg
from discrete1.critical1d import (
    hybrid_power_iteration,
    ml_power_iteration,
    power_iteration,
)
from discrete1.ml.tools import update_cross_sections


def _pu_1g_slab(bc_x):
    """1-group critical plutonium slab (LANL benchmark suite)."""
    cells_x = 50
    angles = 16
    angle_x, angle_w = discrete1.angular_x(angles, bc_x)
    xs_total = np.array([[0.32640]])
    xs_scatter = np.array([[[0.225216]]])
    xs_fission = np.array([[[3.24 * 0.0816]]])
    medium_map = np.zeros((cells_x,), dtype=np.int32)
    length = 1.853722 * 2 if np.sum(bc_x) == 0 else 1.853722
    delta_x = np.repeat(length / cells_x, cells_x)
    return xs_total, xs_scatter, xs_fission, medium_map, delta_x, angle_x, angle_w


def _identity_energy_grid(groups):
    """Trivial energy grid where coarse == fine (no actual coarsening)."""
    edges_g = np.linspace(0.0, float(groups), groups + 1)
    edges_gidx_u = np.arange(groups + 1, dtype=np.int32)
    edges_gidx_c = np.arange(groups + 1, dtype=np.int32)
    return edges_g, edges_gidx_u, edges_gidx_c


################################################################################
# ml_power_iteration — no-models fallback
################################################################################


@pytest.mark.smoke
@pytest.mark.slab
@pytest.mark.power_iteration
@pytest.mark.parametrize("bc_x", [[0, 0], [0, 1], [1, 0]])
def test_ml_power_iteration_no_models_agrees_with_power_iteration(bc_x):
    """ml_power_iteration with empty model lists falls back to standard power_iteration.

    Uses the 1-group critical Pu slab benchmark.
    """
    xs_total, xs_scatter, xs_fission, medium_map, delta_x, angle_x, angle_w = (
        _pu_1g_slab(bc_x)
    )
    cells_x = medium_map.shape[0]
    flux_old = np.random.rand(cells_x, xs_total.shape[1])

    _, keff_pi = power_iteration(
        xs_total,
        xs_scatter,
        xs_fission,
        medium_map,
        delta_x,
        angle_x,
        angle_w,
        bc_x,
    )
    _, keff_ml = ml_power_iteration(
        flux_old,
        xs_total,
        xs_scatter,
        xs_fission,
        medium_map,
        delta_x,
        angle_x,
        angle_w,
        bc_x,
    )

    assert abs(keff_ml - keff_pi) < 1e-3, f"ml: {keff_ml:.6f} vs pi: {keff_pi:.6f}"


################################################################################
# hybrid_power_iteration — identity (groups_c == groups_u, angles_c == angles_u)
################################################################################


@pytest.mark.smoke
@pytest.mark.slab
@pytest.mark.power_iteration
@pytest.mark.hybrid
@pytest.mark.parametrize("bc_x", [[0, 0], [0, 1], [1, 0]])
def test_hybrid_power_iteration_identity_agrees_with_power_iteration(bc_x):
    """hybrid_power_iteration with groups_c == groups_u and angles_c == angles_u.

    Converges to the same keff as standard power_iteration.
    """
    xs_total, xs_scatter, xs_fission, medium_map, delta_x, angle_x, angle_w = (
        _pu_1g_slab(bc_x)
    )
    angles = angle_x.shape[0]
    groups = xs_total.shape[1]

    energy_grid = _identity_energy_grid(groups)

    _, keff_pi = power_iteration(
        xs_total,
        xs_scatter,
        xs_fission,
        medium_map,
        delta_x,
        angle_x,
        angle_w,
        bc_x,
    )
    _, keff_hy = hybrid_power_iteration(
        xs_total,
        xs_scatter,
        xs_fission,
        medium_map,
        delta_x,
        angle_x,
        angle_w,
        bc_x,
        angles_c=angles,
        groups_c=groups,
        energy_grid=energy_grid,
    )

    assert abs(keff_hy - keff_pi) < 1e-3, f"hybrid: {keff_hy:.6f} vs pi: {keff_pi:.6f}"


################################################################################
# ml_power_iteration — joint_keff with exact-operator models
################################################################################


class _ExactModel:
    """Stand-in for a trained model that applies the true operator."""

    def __init__(self, matrix):
        self.matrix = matrix

    def predict(self, flux, label=None):
        return flux @ self.matrix.T


def _two_group_slab():
    """Scattering-dominated thermal group, so the inner iteration is slow."""
    cells_x = 80
    angle_x, angle_w = discrete1.angular_x(8, [0, 0])
    xs_total = np.array([[0.6, 1.5]])
    xs_scatter = np.array([[[0.5, 0.0], [0.05, 1.3]]])
    xs_fission = np.outer([1.0, 0.0], [0.01, 0.25])[None]
    medium_map = np.zeros((cells_x,), dtype=np.int32)
    delta_x = np.repeat(40.0 / cells_x, cells_x)
    return xs_total, xs_scatter, xs_fission, medium_map, delta_x, angle_x, angle_w


def _exact_ml_keff(joint_keff):
    xs_total, xs_scatter, xs_fission, medium_map, delta_x, angle_x, angle_w = (
        _two_group_slab()
    )
    flux_old = np.random.default_rng(0).random((medium_map.shape[0], 2))
    _, keff = ml_power_iteration(
        flux_old,
        xs_total,
        update_cross_sections(xs_scatter, [0]),
        update_cross_sections(xs_fission, [0]),
        medium_map,
        delta_x,
        angle_x,
        angle_w,
        [0, 0],
        fission_models=[_ExactModel(xs_fission[0])],
        scatter_models=[_ExactModel(xs_scatter[0])],
        joint_keff=joint_keff,
    )
    return keff


def _reference_keff():
    xs_total, xs_scatter, xs_fission, medium_map, delta_x, angle_x, angle_w = (
        _two_group_slab()
    )
    _, keff = power_iteration(
        xs_total,
        xs_scatter,
        xs_fission,
        medium_map,
        delta_x,
        angle_x,
        angle_w,
        [0, 0],
    )
    return keff


@pytest.mark.slab
@pytest.mark.power_iteration
def test_joint_keff_rate_matches_power_iteration():
    assert abs(_exact_ml_keff("rate") - _reference_keff()) < 1e-5


@pytest.mark.slab
@pytest.mark.power_iteration
def test_joint_keff_growth_matches_with_converged_inner_iteration(monkeypatch):
    # "growth" is only off by the cap on the inner ML source iteration
    monkeypatch.setattr(mg, "count_pp", 5000)
    assert abs(_exact_ml_keff("growth") - _reference_keff()) < 1e-5


def test_joint_keff_rejects_unknown_value():
    with pytest.raises(ValueError, match="joint_keff"):
        _exact_ml_keff("norm")
