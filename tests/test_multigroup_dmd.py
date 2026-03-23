"""DMD tests for multigroup drivers and low-level extrapolation helper.

These tests validate two things:
1) ``tools.dmd`` improves an iterate for a linear fixed-point sequence.
2) ``multi_group.dynamic_mode_decomp`` produces a flux close to the
   converged ``multi_group.source_iteration`` solution.
"""

import numpy as np
import pytest

import discrete1
from discrete1 import tools
from discrete1.multi_group import dynamic_mode_decomp, source_iteration


def _build_linear_dmd_snapshots(A, b, x0, K):
    """Build 2D DMD snapshots matching the solver's snapshot convention."""
    n_state = x0.size
    states = np.zeros((K + 1, n_state), dtype=np.float64)
    states[0] = x0.copy()

    for ii in range(K):
        states[ii + 1] = A @ states[ii] + b

    diffs = states[1:] - states[:-1]  # shape (K, n_state)

    # tools.dmd expects (n_state, K-1) matrices.
    y_minus = diffs[:-1].T
    y_plus = diffs[1:].T

    return states, y_minus, y_plus


@pytest.mark.dmd
def test_tools_dmd_improves_linear_fixed_point_estimate():
    """DMD should move the iterate closer to the true linear fixed point."""
    # Stable linear map x_{k+1} = A x_k + b with known fixed point.
    A = np.array(
        [
            [0.50, 0.10, 0.00, 0.00],
            [0.05, 0.45, 0.05, 0.00],
            [0.00, 0.08, 0.40, 0.05],
            [0.00, 0.00, 0.06, 0.35],
        ],
        dtype=np.float64,
    )
    b = np.array([0.30, 0.10, 0.05, 0.02], dtype=np.float64)

    K = 6
    shape = (2, 2)
    x0 = np.array([0.9, 0.2, 0.1, 0.7], dtype=np.float64)

    states, y_minus, y_plus = _build_linear_dmd_snapshots(A, b, x0, K)
    flux_old = states[K].reshape(shape)

    dmd_flux = tools.dmd(flux_old, y_minus, y_plus, K).ravel()
    x_star = np.linalg.solve(np.eye(A.shape[0]) - A, b)

    err_before = np.linalg.norm(states[K] - x_star)
    err_after = np.linalg.norm(dmd_flux - x_star)

    assert np.isfinite(dmd_flux).all()
    assert err_after < err_before


@pytest.mark.multigroup
@pytest.mark.dmd
def test_multigroup_dmd_matches_source_iteration_solution():
    """Multigroup DMD should be close to converged source iteration."""
    cells_x = 40
    groups = 3
    angles = 8
    bc_x = [0, 0]

    angle_x, angle_w = discrete1.angular_x(angles, bc_x)

    flux_init = np.ones((cells_x, groups), dtype=np.float64)

    xs_total = np.array([[1.00, 0.90, 0.80]], dtype=np.float64)
    xs_scatter = np.array(
        [
            [
                [0.30, 0.05, 0.00],
                [0.06, 0.25, 0.04],
                [0.02, 0.05, 0.20],
            ]
        ],
        dtype=np.float64,
    )

    external = np.zeros((cells_x, 1, groups), dtype=np.float64)
    external[:, 0, 0] = 1.00
    external[:, 0, 1] = 0.40
    external[:, 0, 2] = 0.15

    boundary = np.zeros((2, 1, groups), dtype=np.float64)
    medium_map = np.zeros((cells_x,), dtype=np.int32)
    delta_x = np.repeat(5.0 / cells_x, cells_x)

    flux_si = source_iteration(
        flux_init.copy(),
        xs_total,
        xs_scatter,
        external,
        boundary,
        medium_map,
        delta_x,
        angle_x,
        angle_w,
        bc_x,
        geometry=1,
    )

    flux_dmd = dynamic_mode_decomp(
        flux_init.copy(),
        xs_total,
        xs_scatter,
        external,
        boundary,
        medium_map,
        delta_x,
        angle_x,
        angle_w,
        bc_x,
        geometry=1,
        R=2,
        K=5,
    )

    rel_diff = np.linalg.norm(flux_dmd - flux_si) / np.linalg.norm(flux_si)
    err_init = np.linalg.norm(flux_init - flux_si)
    err_dmd = np.linalg.norm(flux_dmd - flux_si)

    assert np.isfinite(flux_dmd).all()
    assert rel_diff < 5e-2
    assert err_dmd < err_init


@pytest.mark.multigroup
@pytest.mark.dmd
def test_multigroup_anisotropic_dmd_matches_source_iteration_solution():
    """Anisotropic multigroup DMD should be close to source iteration."""
    cells_x = 40
    groups = 3
    moments = 2
    angles = 8
    bc_x = [0, 0]

    angle_x, angle_w = discrete1.angular_x(angles, bc_x)

    flux_init = np.ones((cells_x, groups), dtype=np.float64)

    xs_total = np.array([[1.00, 0.90, 0.80]], dtype=np.float64)
    xs_scatter = np.zeros((1, groups, groups, moments), dtype=np.float64)

    # Isotropic (L=0) scatter moments
    xs_scatter[0, :, :, 0] = np.array(
        [
            [0.30, 0.05, 0.00],
            [0.06, 0.25, 0.04],
            [0.02, 0.05, 0.20],
        ],
        dtype=np.float64,
    )

    # First anisotropic moment (L=1), small to preserve stability
    xs_scatter[0, :, :, 1] = np.array(
        [
            [0.03, 0.00, 0.00],
            [0.01, 0.02, 0.00],
            [0.00, 0.01, 0.02],
        ],
        dtype=np.float64,
    )

    external = np.zeros((cells_x, 1, groups), dtype=np.float64)
    external[:, 0, 0] = 1.00
    external[:, 0, 1] = 0.40
    external[:, 0, 2] = 0.15

    boundary = np.zeros((2, 1, groups), dtype=np.float64)
    medium_map = np.zeros((cells_x,), dtype=np.int32)
    delta_x = np.repeat(5.0 / cells_x, cells_x)

    flux_si = source_iteration(
        flux_init.copy(),
        xs_total,
        xs_scatter,
        external,
        boundary,
        medium_map,
        delta_x,
        angle_x,
        angle_w,
        bc_x,
        geometry=1,
    )

    flux_dmd = dynamic_mode_decomp(
        flux_init.copy(),
        xs_total,
        xs_scatter,
        external,
        boundary,
        medium_map,
        delta_x,
        angle_x,
        angle_w,
        bc_x,
        geometry=1,
        R=2,
        K=5,
    )

    rel_diff = np.linalg.norm(flux_dmd - flux_si) / np.linalg.norm(flux_si)
    err_init = np.linalg.norm(flux_init - flux_si)
    err_dmd = np.linalg.norm(flux_dmd - flux_si)

    assert np.isfinite(flux_dmd).all()
    assert rel_diff < 8e-2
    assert err_dmd < err_init
