"""Unit tests for discrete1.utils.hybrid helpers.

These tests focus on small, deterministic arrays to validate
indexing and coarsening behaviors without invoking large solvers.
"""

import numpy as np

import discrete1
from discrete1.utils import hybrid


def test_energy_coarse_index_even_and_remainder():
    # Even split
    idx = hybrid.energy_coarse_index(8, 4)
    assert np.array_equal(idx, np.array([0, 2, 4, 6, 8], dtype=np.int32))

    # Remainder split (10 -> 4): increments should be [3, 2, 2, 3]
    idx = hybrid.energy_coarse_index(10, 4)
    assert idx.dtype == np.int32
    assert idx[0] == 0 and idx[-1] == 10
    assert np.all(np.diff(idx) >= 2)  # at least floor(10/4)
    assert np.array_equal(idx, np.array([0, 3, 5, 7, 10], dtype=np.int32))


def test_coarsen_external_unweighted_and_weighted():
    # Fine edges and group widths
    edges_g = np.array([0.0, 1.0, 3.0, 6.0, 10.0])  # widths [1, 2, 3, 4]
    edges_gidx = np.array([0, 2, 4], dtype=np.int32)  # two coarse groups: [0:2], [2:4]

    # Make an array with last axis as groups; other dims arbitrary
    # Shape (2, 3, 4)
    ext = np.zeros((2, 3, 4))
    # Fill last axis with simple increasing values: 1, 2, 3, 4
    ext[..., 0] = 1.0
    ext[..., 1] = 2.0
    ext[..., 2] = 3.0
    ext[..., 3] = 4.0

    # Unweighted coarsen should sum groups inside the coarse bins
    out = hybrid.coarsen_external(ext, edges_g, edges_gidx, weight=False)
    assert out.shape == (2, 3, 2)
    # Expected: [1+2, 3+4] = [3, 7]
    assert np.allclose(out[..., 0], 3.0)
    assert np.allclose(out[..., 1], 7.0)

    # Weighted: multiply by fine widths then divide by coarse width
    # Coarse widths are [3, 7]
    out_w = hybrid.coarsen_external(ext, edges_g, edges_gidx, weight=True)
    expected0 = (1.0 * 1.0 + 2.0 * 2.0) / 3.0  # (1 + 4) / 3 = 5/3
    expected1 = (3.0 * 3.0 + 4.0 * 4.0) / 7.0  # (9 + 16) / 7 = 25/7
    assert np.allclose(out_w[..., 0], expected0)
    assert np.allclose(out_w[..., 1], expected1)


def test_coarsen_external_single_group_identity():
    ext = np.random.rand(2, 3, 1)
    edges_g = np.array([0.0, 1.0])
    edges_gidx = np.array([0, 1], dtype=np.int32)
    out = hybrid.coarsen_external(ext, edges_g, edges_gidx, weight=True)
    assert np.array_equal(out, ext)


def test_coarsen_velocity_means():
    v = np.array([10.0, 20.0, 30.0, 40.0])
    edges_gidx = np.array([0, 2, 4], dtype=np.int32)
    out = hybrid.coarsen_velocity(v, edges_gidx)
    assert np.allclose(out, np.array([15.0, 35.0]))


def test_xs_vector_and_matrix_coarsen_via_materials():
    # Two materials, four fine groups -> two coarse groups
    materials = 2
    groups = 4
    edges_g = np.array([0.0, 1.0, 3.0, 6.0, 10.0])  # widths [1, 2, 3, 4]
    edges_gidx = np.array([0, 2, 4], dtype=np.int32)

    # xs_total: shape (materials, groups)
    xs_total = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],  # mat 0
            [2.0, 4.0, 6.0, 8.0],  # mat 1
        ]
    )

    # xs_scatter, xs_fission: shape (materials, groups, groups)
    # Use ones to make expected sums analytic.
    xs_scatter = np.ones((materials, groups, groups))
    xs_fission = np.ones((materials, groups, groups)) * 2.0

    ct, cs, cf = hybrid.coarsen_materials(
        xs_total, xs_scatter, xs_fission, edges_g, edges_gidx
    )

    # Vector coarsen expected values (width-weighted average over source groups)
    # coarse[mat,0] = (x0*1 + x1*2)/3; coarse[mat,1] = (x2*3 + x3*4)/7
    exp_ct_mat0 = np.array([(1 * 1 + 2 * 2) / 3, (3 * 3 + 4 * 4) / 7])
    exp_ct_mat1 = np.array([(2 * 1 + 4 * 2) / 3, (6 * 3 + 8 * 4) / 7])
    assert np.allclose(ct[0], exp_ct_mat0)
    assert np.allclose(ct[1], exp_ct_mat1)

    # Matrix coarsen with xs_scatter = 1s
    # fine = matrix * delta_fine along source axis (rows)
    # Block sums then divided by coarse width of target group (columns)
    # widths for reference (not needed further):
    # delta_f = np.diff(edges_g)  # [1,2,3,4]
    # delta_c = np.diff(edges_g[edges_gidx])  # [3,7]

    # For ones matrix, the coarsened blocks are all 2.
    # Reason: fine = ones * delta_fine (applied to target columns),
    # summing over source rows in a block gives 2 * sum(delta_fine in target block),
    # dividing by the coarse target width cancels and yields 2.
    exp_cs = np.full((materials, 2, 2), 2.0)
    assert np.allclose(cs, exp_cs)

    # xs_fission = 2s => each entry doubles to 4
    exp_cf = np.full((materials, 2, 2), 4.0)
    assert np.allclose(cf, exp_cf)


def test_xs_matrix_coarsen_none_returns_none():
    # Access private for coverage
    out = hybrid._xs_matrix_coarsen(
        None, np.array([0.0, 1.0]), np.array([0, 1], dtype=np.int32)
    )
    assert out is None


def test_indexing_and_factor_small_case():
    # Fine and coarse mapping for a 4->2 example
    edges_g = np.array([0.0, 3.0, 5.0, 8.0, 10.0])  # widths [3,2,3,2]
    edges_gidx_f = hybrid.energy_coarse_index(4, 4)  # [0,1,2,3,4]
    edges_gidx_c = np.array([0, 2, 4], dtype=np.int32)

    fine_idx, coarse_idx, factor = hybrid.indexing(edges_g, edges_gidx_f, edges_gidx_c)

    # Edges between coarse/fine grids
    assert np.array_equal(fine_idx, np.array([0, 2, 4], dtype=np.int32))
    # Coarse group of each fine group
    assert np.array_equal(coarse_idx, np.array([0, 0, 1, 1], dtype=np.int32))
    # Factor = fine_width / coarse_width at each fine location
    expected_factor = np.array([3.0 / 5.0, 2.0 / 5.0, 3.0 / 5.0, 2.0 / 5.0])
    assert np.allclose(factor, expected_factor)


def test_energy_grid_change_matches_indexing():
    # Use the packaged 87-group grid but small problem groups for speed
    starting_grid = 87
    groups_u = 4
    groups_c = 2

    # energy_grid_change should align with indexing() over the same grid
    coarse_idx, factor, edges_gidx_c, edges_g = hybrid.energy_grid_change(
        starting_grid, groups_u, groups_c
    )

    edges_g_from_grid, edges_gidx_u, edges_gidx_c2 = discrete1.energy_grid(
        starting_grid, groups_u, groups_c
    )

    fine_idx2, coarse_idx2, factor2 = hybrid.indexing(
        edges_g_from_grid, edges_gidx_u, edges_gidx_c2
    )

    assert np.array_equal(coarse_idx, coarse_idx2)
    assert np.allclose(factor, factor2)
    assert np.array_equal(edges_gidx_c, edges_gidx_c2)
    assert np.allclose(edges_g, edges_g_from_grid)
