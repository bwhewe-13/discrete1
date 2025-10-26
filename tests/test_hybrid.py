"""Hybrid problem tests.

Tests that validate hybrid (multiscale/multigroup) workflows.
"""

import numpy as np
import pytest

import discrete1
from discrete1 import hybrid1d, timed1d, vhybrid1d
from discrete1.utils import hybrid as hytools


def _example_problem_01(groups_u, groups_c, angles_u, angles_c):
    # Given parameters
    cells_x = 100
    steps = 5
    dt = 1e-8
    bc_x = [0, 0]

    # Spatial
    length_x = 10.0
    delta_x = np.repeat(length_x / cells_x, cells_x)
    edges_x = np.linspace(0, length_x, cells_x + 1)

    # Energy Grid
    energy_grid = discrete1.energy_grid(87, groups_u, groups_c)
    edges_g, edges_gidx_u, edges_gidx_c = energy_grid
    velocity_u = discrete1.energy_velocity(groups_u, edges_g)

    # Angular
    angle_xu, angle_wu = discrete1.angular_x(angles_u, bc_x)
    angle_xc, angle_wc = discrete1.angular_x(angles_c, bc_x)

    # Layout and Materials
    layout = [[0, "stainless-steel-440", "0-4, 6-10"], [1, "uranium-%20%", "4-6"]]
    medium_map = discrete1.spatial1d(layout, edges_x)

    # Cross Sections - Uncollided
    materials = np.array(layout)[:, 1]
    xs_total_u, xs_scatter_u, xs_fission_u = discrete1.materials(87, materials)

    # External and boundary sources
    external = np.zeros((1, cells_x, 1, 1))
    boundary = discrete1.boundary1d.deuterium_tritium(0, edges_g)
    edges_t = np.linspace(0, steps * dt, steps + 1)
    boundary = discrete1.boundary1d.time_dependence_decay_02(boundary, edges_t)

    flux_last = np.zeros((cells_x, angles_u, groups_u))

    return {
        "flux_last": flux_last,
        "xs_total_u": xs_total_u,
        "xs_scatter_u": xs_scatter_u,
        "xs_fission_u": xs_fission_u,
        "velocity_u": velocity_u,
        "external": external,
        "boundary": boundary,
        "medium_map": medium_map,
        "delta_x": delta_x,
        "angle_xu": angle_xu,
        "angle_xc": angle_xc,
        "angle_wu": angle_wu,
        "angle_wc": angle_wc,
        "bc_x": bc_x,
        "steps": steps,
        "dt": dt,
    }


def _get_hybrid_params(groups_u, groups_c, problem_dict):
    # Get hybrid parameters
    energy_grid = discrete1.energy_grid(87, groups_u, groups_c)
    edges_g, edges_gidx_u, edges_gidx_c = energy_grid
    fine_idx, coarse_idx, factor = hytools.indexing(*energy_grid)

    # Check for same number of energy groups
    if groups_u == groups_c:
        xs_total_c = problem_dict["xs_total_u"].copy()
        xs_scatter_c = problem_dict["xs_scatter_u"].copy()
        xs_fission_c = problem_dict["xs_fission_u"].copy()
        velocity_c = problem_dict["velocity_u"].copy()
    else:
        xs_collided = hytools.coarsen_materials(
            problem_dict["xs_total_u"],
            problem_dict["xs_scatter_u"],
            problem_dict["xs_fission_u"],
            edges_g[edges_gidx_u],
            edges_gidx_c,
        )
        xs_total_c, xs_scatter_c, xs_fission_c = xs_collided
        velocity_c = hytools.coarsen_velocity(problem_dict["velocity_u"], edges_gidx_c)

    return {
        "energy_grid": energy_grid,
        "edges_g": edges_g,
        "edges_gidx_c": edges_gidx_c,
        "fine_idx": fine_idx,
        "coarse_idx": coarse_idx,
        "factor": factor,
        "xs_total_c": xs_total_c,
        "xs_scatter_c": xs_scatter_c,
        "xs_fission_c": xs_fission_c,
        "velocity_c": velocity_c,
    }


@pytest.mark.slab
@pytest.mark.bdf1
@pytest.mark.multigroup
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_auto_bdf1_01():

    groups_u = 87
    groups_c = 87
    angles_u = 2
    angles_c = 2

    problem_dict = _example_problem_01(groups_u, groups_c, angles_u, angles_c)
    hybrid_dict = _get_hybrid_params(groups_u, groups_c, problem_dict)

    mg_flux = timed1d.backward_euler(
        problem_dict["flux_last"],
        problem_dict["xs_total_u"],
        problem_dict["xs_scatter_u"],
        problem_dict["xs_fission_u"],
        problem_dict["velocity_u"],
        problem_dict["external"],
        problem_dict["boundary"],
        problem_dict["medium_map"],
        problem_dict["delta_x"],
        problem_dict["angle_xu"],
        problem_dict["angle_wu"],
        problem_dict["bc_x"],
        steps=problem_dict["steps"],
        dt=problem_dict["dt"],
        geometry=1,
    )

    hy_flux = hybrid1d.auto_bdf1(
        problem_dict["flux_last"],
        problem_dict["xs_total_u"],
        problem_dict["xs_scatter_u"],
        problem_dict["xs_fission_u"],
        problem_dict["velocity_u"],
        problem_dict["external"],
        problem_dict["boundary"],
        problem_dict["medium_map"],
        problem_dict["delta_x"],
        problem_dict["angle_xu"],
        problem_dict["angle_wu"],
        problem_dict["bc_x"],
        angles_c,
        groups_c,
        hybrid_dict["energy_grid"],
        steps=problem_dict["steps"],
        dt=problem_dict["dt"],
        geometry=1,
    )

    for step in range(problem_dict["steps"]):
        assert np.sum(np.fabs(mg_flux[step] - hy_flux[step])) < 1e-7, (
            str(step) + " not equivalent"
        )


@pytest.mark.slab
@pytest.mark.bdf1
@pytest.mark.multigroup
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_backward_euler_01():

    groups_u = 87
    groups_c = 87
    angles_u = 2
    angles_c = 2

    problem_dict = _example_problem_01(groups_u, groups_c, angles_u, angles_c)
    hybrid_dict = _get_hybrid_params(groups_u, groups_c, problem_dict)

    mg_flux = timed1d.backward_euler(
        problem_dict["flux_last"],
        problem_dict["xs_total_u"],
        problem_dict["xs_scatter_u"],
        problem_dict["xs_fission_u"],
        problem_dict["velocity_u"],
        problem_dict["external"],
        problem_dict["boundary"],
        problem_dict["medium_map"],
        problem_dict["delta_x"],
        problem_dict["angle_xu"],
        problem_dict["angle_wu"],
        problem_dict["bc_x"],
        steps=problem_dict["steps"],
        dt=problem_dict["dt"],
        geometry=1,
    )

    hy_flux = hybrid1d.backward_euler(
        problem_dict["flux_last"],
        problem_dict["xs_total_u"],
        hybrid_dict["xs_total_c"],
        problem_dict["xs_scatter_u"],
        hybrid_dict["xs_scatter_c"],
        problem_dict["xs_fission_u"],
        hybrid_dict["xs_fission_c"],
        problem_dict["velocity_u"],
        hybrid_dict["velocity_c"],
        problem_dict["external"],
        problem_dict["boundary"],
        problem_dict["medium_map"],
        problem_dict["delta_x"],
        problem_dict["angle_xu"],
        problem_dict["angle_xc"],
        problem_dict["angle_wu"],
        problem_dict["angle_wc"],
        problem_dict["bc_x"],
        hybrid_dict["fine_idx"],
        hybrid_dict["coarse_idx"],
        hybrid_dict["factor"],
        steps=problem_dict["steps"],
        dt=problem_dict["dt"],
        geometry=1,
    )

    for step in range(problem_dict["steps"]):
        assert np.sum(np.fabs(mg_flux[step] - hy_flux[step])) < 1e-7, (
            str(step) + " not equivalent"
        )


@pytest.mark.slab
@pytest.mark.bdf1
@pytest.mark.multigroup
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize(
    ("groups_c", "angles_c", "varying"),
    [
        (87, 4, True),
        (43, 4, True),
        (43, 2, True),
        (87, 4, False),
        (43, 4, False),
        (43, 2, False),
    ],
)
def test_v_bdf1_01(groups_c, angles_c, varying):

    groups_u = 87
    angles_u = 4

    problem_dict = _example_problem_01(groups_u, groups_c, angles_u, angles_c)
    hybrid_dict = _get_hybrid_params(groups_u, groups_c, problem_dict)

    hy_flux = hybrid1d.auto_bdf1(
        problem_dict["flux_last"],
        problem_dict["xs_total_u"].copy(),
        problem_dict["xs_scatter_u"],
        problem_dict["xs_fission_u"],
        problem_dict["velocity_u"],
        problem_dict["external"],
        problem_dict["boundary"],
        problem_dict["medium_map"],
        problem_dict["delta_x"],
        problem_dict["angle_xu"],
        problem_dict["angle_wu"],
        problem_dict["bc_x"],
        angles_c,
        groups_c,
        hybrid_dict["energy_grid"],
        steps=problem_dict["steps"],
        dt=problem_dict["dt"],
        geometry=1,
    )

    if varying:
        groups_c = np.array([groups_c] * problem_dict["steps"])
        angles_c = np.array([angles_c] * problem_dict["steps"])

    vhy_flux = vhybrid1d.backward_euler(
        groups_c,
        angles_c,
        problem_dict["flux_last"],
        problem_dict["xs_total_u"],
        problem_dict["xs_scatter_u"],
        problem_dict["xs_fission_u"],
        problem_dict["velocity_u"],
        problem_dict["external"],
        problem_dict["boundary"],
        problem_dict["medium_map"],
        problem_dict["delta_x"],
        problem_dict["angle_xu"],
        problem_dict["angle_wu"],
        problem_dict["bc_x"],
        steps=problem_dict["steps"],
        dt=problem_dict["dt"],
        geometry=1,
        energy_grid=87,
    )

    for step in range(problem_dict["steps"]):
        assert np.sum(np.fabs(vhy_flux[step] - hy_flux[step])) < 1e-7, (
            str(step) + " not equivalent"
        )


@pytest.mark.slab
@pytest.mark.bdf2
@pytest.mark.multigroup
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_auto_bdf2_01():

    groups_u = 87
    groups_c = 87
    angles_u = 2
    angles_c = 2

    problem_dict = _example_problem_01(groups_u, groups_c, angles_u, angles_c)
    hybrid_dict = _get_hybrid_params(groups_u, groups_c, problem_dict)

    mg_flux = timed1d.bdf2(
        problem_dict["flux_last"],
        problem_dict["xs_total_u"],
        problem_dict["xs_scatter_u"],
        problem_dict["xs_fission_u"],
        problem_dict["velocity_u"],
        problem_dict["external"],
        problem_dict["boundary"],
        problem_dict["medium_map"],
        problem_dict["delta_x"],
        problem_dict["angle_xu"],
        problem_dict["angle_wu"],
        problem_dict["bc_x"],
        steps=problem_dict["steps"],
        dt=problem_dict["dt"],
        geometry=1,
    )

    hy_flux = hybrid1d.auto_bdf2(
        problem_dict["flux_last"],
        problem_dict["xs_total_u"],
        problem_dict["xs_scatter_u"],
        problem_dict["xs_fission_u"],
        problem_dict["velocity_u"],
        problem_dict["external"],
        problem_dict["boundary"],
        problem_dict["medium_map"],
        problem_dict["delta_x"],
        problem_dict["angle_xu"],
        problem_dict["angle_wu"],
        problem_dict["bc_x"],
        angles_c,
        groups_c,
        hybrid_dict["energy_grid"],
        steps=problem_dict["steps"],
        dt=problem_dict["dt"],
        geometry=1,
    )

    for step in range(problem_dict["steps"]):
        assert np.sum(np.fabs(mg_flux[step] - hy_flux[step])) < 5e-7, (
            str(step) + " not equivalent"
        )


@pytest.mark.slab
@pytest.mark.bdf2
@pytest.mark.multigroup
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_bdf2_01():

    groups_u = 87
    groups_c = 87
    angles_u = 2
    angles_c = 2

    problem_dict = _example_problem_01(groups_u, groups_c, angles_u, angles_c)
    hybrid_dict = _get_hybrid_params(groups_u, groups_c, problem_dict)

    mg_flux = timed1d.bdf2(
        problem_dict["flux_last"],
        problem_dict["xs_total_u"],
        problem_dict["xs_scatter_u"],
        problem_dict["xs_fission_u"],
        problem_dict["velocity_u"],
        problem_dict["external"],
        problem_dict["boundary"],
        problem_dict["medium_map"],
        problem_dict["delta_x"],
        problem_dict["angle_xu"],
        problem_dict["angle_wu"],
        problem_dict["bc_x"],
        steps=problem_dict["steps"],
        dt=problem_dict["dt"],
        geometry=1,
    )

    hy_flux = hybrid1d.bdf2(
        problem_dict["flux_last"],
        problem_dict["xs_total_u"],
        hybrid_dict["xs_total_c"],
        problem_dict["xs_scatter_u"],
        hybrid_dict["xs_scatter_c"],
        problem_dict["xs_fission_u"],
        hybrid_dict["xs_fission_c"],
        problem_dict["velocity_u"],
        hybrid_dict["velocity_c"],
        problem_dict["external"],
        problem_dict["boundary"],
        problem_dict["medium_map"],
        problem_dict["delta_x"],
        problem_dict["angle_xu"],
        problem_dict["angle_xc"],
        problem_dict["angle_wu"],
        problem_dict["angle_wc"],
        problem_dict["bc_x"],
        hybrid_dict["fine_idx"],
        hybrid_dict["coarse_idx"],
        hybrid_dict["factor"],
        steps=problem_dict["steps"],
        dt=problem_dict["dt"],
        geometry=1,
    )

    for step in range(problem_dict["steps"]):
        assert np.sum(np.fabs(mg_flux[step] - hy_flux[step])) < 5e-7, (
            str(step) + " not equivalent"
        )


@pytest.mark.slab
@pytest.mark.bdf2
@pytest.mark.multigroup
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize(
    ("groups_c", "angles_c", "varying"),
    [
        (87, 4, True),
        (43, 4, True),
        (43, 2, True),
        (87, 4, False),
        (43, 4, False),
        (43, 2, False),
    ],
)
def test_v_bdf2_01(groups_c, angles_c, varying):
    groups_u = 87
    angles_u = 4

    problem_dict = _example_problem_01(groups_u, groups_c, angles_u, angles_c)
    hybrid_dict = _get_hybrid_params(groups_u, groups_c, problem_dict)

    hy_flux = hybrid1d.auto_bdf2(
        problem_dict["flux_last"],
        problem_dict["xs_total_u"].copy(),
        problem_dict["xs_scatter_u"],
        problem_dict["xs_fission_u"],
        problem_dict["velocity_u"],
        problem_dict["external"],
        problem_dict["boundary"],
        problem_dict["medium_map"],
        problem_dict["delta_x"],
        problem_dict["angle_xu"],
        problem_dict["angle_wu"],
        problem_dict["bc_x"],
        angles_c,
        groups_c,
        hybrid_dict["energy_grid"],
        steps=problem_dict["steps"],
        dt=problem_dict["dt"],
        geometry=1,
    )

    if varying:
        groups_c = np.array([groups_c] * problem_dict["steps"])
        angles_c = np.array([angles_c] * problem_dict["steps"])

    vhy_flux = vhybrid1d.bdf2(
        groups_c,
        angles_c,
        problem_dict["flux_last"],
        problem_dict["xs_total_u"],
        problem_dict["xs_scatter_u"],
        problem_dict["xs_fission_u"],
        problem_dict["velocity_u"],
        problem_dict["external"],
        problem_dict["boundary"],
        problem_dict["medium_map"],
        problem_dict["delta_x"],
        problem_dict["angle_xu"],
        problem_dict["angle_wu"],
        problem_dict["bc_x"],
        steps=problem_dict["steps"],
        dt=problem_dict["dt"],
        geometry=1,
        energy_grid=87,
    )

    for step in range(problem_dict["steps"]):
        assert np.sum(np.fabs(vhy_flux[step] - hy_flux[step])) < 2e-7, (
            str(step) + " not equivalent"
        )
