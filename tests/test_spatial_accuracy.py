import pytest
import numpy as np

import discrete1
from discrete1.fixed1d import source_iteration, dynamic_mode_decomp
from discrete1.utils import manufactured as mms
from tests import problems


@pytest.mark.smoke
@pytest.mark.slab
@pytest.mark.si
@pytest.mark.parametrize(
    ("angular", "edges"), [(True, 0), (True, 1), (False, 0), (False, 1)]
)
def test_mms_ss_si_01(angular, edges):
    # Initialize error list
    errors = []
    # Initialize numbers of spatial cells
    cells_list = np.array([50, 100, 200])

    for cells_x in cells_list:
        data = problems.manufactured_ss_01(cells_x, 2)
        (
            xs_total,
            xs_scatter,
            xs_fission,
            external,
            boundary,
            medium_map,
            delta_x,
            angle_x,
            angle_w,
            edges_x,
            bc_x,
        ) = data

        flux = source_iteration(
            xs_total,
            xs_scatter,
            xs_fission,
            external,
            boundary,
            medium_map,
            delta_x,
            angle_x,
            angle_w,
            bc_x,
            geometry=1,
            angular=angular,
            edges=edges,
        )

        centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
        space_x = edges_x.copy() if edges else centers_x.copy()

        # Calculate analytical
        exact = mms.solution_ss_01(space_x, angle_x)[:, :, None]
        if not angular:
            exact = np.sum(exact * angle_w[None, :, None], axis=1)

        # Append error list
        errors.append(mms.spatial_error(flux, exact))

    atol = 5e-2 if edges else 5e-3
    for ii in range(len(errors) - 1):
        dx = cells_list[ii + 1] / cells_list[ii]
        assert abs(mms.order_accuracy(errors[ii], errors[ii + 1], dx) - 2) < atol


@pytest.mark.slab
@pytest.mark.si
@pytest.mark.parametrize(
    ("angular", "edges"), [(True, 0), (True, 1), (False, 0), (False, 1)]
)
def test_mms_ss_si_02(angular, edges):
    # Initialize error list
    errors = []
    # Initialize number of spatial cells
    cells_list = np.array([50, 100, 200])

    for cells_x in cells_list:
        data = problems.manufactured_ss_02(cells_x, 2)
        (
            xs_total,
            xs_scatter,
            xs_fission,
            external,
            boundary,
            medium_map,
            delta_x,
            angle_x,
            angle_w,
            edges_x,
            bc_x,
        ) = data

        flux = source_iteration(
            xs_total,
            xs_scatter,
            xs_fission,
            external,
            boundary,
            medium_map,
            delta_x,
            angle_x,
            angle_w,
            bc_x,
            geometry=1,
            angular=angular,
            edges=edges,
        )

        centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
        space_x = edges_x.copy() if edges else centers_x.copy()

        # Calculate analytical
        exact = mms.solution_ss_02(space_x, angle_x)[:, :, None]
        if not angular:
            exact = np.sum(exact * angle_w[None, :, None], axis=1)

        # Append error list
        errors.append(mms.spatial_error(flux, exact))

    atol = 5e-2 if edges else 5e-3
    for ii in range(len(errors) - 1):
        dx = cells_list[ii + 1] / cells_list[ii]
        assert abs(mms.order_accuracy(errors[ii], errors[ii + 1], dx) - 2) < atol


@pytest.mark.slab
@pytest.mark.si
@pytest.mark.parametrize(
    ("angular", "edges"), [(True, 0), (True, 1), (False, 0), (False, 1)]
)
def test_mms_ss_si_03(angular, edges):
    # Initialize error list
    errors = []
    # Initialize numbers of spatial cells
    cells_list = np.array([50, 100, 200])

    for cells_x in cells_list:
        data = problems.manufactured_ss_03(cells_x, 8)
        (
            xs_total,
            xs_scatter,
            xs_fission,
            external,
            boundary,
            medium_map,
            delta_x,
            angle_x,
            angle_w,
            edges_x,
            bc_x,
        ) = data

        flux = source_iteration(
            xs_total,
            xs_scatter,
            xs_fission,
            external,
            boundary,
            medium_map,
            delta_x,
            angle_x,
            angle_w,
            bc_x,
            geometry=1,
            angular=angular,
            edges=edges,
        )

        centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
        space_x = edges_x.copy() if edges else centers_x.copy()

        # Calculate analytical
        exact = mms.solution_ss_03(space_x, angle_x)[:, :, None]
        if not angular:
            exact = np.sum(exact * angle_w[None, :, None], axis=1)

        # Append error list
        errors.append(mms.spatial_error(flux, exact))
    atol = 5e-2 if edges else 2e-2
    for ii in range(len(errors) - 1):
        dx = cells_list[ii + 1] / cells_list[ii]
        assert abs(mms.order_accuracy(errors[ii], errors[ii + 1], dx) - 2) < atol


@pytest.mark.slab
@pytest.mark.si
@pytest.mark.parametrize(
    ("angular", "edges"), [(True, 0), (True, 1), (False, 0), (False, 1)]
)
def test_mms_ss_si_04(angular, edges):
    # Initialize error list
    errors = []
    # Initialize numbers of spatial cells
    cells_list = np.array([100, 200, 400])

    for cells_x in cells_list:
        data = problems.manufactured_ss_04(cells_x, 2)
        (
            xs_total,
            xs_scatter,
            xs_fission,
            external,
            boundary,
            medium_map,
            delta_x,
            angle_x,
            angle_w,
            edges_x,
            bc_x,
        ) = data

        flux = source_iteration(
            xs_total,
            xs_scatter,
            xs_fission,
            external,
            boundary,
            medium_map,
            delta_x,
            angle_x,
            angle_w,
            bc_x,
            geometry=1,
            angular=angular,
            edges=edges,
        )

        centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
        space_x = edges_x.copy() if edges else centers_x.copy()

        # Calculate analytical
        exact = mms.solution_ss_04(space_x, angle_x)[:, :, None]
        if not angular:
            exact = np.sum(exact * angle_w[None, :, None], axis=1)

        # Append error list
        errors.append(mms.spatial_error(flux, exact))

    atol = 5e-2 if edges else 5e-3
    for ii in range(len(errors) - 1):
        dx = cells_list[ii + 1] / cells_list[ii]
        assert abs(mms.order_accuracy(errors[ii], errors[ii + 1], dx) - 2) < atol


@pytest.mark.slab
@pytest.mark.si
@pytest.mark.parametrize(
    ("angular", "edges"), [(True, 0), (True, 1), (False, 0), (False, 1)]
)
def test_mms_ss_si_05(angular, edges):
    # Initialize error list
    errors = []
    # Initialize numbers of spatial cells
    cells_list = np.array([50, 100, 200])

    for cells_x in cells_list:
        data = problems.manufactured_ss_05(cells_x, 8)
        (
            xs_total,
            xs_scatter,
            xs_fission,
            external,
            boundary,
            medium_map,
            delta_x,
            angle_x,
            angle_w,
            edges_x,
            bc_x,
        ) = data

        flux = source_iteration(
            xs_total,
            xs_scatter,
            xs_fission,
            external,
            boundary,
            medium_map,
            delta_x,
            angle_x,
            angle_w,
            bc_x,
            geometry=1,
            angular=angular,
            edges=edges,
        )

        centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
        space_x = edges_x.copy() if edges else centers_x.copy()

        # Calculate analytical
        exact = mms.solution_ss_05(space_x, angle_x)[:, :, None]
        if not angular:
            exact = np.sum(exact * angle_w[None, :, None], axis=1)

        # Append error list
        errors.append(mms.spatial_error(flux, exact))
    atol = 5e-2 if edges else 5e-3
    for ii in range(len(errors) - 1):
        dx = cells_list[ii + 1] / cells_list[ii]
        assert abs(mms.order_accuracy(errors[ii], errors[ii + 1], dx) - 2) < atol


@pytest.mark.smoke
@pytest.mark.slab
@pytest.mark.dmd
@pytest.mark.parametrize(
    ("angular", "edges"), [(True, 0), (True, 1), (False, 0), (False, 1)]
)
def test_mms_ss_dmd_01(angular, edges):
    # Initialize error list
    errors = []
    # Initialize numbers of spatial cells
    cells_list = np.array([50, 100, 200])

    for cells_x in cells_list:
        data = problems.manufactured_ss_01(cells_x, 2)
        (
            xs_total,
            xs_scatter,
            xs_fission,
            external,
            boundary,
            medium_map,
            delta_x,
            angle_x,
            angle_w,
            edges_x,
            bc_x,
        ) = data

        flux = dynamic_mode_decomp(
            xs_total,
            xs_scatter,
            xs_fission,
            external,
            boundary,
            medium_map,
            delta_x,
            angle_x,
            angle_w,
            bc_x,
            geometry=1,
            angular=angular,
            edges=edges,
            R=2,
            K=5,
        )

        centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
        space_x = edges_x.copy() if edges else centers_x.copy()

        # Calculate analytical
        exact = mms.solution_ss_01(space_x, angle_x)[:, :, None]
        if not angular:
            exact = np.sum(exact * angle_w[None, :, None], axis=1)

        # Append error list
        errors.append(mms.spatial_error(flux, exact))

    atol = 5e-2 if edges else 5e-3
    for ii in range(len(errors) - 1):
        dx = cells_list[ii + 1] / cells_list[ii]
        assert abs(mms.order_accuracy(errors[ii], errors[ii + 1], dx) - 2) < atol


@pytest.mark.slab
@pytest.mark.dmd
@pytest.mark.parametrize(
    ("angular", "edges"), [(True, 0), (True, 1), (False, 0), (False, 1)]
)
def test_mms_ss_dmd_02(angular, edges):
    # Initialize error list
    errors = []
    # Initialize number of spatial cells
    cells_list = np.array([50, 100, 200])

    for cells_x in cells_list:
        data = problems.manufactured_ss_02(cells_x, 2)
        (
            xs_total,
            xs_scatter,
            xs_fission,
            external,
            boundary,
            medium_map,
            delta_x,
            angle_x,
            angle_w,
            edges_x,
            bc_x,
        ) = data

        flux = dynamic_mode_decomp(
            xs_total,
            xs_scatter,
            xs_fission,
            external,
            boundary,
            medium_map,
            delta_x,
            angle_x,
            angle_w,
            bc_x,
            geometry=1,
            angular=angular,
            edges=edges,
            R=2,
            K=5,
        )

        centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
        space_x = edges_x.copy() if edges else centers_x.copy()

        # Calculate analytical
        exact = mms.solution_ss_02(space_x, angle_x)[:, :, None]
        if not angular:
            exact = np.sum(exact * angle_w[None, :, None], axis=1)

        # Append error list
        errors.append(mms.spatial_error(flux, exact))

    atol = 5e-2 if edges else 5e-3
    for ii in range(len(errors) - 1):
        dx = cells_list[ii + 1] / cells_list[ii]
        assert abs(mms.order_accuracy(errors[ii], errors[ii + 1], dx) - 2) < atol


@pytest.mark.slab
@pytest.mark.dmd
@pytest.mark.parametrize(
    ("angular", "edges"), [(True, 0), (True, 1), (False, 0), (False, 1)]
)
def test_mms_ss_dmd_03(angular, edges):
    # Initialize error list
    errors = []
    # Initialize numbers of spatial cells
    cells_list = np.array([50, 100, 200])

    for cells_x in cells_list:
        data = problems.manufactured_ss_03(cells_x, 8)
        (
            xs_total,
            xs_scatter,
            xs_fission,
            external,
            boundary,
            medium_map,
            delta_x,
            angle_x,
            angle_w,
            edges_x,
            bc_x,
        ) = data

        flux = dynamic_mode_decomp(
            xs_total,
            xs_scatter,
            xs_fission,
            external,
            boundary,
            medium_map,
            delta_x,
            angle_x,
            angle_w,
            bc_x,
            geometry=1,
            angular=angular,
            edges=edges,
            R=2,
            K=5,
        )

        centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
        space_x = edges_x.copy() if edges else centers_x.copy()

        # Calculate analytical
        exact = mms.solution_ss_03(space_x, angle_x)[:, :, None]
        if not angular:
            exact = np.sum(exact * angle_w[None, :, None], axis=1)

        # Append error list
        errors.append(mms.spatial_error(flux, exact))
    atol = 5e-2 if edges else 2e-2
    for ii in range(len(errors) - 1):
        dx = cells_list[ii + 1] / cells_list[ii]
        assert abs(mms.order_accuracy(errors[ii], errors[ii + 1], dx) - 2) < atol


@pytest.mark.slab
@pytest.mark.dmd
@pytest.mark.parametrize(
    ("angular", "edges"), [(True, 0), (True, 1), (False, 0), (False, 1)]
)
def test_mms_ss_dmd_04(angular, edges):
    # Initialize error list
    errors = []
    # Initialize numbers of spatial cells
    cells_list = np.array([100, 200, 400])

    for cells_x in cells_list:
        data = problems.manufactured_ss_04(cells_x, 2)
        (
            xs_total,
            xs_scatter,
            xs_fission,
            external,
            boundary,
            medium_map,
            delta_x,
            angle_x,
            angle_w,
            edges_x,
            bc_x,
        ) = data

        flux = dynamic_mode_decomp(
            xs_total,
            xs_scatter,
            xs_fission,
            external,
            boundary,
            medium_map,
            delta_x,
            angle_x,
            angle_w,
            bc_x,
            geometry=1,
            angular=angular,
            edges=edges,
            R=2,
            K=5,
        )

        centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
        space_x = edges_x.copy() if edges else centers_x.copy()

        # Calculate analytical
        exact = mms.solution_ss_04(space_x, angle_x)[:, :, None]
        if not angular:
            exact = np.sum(exact * angle_w[None, :, None], axis=1)

        # Append error list
        errors.append(mms.spatial_error(flux, exact))

    atol = 5e-2 if edges else 5e-3
    for ii in range(len(errors) - 1):
        dx = cells_list[ii + 1] / cells_list[ii]
        assert abs(mms.order_accuracy(errors[ii], errors[ii + 1], dx) - 2) < atol


@pytest.mark.slab
@pytest.mark.dmd
@pytest.mark.parametrize(
    ("angular", "edges"), [(True, 0), (True, 1), (False, 0), (False, 1)]
)
def test_mms_ss_dmd_05(angular, edges):
    # Initialize error list
    errors = []
    # Initialize numbers of spatial cells
    cells_list = np.array([50, 100, 200])

    for cells_x in cells_list:
        data = problems.manufactured_ss_05(cells_x, 8)
        (
            xs_total,
            xs_scatter,
            xs_fission,
            external,
            boundary,
            medium_map,
            delta_x,
            angle_x,
            angle_w,
            edges_x,
            bc_x,
        ) = data

        flux = dynamic_mode_decomp(
            xs_total,
            xs_scatter,
            xs_fission,
            external,
            boundary,
            medium_map,
            delta_x,
            angle_x,
            angle_w,
            bc_x,
            geometry=1,
            angular=angular,
            edges=edges,
            R=2,
            K=5,
        )

        centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])
        space_x = edges_x.copy() if edges else centers_x.copy()

        # Calculate analytical
        exact = mms.solution_ss_05(space_x, angle_x)[:, :, None]
        if not angular:
            exact = np.sum(exact * angle_w[None, :, None], axis=1)

        # Append error list
        errors.append(mms.spatial_error(flux, exact))
    atol = 5e-2 if edges else 5e-3
    for ii in range(len(errors) - 1):
        dx = cells_list[ii + 1] / cells_list[ii]
        assert abs(mms.order_accuracy(errors[ii], errors[ii + 1], dx) - 2) < atol
