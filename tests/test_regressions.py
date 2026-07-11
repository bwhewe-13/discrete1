"""Regression tests for repo-review bug fixes.

Covers: packaged-data path joins (materials / external sources), the sphere
known-source driver (time-dependent and angular/edge outputs in spherical
geometry), anisotropic angular-flux reconstruction in fixed1d, nonuniform
sphere meshes, and the zero-safe convergence metric.
"""

import numpy as np
import pytest

import discrete1
from discrete1 import external1d, fixed1d, timed1d, tools
from discrete1.critical1d import power_iteration

# The packaged cross-section data (discrete1/sources) is gitignored and not
# shipped to CI, so tests that load it only run where the data exists.
requires_packaged_data = pytest.mark.skipif(
    not (external1d.DATA_PATH / "materials" / "hydrogen.npz").is_file(),
    reason="packaged cross-section data (discrete1/sources) not available",
)

# One-group plutonium data from the sphere criticality benchmark (Sood 2003).
PU_XS_TOTAL = np.array([[0.32640]])
PU_XS_SCATTER = np.array([[[0.225216]]])
PU_XS_FISSION = np.array([[[2.84 * 0.0816]]])
PU_RADIUS = 6.082547


def _sphere_setup(delta_x, angles=8):
    cells = delta_x.shape[0]
    medium_map = np.zeros(cells, dtype=np.int32)
    bc_x = [1, 0]
    angle_x, angle_w = discrete1.angular_x(angles, bc_x)
    return medium_map, bc_x, angle_x, angle_w


@requires_packaged_data
def test_materials_factory_loads_packaged_data():
    # DATA_PATH is an importlib.resources Traversable; joining it with "+"
    # raised TypeError and broke every example script.
    xs_total, xs_scatter, xs_fission = discrete1.materials(
        87, ["hydrogen", "uranium-%20%"]
    )
    assert xs_total.shape == (2, 87)
    assert xs_scatter.shape == (2, 87, 87)
    assert xs_fission.shape == (2, 87, 87)
    assert np.all(xs_total > 0.0)


@requires_packaged_data
def test_external_ambe_loads_packaged_data():
    x = np.linspace(0.0, 10.0, 21)
    edges_g = np.linspace(0.1, 15.0, 88)
    source = external1d.ambe(x, 10, edges_g)
    assert source.shape == (21, 1, 87)
    assert np.isfinite(source).all()


@pytest.mark.sphere
@pytest.mark.si
def test_fixed1d_sphere_angular_matches_scalar():
    # sphere_known_source_sn used to rebind its output buffer to a 1-D zeros
    # array and crash with IndexError on every call.
    cells = 50
    delta_x = np.repeat(PU_RADIUS / cells, cells)
    medium_map, bc_x, angle_x, angle_w = _sphere_setup(delta_x)
    external = 0.5 * np.ones((cells, 1, 1))
    boundary = np.zeros((2, 1, 1))

    scalar = fixed1d.source_iteration(
        PU_XS_TOTAL,
        PU_XS_SCATTER,
        0.0 * PU_XS_FISSION,
        external,
        boundary,
        medium_map,
        delta_x,
        angle_x,
        angle_w,
        bc_x,
        geometry=2,
    )
    angular = fixed1d.source_iteration(
        PU_XS_TOTAL,
        PU_XS_SCATTER,
        0.0 * PU_XS_FISSION,
        external,
        boundary,
        medium_map,
        delta_x,
        angle_x,
        angle_w,
        bc_x,
        geometry=2,
        angular=True,
    )
    assert angular.shape == (cells, angle_x.shape[0], 1)
    assert np.isfinite(angular).all()
    collapsed = np.einsum("ijk,j->ik", angular, angle_w)
    assert np.allclose(collapsed, scalar, rtol=1e-4, atol=1e-8)


@pytest.mark.sphere
@pytest.mark.si
def test_fixed1d_sphere_scalar_edges():
    # Also exercises the backward sphere kernel's edge deposit, which used to
    # write the outgoing (inner-face) value at the outer-face index.
    cells = 50
    delta_x = np.repeat(PU_RADIUS / cells, cells)
    medium_map, bc_x, angle_x, angle_w = _sphere_setup(delta_x)
    external = 0.5 * np.ones((cells, 1, 1))
    boundary = np.zeros((2, 1, 1))

    centers = fixed1d.source_iteration(
        PU_XS_TOTAL,
        PU_XS_SCATTER,
        0.0 * PU_XS_FISSION,
        external,
        boundary,
        medium_map,
        delta_x,
        angle_x,
        angle_w,
        bc_x,
        geometry=2,
    )
    edges = fixed1d.source_iteration(
        PU_XS_TOTAL,
        PU_XS_SCATTER,
        0.0 * PU_XS_FISSION,
        external,
        boundary,
        medium_map,
        delta_x,
        angle_x,
        angle_w,
        bc_x,
        geometry=2,
        edges=1,
    )
    assert edges.shape == (cells + 1, 1)
    assert np.isfinite(edges).all()
    assert np.all(edges >= 0.0)
    # Interior edge values interleave the neighboring center values.
    mid = 0.5 * (edges[1:] + edges[:-1])
    assert np.allclose(mid, centers, rtol=0.05, atol=1e-6)


@pytest.mark.sphere
@pytest.mark.bdf1
def test_timed1d_sphere_backward_euler_runs():
    # Time-dependent sphere problems crashed inside sphere_known_source_sn.
    cells = 40
    groups, angles = 1, 4
    delta_x = np.repeat(4.0 / cells, cells)
    medium_map, bc_x, angle_x, angle_w = _sphere_setup(delta_x, angles=angles)
    steps, dt = 3, 1.0

    flux = timed1d.backward_euler(
        np.zeros((cells, angles, groups)),
        PU_XS_TOTAL,
        PU_XS_SCATTER,
        0.0 * PU_XS_FISSION,
        np.ones((groups,)),
        np.ones((1, cells, 1, 1)),
        np.zeros((1, 2, 1, 1)),
        medium_map,
        delta_x,
        angle_x,
        angle_w,
        bc_x,
        steps,
        dt,
        geometry=2,
    )
    assert flux.shape == (steps, cells, groups)
    assert np.isfinite(flux).all()
    assert np.all(flux >= 0.0)
    # Flux builds up toward steady state under a constant source.
    assert flux[-1].sum() > flux[0].sum()


@pytest.mark.slab
@pytest.mark.anisotropic
def test_fixed1d_aniso_angular_matches_isotropic():
    # known_source_calculation used to feed the 4-D anisotropic matrix into
    # the strictly 3-D _source_total kernel (numba TypingError). With all
    # L >= 1 moments zero the anisotropic path must match the isotropic one.
    cells, angles = 40, 8
    delta_x = np.repeat(2.0 / cells, cells)
    medium_map = np.zeros(cells, dtype=np.int32)
    bc_x = [0, 0]
    angle_x, angle_w = discrete1.angular_x(angles, bc_x)
    external = np.ones((cells, 1, 1))
    boundary = np.zeros((2, 1, 1))

    xs_scatter_iso = np.array([[[0.3]]])
    xs_scatter_aniso = np.zeros((1, 1, 1, 2))
    xs_scatter_aniso[..., 0] = 0.3

    kwargs = dict(geometry=1, angular=True)
    iso = fixed1d.source_iteration(
        PU_XS_TOTAL,
        xs_scatter_iso,
        np.zeros((1, 1, 1)),
        external,
        boundary,
        medium_map,
        delta_x,
        angle_x,
        angle_w,
        bc_x,
        **kwargs,
    )
    aniso = fixed1d.source_iteration(
        PU_XS_TOTAL,
        xs_scatter_aniso,
        np.zeros((1, 1, 1)),
        external,
        boundary,
        medium_map,
        delta_x,
        angle_x,
        angle_w,
        bc_x,
        **kwargs,
    )
    assert np.allclose(aniso, iso, rtol=1e-5, atol=1e-10)


@pytest.mark.sphere
@pytest.mark.power_iteration
def test_sphere_nonuniform_mesh_keff():
    # Sphere kernels used r_i = i * delta_x[i], silently assuming a uniform
    # mesh; a nonuniform mesh gave wrong shell areas/volumes and keff.
    uniform = np.repeat(PU_RADIUS / 150, 150)
    # Same sphere, graded mesh: fine near the center, coarser outside.
    weights = np.linspace(1.0, 2.0, 150)
    nonuniform = weights / weights.sum() * PU_RADIUS
    assert nonuniform.sum() == pytest.approx(uniform.sum())

    keffs = []
    for delta_x in (uniform, nonuniform):
        medium_map, bc_x, angle_x, angle_w = _sphere_setup(delta_x, angles=16)
        _, keff = power_iteration(
            PU_XS_TOTAL,
            PU_XS_SCATTER,
            PU_XS_FISSION,
            medium_map,
            delta_x,
            angle_x,
            angle_w,
            bc_x,
            geometry=2,
        )
        keffs.append(keff)

    assert keffs[0] == pytest.approx(1.0, abs=2e-3)  # benchmark critical radius
    assert keffs[1] == pytest.approx(keffs[0], abs=2e-3)


def test_flux_change_zero_safe():
    # The old metric divided elementwise by the flux: any zero-flux cell made
    # the change NaN and every solver loop ran to its iteration cap.
    flux = np.array([0.0, 1.0, 2.0])
    flux_old = np.array([0.0, 1.0, 1.0])
    change = tools.flux_change(flux, flux_old)
    assert np.isfinite(change)
    assert change == pytest.approx(1.0 / np.sqrt(5.0))
    assert tools.flux_change(np.zeros(3), np.zeros(3)) == 0.0
