"""Tests for the transport-coupled burnup driver.

Marked ``burnup`` (and ``depletion``); these run a full transport solve per
step on a small slab with the synthetic nuclide library.
"""

import numpy as np
import pytest

import discrete1
from discrete1 import burnup1d
from discrete1.nuclides import synthetic_library

pytestmark = [pytest.mark.burnup, pytest.mark.depletion]


def _slab_setup(cells=40, length=20.0, angles=8):
    edges_x = np.linspace(0.0, length, cells + 1)
    delta_x = np.diff(edges_x)
    medium_map = np.zeros(cells, dtype=np.int32)  # single region
    angle_x, angle_w = discrete1.angular_x(angles, bc_x=[0, 0])
    return medium_map, delta_x, angle_x, angle_w


def test_macroscopic_xs_scales_with_density():
    lib = synthetic_library(groups=1)
    densities = np.array([[0.04, 0.0, 0.0, 0.0]])
    xs_total, xs_scatter, nu_fission = burnup1d.macroscopic_xs(lib, densities)
    assert xs_total[0, 0] == pytest.approx(0.04 * lib.xs_total[0, 0])
    assert nu_fission[0, 0] == pytest.approx(0.04 * lib.nu_fission[0, 0])
    assert xs_scatter.shape == (1, 1, 1)


def test_region_flux_volume_average():
    flux = np.array([[1.0], [3.0], [5.0]])
    medium_map = np.array([0, 0, 1])
    delta_x = np.array([1.0, 1.0, 2.0])
    rflux = burnup1d.region_flux(flux, medium_map, delta_x, n_regions=2)
    assert rflux[0, 0] == pytest.approx(2.0)  # mean of 1 and 3
    assert rflux[1, 0] == pytest.approx(5.0)


def test_cell_volumes_geometry():
    delta_x = np.array([1.0, 1.0, 2.0])
    # Slab: cell widths are the volume weights
    assert np.allclose(burnup1d._cell_volumes(delta_x, geometry=1), delta_x)
    # Sphere: shell volumes from the accumulated radii
    edges = np.array([0.0, 1.0, 2.0, 4.0])
    expected = 4.0 / 3.0 * np.pi * np.diff(edges**3)
    assert np.allclose(burnup1d._cell_volumes(delta_x, geometry=2), expected)


def test_region_flux_sphere_weighting():
    # In a sphere the outer shell dominates the region average.
    flux = np.array([[1.0], [3.0]])
    medium_map = np.array([0, 0])
    volumes = burnup1d._cell_volumes(np.array([1.0, 1.0]), geometry=2)
    rflux = burnup1d.region_flux(flux, medium_map, volumes, n_regions=1)
    expected = (volumes[0] * 1.0 + volumes[1] * 3.0) / volumes.sum()
    assert rflux[0, 0] == pytest.approx(expected)
    assert rflux[0, 0] > 2.0  # the unweighted slab average


def test_burnup_depletes_fuel_and_lowers_keff():
    lib = synthetic_library(groups=1)
    medium_map, delta_x, angle_x, angle_w = _slab_setup()
    densities0 = np.array([[0.04, 0.0, 0.0, 0.0]])

    dt = 2.6e6  # ~30 days in seconds
    history, keff = burnup1d.burnup(
        lib,
        densities0,
        medium_map,
        delta_x,
        angle_x,
        angle_w,
        bc_x=[0, 0],
        dt_steps=[dt, dt, dt],
        power=1.0e6,
        order=48,
    )

    assert history.shape == (4, 1, 4)
    # keff_history[s] corresponds to density_history[s], including end of life
    assert keff.shape == (4,)
    # Fuel monotonically depletes
    fuel = history[:, 0, lib.index["fuel"]]
    assert np.all(np.diff(fuel) < 0.0)
    # Fission products accumulate
    assert history[-1, 0, lib.index["fp_stable"]] > 0.0
    # k-effective drops as fuel burns (final entry is end of life)
    assert keff[-1] < keff[0]
    assert np.all(np.diff(keff) < 0.0)
    # Depleted densities are clamped non-negative
    assert np.all(history >= 0.0)


def test_predictor_only_runs():
    lib = synthetic_library(groups=1)
    medium_map, delta_x, angle_x, angle_w = _slab_setup(cells=20)
    densities0 = np.array([[0.04, 0.0, 0.0, 0.0]])
    history, keff = burnup1d.burnup(
        lib,
        densities0,
        medium_map,
        delta_x,
        angle_x,
        angle_w,
        bc_x=[0, 0],
        dt_steps=[2.6e6],
        power=1.0e6,
        order=16,
        predictor_corrector=False,
    )
    assert history.shape == (2, 1, 4)
    assert keff.shape == (2,)
    assert history[1, 0, lib.index["fuel"]] < densities0[0, 0]


def test_burnup_sphere_geometry():
    # Sphere driver smoke test: reflective center, vacuum surface.
    lib = synthetic_library(groups=1)
    cells = 20
    delta_x = np.full(cells, 0.5)
    medium_map = np.zeros(cells, dtype=np.int32)
    bc_x = [1, 0]
    angle_x, angle_w = discrete1.angular_x(8, bc_x=bc_x)
    history, keff = burnup1d.burnup(
        lib,
        np.array([[0.04, 0.0, 0.0, 0.0]]),
        medium_map,
        delta_x,
        angle_x,
        angle_w,
        bc_x=bc_x,
        dt_steps=[2.6e6],
        power=1.0e6,
        geometry=2,
        order=16,
    )
    assert history.shape == (2, 1, 4)
    assert keff.shape == (2,)
    assert history[1, 0, lib.index["fuel"]] < 0.04
    assert keff[1] < keff[0]
