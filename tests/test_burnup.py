"""Tests for the transport-coupled burnup driver.

Marked ``burnup`` (and ``depletion``); these run a full transport solve per
step on a small slab with the synthetic nuclide library.
"""

import numpy as np
import pytest

import discrete1
from discrete1 import burnup1d, critical1d
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


def test_burnup_with_energy_dependent_chi():
    # (G, G) chi[g_in, g_out]: exercises the _solve_transport tiling branch
    # for a 2D library.chi (tiled to (regions, G, G)) all the way through
    # critical1d.power_iteration's chi.ndim == 3 dispatch.
    lib = synthetic_library(groups=2)
    lib.chi = np.array([[0.9, 0.1], [0.2, 0.8]])
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
        dt_steps=[dt],
        power=1.0e6,
        order=16,
    )

    assert history.shape == (2, 1, 4)
    assert keff.shape == (2,)
    fuel = history[:, 0, lib.index["fuel"]]
    assert fuel[1] < fuel[0]
    assert keff[1] < keff[0]


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


def test_ml_burnup_no_models_agrees_with_burnup():
    # Both model lists are None, so every solve falls back to plain
    # power_iteration -- ml_burnup must reproduce burnup's result.
    lib = synthetic_library(groups=1)
    medium_map, delta_x, angle_x, angle_w = _slab_setup(cells=20)
    densities0 = np.array([[0.04, 0.0, 0.0, 0.0]])

    hist_plain, keff_plain = burnup1d.burnup(
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
    )
    hist_ml, keff_ml = burnup1d.ml_burnup(
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
    )
    assert np.allclose(hist_plain, hist_ml)
    assert np.allclose(keff_plain, keff_ml)


def test_ml_burnup_uses_ml_power_iteration_when_models_given(monkeypatch):
    # ml_burnup must route each step through critical1d.ml_power_iteration
    # when models are supplied, never through plain power_iteration.
    lib = synthetic_library(groups=1)
    medium_map, delta_x, angle_x, angle_w = _slab_setup(cells=20)
    densities0 = np.array([[0.04, 0.0, 0.0, 0.0]])

    calls = {"ml": 0, "plain": 0}
    real_ml = critical1d.ml_power_iteration
    real_plain = critical1d.power_iteration

    def spy_ml(*args, **kwargs):
        calls["ml"] += 1
        return real_ml(*args, **kwargs)

    def spy_plain(*args, **kwargs):
        calls["plain"] += 1
        return real_plain(*args, **kwargs)

    monkeypatch.setattr(critical1d, "ml_power_iteration", spy_ml)
    monkeypatch.setattr(critical1d, "power_iteration", spy_plain)

    # [0] is the "no model, use physics" placeholder (tools.scatter_prod_predict),
    # so this exercises ml_power_iteration without a real trained model.
    burnup1d.ml_burnup(
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
        scatter_models=[0],
    )

    assert calls["ml"] > 0
    assert calls["plain"] == 0


def test_ml_burnup_fission_models_without_full_matrix_fails():
    # fission_models needs the full production matrix; without
    # full_fission_matrix=True this hits a shape mismatch.
    lib = synthetic_library(groups=2)
    medium_map, delta_x, angle_x, angle_w = _slab_setup(cells=10)
    densities0 = np.array([[0.04, 0.0, 0.0, 0.0]])

    with pytest.raises(Exception):
        burnup1d.ml_burnup(
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
            fission_models=[0],
        )


def test_ml_burnup_fission_models_with_full_matrix_matches_burnup():
    # [0] (physics placeholder) with full_fission_matrix=True must
    # reproduce the ordinary chi/nu_fission result exactly.
    lib = synthetic_library(groups=2)
    medium_map, delta_x, angle_x, angle_w = _slab_setup(cells=10)
    densities0 = np.array([[0.04, 0.0, 0.0, 0.0]])

    hist_plain, keff_plain = burnup1d.burnup(
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
    )
    hist_fm, keff_fm = burnup1d.ml_burnup(
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
        fission_models=[0],
        full_fission_matrix=True,
    )
    assert np.allclose(hist_plain, hist_fm)
    assert np.allclose(keff_plain, keff_fm)


def test_ml_burnup_flux_old_converges_like_default():
    # A supplied flux_old changes the starting point, not the physics --
    # must converge to the same result as the random-start default.
    lib = synthetic_library(groups=1)
    medium_map, delta_x, angle_x, angle_w = _slab_setup(cells=20)
    densities0 = np.array([[0.04, 0.0, 0.0, 0.0]])

    hist_default, keff_default = burnup1d.ml_burnup(
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
    )
    seed = np.full((medium_map.shape[0], 1), 2.0)
    hist_seeded, keff_seeded = burnup1d.ml_burnup(
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
        flux_old=seed,
    )
    assert np.allclose(hist_default, hist_seeded, atol=1e-6)
    assert np.allclose(keff_default, keff_seeded, atol=1e-3)


def test_ml_burnup_warm_starts_across_solves(monkeypatch):
    # The corrector must warm-start from the predictor's flux, and the next
    # step's predictor from the corrector's -- never the seed or a repeat.
    lib = synthetic_library(groups=1)
    medium_map, delta_x, angle_x, angle_w = _slab_setup(cells=20)
    densities0 = np.array([[0.04, 0.0, 0.0, 0.0]])

    seen_flux_old = []
    real_ml = critical1d.ml_power_iteration

    def spy(flux_old, *args, **kwargs):
        seen_flux_old.append(flux_old)
        return real_ml(flux_old, *args, **kwargs)

    monkeypatch.setattr(critical1d, "ml_power_iteration", spy)

    user_seed = np.full((medium_map.shape[0], 1), 2.0)
    burnup1d.ml_burnup(
        lib,
        densities0,
        medium_map,
        delta_x,
        angle_x,
        angle_w,
        bc_x=[0, 0],
        dt_steps=[2.6e6, 2.6e6],
        power=1.0e6,
        order=16,
        scatter_models=[0],
        flux_old=user_seed,
    )

    # predictor(step0), corrector(step0), predictor(step1), corrector(step1),
    # end-of-life
    assert len(seen_flux_old) == 5
    assert np.array_equal(seen_flux_old[0], user_seed)
    # Each later solve warm-starts from the immediately preceding one, not
    # the original seed and not a repeat of the previous array.
    for prev, nxt in zip(seen_flux_old[:-1], seen_flux_old[1:]):
        assert not np.array_equal(nxt, user_seed)
        assert not np.array_equal(nxt, prev)
    # The user's original array must never be mutated.
    assert np.array_equal(user_seed, np.full((medium_map.shape[0], 1), 2.0))


def test_power_iteration_flux_old_not_mutated():
    xs_total = np.array([[0.3360, 0.2208]])
    xs_scatter = np.array([np.array([[0.23616, 0.0], [0.0432, 0.0792]]).T])
    chi = np.array([[0.425, 0.575]])
    nu = np.array([[2.93, 3.10]])
    sigmaf = np.array([[0.08544, 0.0936]])
    nusigf = nu * sigmaf
    cells_x = 200
    delta_x = np.repeat(1.795602 * 2 / cells_x, cells_x)
    medium_map = np.zeros(cells_x, dtype=np.int32)
    angle_x, angle_w = discrete1.angular_x(20, [0, 0])

    _, keff_default = critical1d.power_iteration(
        xs_total,
        xs_scatter,
        nusigf,
        medium_map,
        delta_x,
        angle_x,
        angle_w,
        [0, 0],
        chi=chi,
    )

    seed = np.full((cells_x, 2), 3.0)
    seed_before = seed.copy()
    _, keff_seeded = critical1d.power_iteration(
        xs_total,
        xs_scatter,
        nusigf,
        medium_map,
        delta_x,
        angle_x,
        angle_w,
        [0, 0],
        chi=chi,
        flux_old=seed,
    )
    assert abs(keff_default - keff_seeded) < 2e-3
    assert np.array_equal(seed, seed_before)
