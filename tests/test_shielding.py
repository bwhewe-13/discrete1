"""Tests for composition-dependent cross sections in the burnup driver.

``burnup(..., xs_update=fn)`` is method-agnostic; :mod:`discrete1.shielding`
is one update, Bondarenko tables. The table here is written by hand on the
synthetic library, with ``fuel2`` (bred from ``fuel``) as the shielded
nuclide, the way Pu-239 is bred from U-238.
"""

import math

import numpy as np
import pytest

import discrete1
from discrete1 import burnup1d
from discrete1.nuclides import synthetic_library
from discrete1.shielding import bondarenko_update

pytestmark = [pytest.mark.burnup, pytest.mark.depletion]

GRID = np.array([10.0, 100.0])
F_CAPTURE = np.array([[0.5, 0.8], [0.9, 0.95]])  # (G, S) for fuel2
F_TOTAL = np.array([[0.7, 0.9], [0.95, 0.99]])
INF_CAPTURE = np.array([4.0, 2.0])
SIGMA_P = np.array([10.0, 10.0, 0.0, 5.0])  # fuel, fuel2, fp, fp_stable
DELTA = np.array([[-0.2, -0.1], [-0.05, -0.02]])  # (G_out, S), incident group 1


def _table(library, escape=0.0):
    k = library.index["fuel2"]
    g = library.groups
    return {
        "names": np.array(library.names),
        "shield_sigma0": GRID,
        "shield_sigma0_infinite": 1.0e10,
        "shield_index": np.array([k]),
        "shield_inf_total": library.xs_total[[k]],
        "shield_inf_fission": library.fission_xs[[k]],
        "shield_inf_nu_fission": library.nu_fission[[k]],
        "shield_inf_capture": INF_CAPTURE[None, :],
        "shield_inf_scatter": library.xs_scatter[[k]],
        "shield_f_total": F_TOTAL[None],
        "shield_f_fission": np.ones((1, g, GRID.size)),
        "shield_f_capture": F_CAPTURE[None],
        "shield_scatter_k": np.array([0]),
        "shield_scatter_g": np.array([1]),
        "shield_scatter_delta": DELTA[None],
        "shield_sigma_p": SIGMA_P,
        "shield_escape": escape,
    }


@pytest.fixture
def library():
    return synthetic_library(groups=2)


def _densities(library, fuel, fuel2):
    n = np.zeros(library.n_nuclides)
    n[library.index["fuel"]] = fuel
    n[library.index["fuel2"]] = fuel2
    return n


def _capture(library, update, densities, region=0):
    k = library.index["fuel2"]
    return update(library, densities, region).reaction_xs["(n,gamma)"][k]


def test_absent_nuclide_is_at_infinite_dilution(library):
    update = bondarenko_update(_table(library))
    shielded = update(library, _densities(library, 0.01, 0.0), 0)
    k = library.index["fuel2"]
    np.testing.assert_allclose(shielded.reaction_xs["(n,gamma)"][k], INF_CAPTURE)
    np.testing.assert_allclose(shielded.xs_total[k], library.xs_total[k])


def test_background_on_a_grid_point(library):
    # fuel2's background: fuel's 0.01 * 10 b over its own 0.01 = 10 b.
    update = bondarenko_update(_table(library))
    capture = _capture(library, update, _densities(library, 0.01, 0.01))
    np.testing.assert_allclose(capture, INF_CAPTURE * F_CAPTURE[:, 0])


def test_interpolates_in_log_sigma0(library):
    # 10^1.5 b sits half way between the grid points in log(sigma0).
    update = bondarenko_update(_table(library))
    n2 = 0.01 * 10.0 / 10.0**1.5
    capture = _capture(library, update, _densities(library, 0.01, n2))
    np.testing.assert_allclose(capture, INF_CAPTURE * F_CAPTURE.mean(axis=1))


def test_between_the_grid_and_infinite_dilution(library):
    update = bondarenko_update(_table(library))
    sigma0 = 1.0e4
    capture = _capture(library, update, _densities(library, 0.01, 0.1 / sigma0))
    w = math.log(sigma0 / 100.0) / math.log(1.0e10 / 100.0)
    np.testing.assert_allclose(
        capture, INF_CAPTURE * ((1 - w) * F_CAPTURE[:, 1] + w * 1.0)
    )


def test_escape_per_region(library):
    # An escape of 0.1 /cm doubles fuel2's 10 b background to 20 b in region
    # 1; region 0 (outside the fuel) keeps the homogeneous 10 b.
    update = bondarenko_update(_table(library), escape=[0.0, 0.1])
    n = _densities(library, 0.01, 0.01)
    w = math.log(2.0) / math.log(10.0)
    expected = INF_CAPTURE * ((1 - w) * F_CAPTURE[:, 0] + w * F_CAPTURE[:, 1])
    np.testing.assert_allclose(
        _capture(library, update, n, 0), INF_CAPTURE * F_CAPTURE[:, 0]
    )
    np.testing.assert_allclose(_capture(library, update, n, 1), expected)


def test_stored_escape_is_the_default(library):
    update = bondarenko_update(_table(library, escape=0.1))
    n = _densities(library, 0.01, 0.01)
    assert not np.allclose(_capture(library, update, n), INF_CAPTURE * F_CAPTURE[:, 0])


def test_scatter_difference_goes_into_the_incident_column(library):
    update = bondarenko_update(_table(library))
    k = library.index["fuel2"]
    shielded = update(library, _densities(library, 0.01, 0.01), 0)
    expected = library.xs_scatter[k].copy()
    expected[:, 1] += DELTA[:, 0]  # [g_out, g_in]: incident group 1
    np.testing.assert_allclose(shielded.xs_scatter[k], expected)


def test_other_nuclides_are_untouched(library):
    update = bondarenko_update(_table(library))
    shielded = update(library, _densities(library, 0.01, 0.01), 0)
    k = library.index["fuel2"]
    others = [i for i in range(library.n_nuclides) if i != k]
    np.testing.assert_array_equal(shielded.xs_total[others], library.xs_total[others])
    # The library passed in is not modified.
    np.testing.assert_array_equal(
        library.reaction_xs["(n,gamma)"],
        synthetic_library(groups=2).reaction_xs["(n,gamma)"],
    )


def test_works_on_a_subset(library):
    update = bondarenko_update(_table(library))
    sub = library.subset(["fuel2", "fuel"])
    capture = update(sub, np.array([0.01, 0.01]), 0).reaction_xs["(n,gamma)"][0]
    np.testing.assert_allclose(capture, INF_CAPTURE * F_CAPTURE[:, 0])
    # Without the tabled nuclide the library comes back as it was.
    rest = library.subset(["fuel", "fp"])
    assert update(rest, np.array([0.01, 0.0]), 0) is rest


def test_reads_an_npz(library, tmp_path):
    path = tmp_path / "lib.npz"
    np.savez(path, **_table(library))
    capture = _capture(
        library, bondarenko_update(str(path)), _densities(library, 0.01, 0.01)
    )
    np.testing.assert_allclose(capture, INF_CAPTURE * F_CAPTURE[:, 0])


def test_below_the_grid_is_held_flat(library):
    # fuel2's background is 0.1 b, below the lowest node at 10 b.
    update = bondarenko_update(_table(library))
    capture = _capture(library, update, _densities(library, 0.01, 1.0))
    np.testing.assert_allclose(capture, INF_CAPTURE * F_CAPTURE[:, 0])


def test_lone_absorber_is_fully_shielded(library):
    # No other scatterers and no escape: sigma0 = 0, not infinite dilution.
    update = bondarenko_update(_table(library))
    capture = _capture(library, update, _densities(library, 0.0, 0.01))
    np.testing.assert_allclose(capture, INF_CAPTURE * F_CAPTURE[:, 0])


def test_total_uses_its_own_factor(library):
    update = bondarenko_update(_table(library))
    k = library.index["fuel2"]
    n2 = 0.01 * 10.0 / 10.0**1.5
    shielded = update(library, _densities(library, 0.01, n2), 0)
    np.testing.assert_allclose(
        shielded.xs_total[k], library.xs_total[k] * F_TOTAL.mean(axis=1)
    )


def test_rejects_a_bad_table(library):
    table = _table(library)
    table["shield_sigma0"] = GRID[::-1]
    with pytest.raises(ValueError, match="ascending"):
        bondarenko_update(table)
    table = _table(library)
    table["shield_f_capture"] = F_CAPTURE[None, :, :1]
    with pytest.raises(ValueError, match="shield_f_capture"):
        bondarenko_update(table)
    with pytest.raises(ValueError, match="escape"):
        bondarenko_update(_table(library), escape=-0.1)


# -- the burnup hook ----------------------------------------------------------


def _run(library, densities=None, medium_map=None, **kwargs):
    cells = 30
    if medium_map is None:
        medium_map = np.zeros(cells, dtype=np.int32)
    if densities is None:
        densities = np.array([[0.04, 0.0, 0.0, 0.0]])
    delta_x = np.full(cells, 20.0 / cells)
    angle_x, angle_w = discrete1.angular_x(8, bc_x=[0, 0])
    return burnup1d.burnup(
        library,
        densities,
        medium_map,
        delta_x,
        angle_x,
        angle_w,
        bc_x=[0, 0],
        dt_steps=[2.6e6, 2.6e6],
        power=1.0e6,
        **kwargs,
    )


def test_identity_update_changes_nothing(library):
    plain = _run(library)
    seen = []

    def identity(lib, densities, region):
        seen.append(region)
        return lib

    hooked = _run(library, xs_update=identity)
    # Per-region sums round differently from one matrix product (1e-16),
    # which power iteration carries up to its convergence tolerance in k.
    np.testing.assert_allclose(hooked[0], plain[0], rtol=1e-6, atol=1e-20)
    np.testing.assert_allclose(hooked[1], plain[1], rtol=1e-6)
    # Two steps x (predictor + corrector) plus end of life, one region.
    assert len(seen) == 5


def test_update_is_called_with_the_current_composition(library):
    compositions = []

    def record(lib, densities, region):
        compositions.append(densities.copy())
        return lib

    history, _ = _run(library, xs_update=record)
    np.testing.assert_array_equal(compositions[0], history[0, 0])
    np.testing.assert_array_equal(compositions[-1], history[-1, 0])


def test_shielding_follows_the_bred_nuclide(library):
    """Shield the capture of a nuclide bred from zero.

    fuel2 starts at infinite dilution and shields as it builds up, so the run
    differs from the unshielded one.
    """
    k = library.index["fuel2"]
    library.reaction_xs["(n,gamma)"][k] = INF_CAPTURE
    plain, _ = _run(library)
    shielded, _ = _run(library, xs_update=bondarenko_update(_table(library)))
    # Less fuel2 capture under shielding leaves more fuel2 behind.
    assert shielded[-1, 0, k] > plain[-1, 0, k]


def test_update_must_keep_the_nuclides(library):
    with pytest.raises(ValueError, match="same nuclides"):
        _run(library, xs_update=lambda lib, n, r: lib.subset(["fuel", "fuel2"]))


def test_macroscopic_xs_per_region_libraries(library):
    densities = np.array([[0.04, 0.0, 0.0, 0.0], [0.02, 0.01, 0.0, 0.0]])
    other = synthetic_library(groups=2)
    other.xs_total[:] *= 2.0
    total, _, _ = burnup1d.macroscopic_xs([library, other], densities)
    np.testing.assert_allclose(total[0], densities[0] @ library.xs_total)
    np.testing.assert_allclose(total[1], densities[1] @ other.xs_total)


def test_macroscopic_xs_needs_one_library_per_region(library):
    densities = np.zeros((2, library.n_nuclides))
    with pytest.raises(ValueError, match="one library per region"):
        burnup1d.macroscopic_xs([library], densities)


def test_region_chi(library):
    other = synthetic_library(groups=2)
    other.chi[:] = [0.75, 0.25]
    chi = burnup1d._region_chi([library, other], 2)
    np.testing.assert_array_equal(chi, [library.chi, other.chi])
    shared = burnup1d._region_chi(library, 2)
    np.testing.assert_array_equal(shared, [library.chi, library.chi])


def test_fuel_and_non_fuel_regions(library):
    k = library.index["fuel2"]
    library.reaction_xs["(n,gamma)"][k] = INF_CAPTURE
    update = bondarenko_update(_table(library), escape=[0.1, 0.0])
    captures = {0: [], 1: []}

    def record(lib, densities, region):
        shielded = update(lib, densities, region)
        captures[region].append(shielded.reaction_xs["(n,gamma)"][k])
        return shielded

    medium_map = np.repeat(np.array([0, 1], dtype=np.int32), 15)
    densities = np.array([[0.04, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.05]])
    history, keff = _run(library, densities, medium_map, xs_update=record)

    assert len(captures[0]) == len(captures[1]) == 5
    assert np.all(np.isfinite(keff))
    # fuel2 starts at infinite dilution and shields once it is bred
    np.testing.assert_allclose(captures[0][0], INF_CAPTURE)
    assert np.all(captures[0][-1] < INF_CAPTURE)
    # None reaches the non-fuel region, which stays at infinite dilution
    assert np.all(history[:, 1, k] == 0.0)
    for capture in captures[1]:
        np.testing.assert_allclose(capture, INF_CAPTURE)
