"""Physics benchmark tests for the burnup / depletion stack.

Unlike the synthetic-library unit tests, these check the solver against
problems with **known analytic solutions** and realistic (textbook) nuclear
data:

- the Xe-135 / I-135 equilibrium under constant flux, and
- the exponential burndown of U-235 by capture + fission under constant flux,
  both directly via :func:`discrete1.depletion.deplete` and through the full
  transport-coupled :func:`discrete1.burnup1d.burnup` driver.

The analytic reference for each case is derived from the same constants placed
in the library, so every test is a self-consistent check that the *solver*
reproduces the known physics of the chain it was given. Cross sections,
half-lives, and fission yields are standard thermal-spectrum values (Lamarsh;
Duderstadt & Hamilton).
"""

import math

import numpy as np
import pytest

import discrete1
import discrete1.constants as const
from discrete1 import burnup1d, depletion
from discrete1.nuclides import NuclideLibrary

pytestmark = pytest.mark.depletion

HOUR = 3600.0
LN2 = math.log(2.0)

# --- Xe-135 chain (thermal-spectrum textbook values) ---
T_I135 = 6.57 * HOUR  # I-135 half-life (s)
T_XE135 = 9.14 * HOUR  # Xe-135 half-life (s)
SIG_F_U235 = 585.0  # U-235 fission cross section (barns)
SIG_A_XE135 = 2.65e6  # Xe-135 (n,gamma) absorption (barns, 2200 m/s)
GAMMA_I = 0.0639  # U-235 thermal cumulative I-135 fission yield
GAMMA_XE = 0.00237  # U-235 thermal independent Xe-135 fission yield

# --- U-235 burndown ---
SIG_C_U235 = 99.0  # U-235 (n,gamma) capture cross section (barns)


def _empty_transport_fields(m, g):
    """Minimal valid transport-coupling arrays (shapes the schema requires)."""
    xs_total = np.zeros((m, g))
    xs_scatter = np.zeros((m, g, g))
    nu_fission = np.zeros((m, g))
    chi = np.full(g, 1.0 / g)
    return xs_total, xs_scatter, nu_fission, chi


def _xe135_library():
    """Build the U-235 -> I-135 -> Xe-135 -> Cs-135 chain (single group).

    Xe-135 both beta-decays to Cs-135 and absorbs neutrons (n,gamma) to a
    stable Xe-136 sink. U-235 fissions, feeding I-135 and Xe-135 directly via
    fission yields.
    """
    names = ["u235", "i135", "xe135", "xe136", "cs135"]
    m, g = len(names), 1
    idx = {name: i for i, name in enumerate(names)}

    half_life = np.array([np.inf, T_I135, T_XE135, np.inf, np.inf])

    # Decay: i135 -> xe135 -> cs135 (full branching).
    decay_from = np.array([idx["i135"], idx["xe135"]], dtype=np.int64)
    decay_to = np.array([idx["xe135"], idx["cs135"]], dtype=np.int64)
    decay_branch = np.array([1.0, 1.0])

    # (n,gamma): xe135 -> xe136 absorption.
    cap_xs = np.zeros((m, g))
    cap_xs[idx["xe135"]] = SIG_A_XE135
    cap_product = np.full(m, -1, dtype=np.int64)
    cap_product[idx["xe135"]] = idx["xe136"]
    reaction_xs = {"(n,gamma)": cap_xs}
    reaction_product = {"(n,gamma)": cap_product}

    # Fission of u235 feeds i135 and xe135 directly.
    fission_xs = np.zeros((m, g))
    fission_xs[idx["u235"]] = SIG_F_U235
    fy_parent = np.array([idx["u235"], idx["u235"]], dtype=np.int64)
    fy_product = np.array([idx["i135"], idx["xe135"]], dtype=np.int64)
    fy_yield = np.array([GAMMA_I, GAMMA_XE])
    kappa = np.zeros(m)
    kappa[idx["u235"]] = 3.2e-11

    xs_total, xs_scatter, nu_fission, chi = _empty_transport_fields(m, g)
    nu_fission[idx["u235"]] = 2.4 * SIG_F_U235

    return NuclideLibrary(
        names=names,
        groups=g,
        half_life=half_life,
        decay_from=decay_from,
        decay_to=decay_to,
        decay_branch=decay_branch,
        reaction_xs=reaction_xs,
        reaction_product=reaction_product,
        fy_parent=fy_parent,
        fy_product=fy_product,
        fy_yield=fy_yield,
        fission_xs=fission_xs,
        kappa=kappa,
        xs_total=xs_total,
        xs_scatter=xs_scatter,
        nu_fission=nu_fission,
        chi=chi,
    )


def _u235_library():
    """Build a U-235 / U-236 / fission-product library (single group).

    U-235 is removed by (n,gamma) capture to U-236 and by fission to a lumped
    fission-product sink, so its number density decays exponentially under
    constant flux. Transport fields are populated so the same library can drive
    the transport-coupled :func:`discrete1.burnup1d.burnup`.
    """
    names = ["u235", "u236", "fp"]
    m, g = len(names), 1
    idx = {name: i for i, name in enumerate(names)}

    half_life = np.array([np.inf, np.inf, np.inf])

    # No decay couplings (all stable on this timescale).
    decay_from = np.array([], dtype=np.int64)
    decay_to = np.array([], dtype=np.int64)
    decay_branch = np.array([])

    # (n,gamma): u235 -> u236.
    cap_xs = np.zeros((m, g))
    cap_xs[idx["u235"]] = SIG_C_U235
    cap_product = np.full(m, -1, dtype=np.int64)
    cap_product[idx["u235"]] = idx["u236"]
    reaction_xs = {"(n,gamma)": cap_xs}
    reaction_product = {"(n,gamma)": cap_product}

    # Fission of u235 yields 2 fission-product atoms.
    fission_xs = np.zeros((m, g))
    fission_xs[idx["u235"]] = SIG_F_U235
    fy_parent = np.array([idx["u235"]], dtype=np.int64)
    fy_product = np.array([idx["fp"]], dtype=np.int64)
    fy_yield = np.array([2.0])
    kappa = np.zeros(m)
    kappa[idx["u235"]] = 3.2e-11

    xs_total, xs_scatter, nu_fission, chi = _empty_transport_fields(m, g)
    nu_fission[idx["u235"]] = 2.4 * SIG_F_U235
    xs_total[idx["u235"]] = SIG_F_U235 + SIG_C_U235 + 10.0
    xs_total[idx["u236"]] = 8.0
    xs_total[idx["fp"]] = 4.0
    for i, sigma_s in enumerate([10.0, 6.0, 3.0]):
        xs_scatter[i] = np.full((g, g), sigma_s)

    return NuclideLibrary(
        names=names,
        groups=g,
        half_life=half_life,
        decay_from=decay_from,
        decay_to=decay_to,
        decay_branch=decay_branch,
        reaction_xs=reaction_xs,
        reaction_product=reaction_product,
        fy_parent=fy_parent,
        fy_product=fy_product,
        fy_yield=fy_yield,
        fission_xs=fission_xs,
        kappa=kappa,
        xs_total=xs_total,
        xs_scatter=xs_scatter,
        nu_fission=nu_fission,
        chi=chi,
    )


def _slab_setup(cells=40, length=20.0, angles=8):
    """Single-region vacuum slab (mirrors tests/test_burnup.py)."""
    edges_x = np.linspace(0.0, length, cells + 1)
    delta_x = np.diff(edges_x)
    medium_map = np.zeros(cells, dtype=np.int32)
    angle_x, angle_w = discrete1.angular_x(angles, bc_x=[0, 0])
    return medium_map, delta_x, angle_x, angle_w


def test_xe135_reaches_analytic_equilibrium():
    # Saturate the I-135 / Xe-135 chain under constant flux and compare with
    # the closed-form equilibrium concentrations.
    lib = _xe135_library()
    i = lib.index
    phi = 1.0e13
    flux = np.array([phi])
    n0 = np.zeros(lib.n_nuclides)
    n0[i["u235"]] = 0.02

    # ~7 days: I-135 (tau ~ 9.5 h) and Xe-135 fully equilibrate; U-235 burns <0.5 %.
    n1 = depletion.deplete(lib, n0, flux, dt=6.0e5, order=48)

    # Reference uses the (barely changed) final U-235 density: the fast chain
    # tracks the slowly evolving fission source quasi-statically.
    fission_rate = SIG_F_U235 * phi * const.CM_TO_BARNS  # 1/s per U-235 atom
    fission_source = fission_rate * n1[i["u235"]]
    lam_i = LN2 / T_I135
    lam_xe = LN2 / T_XE135
    abs_xe = SIG_A_XE135 * phi * const.CM_TO_BARNS

    n_i_eq = GAMMA_I * fission_source / lam_i
    n_xe_eq = (GAMMA_I + GAMMA_XE) * fission_source / (lam_xe + abs_xe)

    assert n1[i["i135"]] == pytest.approx(n_i_eq, rel=1e-2)
    assert n1[i["xe135"]] == pytest.approx(n_xe_eq, rel=1e-2)
    # Cs-135 (decay sink) accumulates; nothing goes negative.
    assert n1[i["cs135"]] > 0.0
    assert np.all(n1 >= -1e-15)


def test_u235_burndown_matches_exponential():
    # Constant flux -> constant burnup matrix -> exact exponential decay.
    lib = _u235_library()
    i = lib.index
    phi = 1.0e14
    flux = np.array([phi])
    n0 = np.zeros(lib.n_nuclides)
    n0[i["u235"]] = 0.02

    dt = 2.0e7  # ~230 days
    n1 = depletion.deplete(lib, n0, flux, dt=dt, order=48)

    removal = (SIG_C_U235 + SIG_F_U235) * phi * const.CM_TO_BARNS
    reference = n0[i["u235"]] * math.exp(-removal * dt)
    assert n1[i["u235"]] == pytest.approx(reference, rel=1e-8)

    # Atom balance: U-235 lost = U-236 gained (capture) + fissions (fp/2).
    u235_lost = n0[i["u235"]] - n1[i["u235"]]
    fissions = n1[i["fp"]] / 2.0
    assert n1[i["u236"]] + fissions == pytest.approx(u235_lost, rel=1e-8)


@pytest.mark.burnup
def test_u235_burndown_through_burnup_driver():
    # The full transport-coupled driver, with flux pinned constant via
    # `flux_level`, must reproduce the same exponential burndown.
    lib = _u235_library()
    i = lib.index
    medium_map, delta_x, angle_x, angle_w = _slab_setup()
    densities0 = np.zeros((1, lib.n_nuclides))
    densities0[0, i["u235"]] = 0.02

    phi = 1.0e14
    dt = 5.0e6
    n_steps = 4
    history, _ = burnup1d.burnup(
        lib,
        densities0,
        medium_map,
        delta_x,
        angle_x,
        angle_w,
        bc_x=[0, 0],
        dt_steps=[dt] * n_steps,
        flux_level=phi,
        order=48,
    )

    removal = (SIG_C_U235 + SIG_F_U235) * phi * const.CM_TO_BARNS
    for step in range(n_steps + 1):
        t = step * dt
        reference = densities0[0, i["u235"]] * math.exp(-removal * t)
        assert history[step, 0, i["u235"]] == pytest.approx(reference, rel=1e-4)
