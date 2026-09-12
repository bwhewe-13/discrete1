"""Unit tests for DepletionChain, the preset registry, and nuclide selection.

The important behaviour pinned here is the **truncation semantics** of
``subset``: dropping a nuclide that another nuclide produces must keep the
parent's loss of atoms while removing the production. Getting that backwards
would silently conserve (or destroy) atoms in every restricted problem.
"""

import numpy as np
import pytest

from discrete1.chains import available_chains, load_chain, save_chain
from discrete1.depletion import build_burnup_matrix
from discrete1.nuclides import (
    REACTIONS,
    library_from_chain,
    synthetic_chain,
    synthetic_library,
)

pytestmark = pytest.mark.depletion

# A flux big enough that every reaction rate is well clear of the decay terms.
FLUX = np.array([1.0e14])


########################################################################
# Chain round-trip and the packaged presets
########################################################################


def test_save_load_chain_round_trip(tmp_path):
    chain = synthetic_chain()
    path = tmp_path / "chain.npz"
    save_chain(chain, path)
    loaded = load_chain(str(path))

    assert loaded.names == chain.names
    assert all(type(name) is str for name in loaded.names)
    assert np.array_equal(loaded.half_life, chain.half_life)
    assert np.array_equal(loaded.decay_from, chain.decay_from)
    assert np.array_equal(loaded.decay_to, chain.decay_to)
    assert np.array_equal(loaded.decay_branch, chain.decay_branch)
    assert np.array_equal(loaded.fy_parent, chain.fy_parent)
    assert np.array_equal(loaded.fy_product, chain.fy_product)
    assert np.array_equal(loaded.fy_yield, chain.fy_yield)
    assert np.array_equal(loaded.kappa, chain.kappa)
    assert loaded.reaction_product.keys() == chain.reaction_product.keys()
    for channel, product in chain.reaction_product.items():
        assert np.array_equal(loaded.reaction_product[channel], product)
    assert loaded.index == chain.index


def test_presets_are_available_and_valid():
    presets = available_chains()
    assert "lwr-actinides" in presets
    for name in presets:
        load_chain(name).validate()  # __post_init__ already ran it; explicit


def test_load_chain_rejects_unknown_preset():
    with pytest.raises(ValueError, match="unknown chain"):
        load_chain("not-a-real-chain")


def test_lwr_actinides_topology():
    chain = load_chain("lwr-actinides")
    idx = chain.index

    # The two documented shortcuts through short-lived intermediates.
    assert chain.reaction_product["(n,gamma)"][idx["U238"]] == idx["Np239"]
    assert chain.reaction_product["(n,gamma)"][idx["Np237"]] == idx["Pu238"]

    # Decay couplings that feed the actinide line.
    couplings = dict(zip(chain.decay_from.tolist(), chain.decay_to.tolist()))
    assert couplings[idx["Np239"]] == idx["Pu239"]
    assert couplings[idx["Pu241"]] == idx["Am241"]
    assert couplings[idx["I135"]] == idx["Xe135"]

    # Stable backgrounds contribute no decay loss.
    lam = chain.decay_constant
    assert lam[idx["H1"]] == 0.0
    assert lam[idx["O16"]] == 0.0
    assert lam[idx["Sm149"]] == 0.0


########################################################################
# subset: identity, reordering, and truncation
########################################################################


def _matrix(library, flux=FLUX):
    """Dense burnup matrix for a one-group library."""
    return build_burnup_matrix(library, flux).toarray()


def test_subset_of_everything_is_the_identity():
    library = synthetic_library(groups=1)
    same = library.subset(library.names)

    assert same.names == library.names
    assert np.array_equal(_matrix(same), _matrix(library))


def test_subset_does_not_alias_the_source():
    library = synthetic_library(groups=1)
    same = library.subset(library.names)
    same.half_life[0] = 1.0
    same.fission_xs[0, 0] = 999.0

    assert np.isinf(library.half_life[0])
    assert library.fission_xs[0, 0] == 10.0


def test_subset_reorders_consistently():
    library = synthetic_library(groups=1)
    order = ["fp_stable", "fuel", "fp", "fuel2"]
    permuted = library.subset(order)

    assert permuted.names == order
    # The permuted matrix is the original under the same row/column shuffle.
    take = [library.index[name] for name in order]
    expected = _matrix(library)[np.ix_(take, take)]
    assert np.allclose(_matrix(permuted), expected)


def test_subset_keeps_loss_and_drops_production():
    """Dropping a capture product keeps the parent's absorption loss."""
    library = synthetic_library(groups=1)
    full = _matrix(library)
    fuel, fuel2 = library.index["fuel"], library.index["fuel2"]

    # "fuel2" only exists as the (n,gamma) product of "fuel".
    kept = ["fuel", "fp", "fp_stable"]
    reduced = library.subset(kept)
    small = _matrix(reduced)
    new_fuel = reduced.index["fuel"]

    # Production of fuel2 is gone (fuel2 is no longer a row at all)...
    assert full[fuel2, fuel] > 0.0
    assert "fuel2" not in reduced.index
    # ...but the diagonal loss of fuel is untouched: it still absorbs.
    assert small[new_fuel, new_fuel] == pytest.approx(full[fuel, fuel])
    assert reduced.reaction_product["(n,gamma)"][new_fuel] == -1


def test_subset_keeps_decay_loss_when_daughter_is_dropped():
    library = synthetic_library(groups=1)
    full = _matrix(library)
    fp = library.index["fp"]

    reduced = library.subset(["fuel", "fuel2", "fp"])
    small = _matrix(reduced)
    new_fp = reduced.index["fp"]

    # fp still decays away at the same rate, it just produces nothing.
    assert small[new_fp, new_fp] == pytest.approx(full[fp, fp])
    assert reduced.decay_to.tolist() == [-1]
    assert reduced.decay_from.tolist() == [new_fp]


def test_subset_dropping_a_fission_parent_removes_its_yields():
    library = synthetic_library(groups=1)
    reduced = library.subset(["fp", "fp_stable"])

    # "fuel" was the only fissioning nuclide; its yield couplings go with it.
    assert reduced.fy_parent.size == 0
    assert reduced.fy_product.size == 0
    assert reduced.fy_yield.size == 0


def test_subset_rejects_unknown_and_duplicate_names():
    library = synthetic_library(groups=1)
    with pytest.raises(ValueError, match="unknown nuclide"):
        library.subset(["fuel", "plutonium"])
    with pytest.raises(ValueError, match="unique"):
        library.subset(["fuel", "fuel"])


def test_chain_subset_matches_library_subset():
    """The two subset implementations agree on the topology they produce."""
    library = synthetic_library(groups=1)
    kept = ["fp", "fuel"]

    from_library = library.subset(kept)
    from_chain = library.chain().subset(kept)

    assert from_chain.names == from_library.names
    assert np.array_equal(from_chain.decay_from, from_library.decay_from)
    assert np.array_equal(from_chain.decay_to, from_library.decay_to)
    assert np.array_equal(from_chain.fy_parent, from_library.fy_parent)
    assert np.array_equal(from_chain.fy_product, from_library.fy_product)
    assert np.array_equal(from_chain.kappa, from_library.kappa)
    for channel, product in from_library.reaction_product.items():
        assert np.array_equal(from_chain.reaction_product[channel], product)


########################################################################
# Regression: the preset reproduces the PyCNiC example's vetted topology
########################################################################

# examples/depletion_pycnic_slab.py used to hand-build this chain inline. It
# now selects it out of the "lwr-actinides" preset, so these literals are kept
# here as the reference: any edit to the preset that changes the example's
# burnup matrix has to fail this test first.
DAY = 86400.0
YEAR = 3.1557e7
HOUR = 3600.0

EXAMPLE_ISOTOPES = [
    "U235",
    "U236",
    "U238",
    "Np239",
    "Pu239",
    "Pu240",
    "Pu241",
    "I135",
    "Xe135",
    "Sm149",
    "H1",
    "O16",
]


def _example_chain():
    """Build the example's original hand-built topology, verbatim."""
    from discrete1.nuclides import DepletionChain

    idx = {name: i for i, name in enumerate(EXAMPLE_ISOTOPES)}
    m = len(EXAMPLE_ISOTOPES)

    half_life = np.array(
        [
            7.04e8 * YEAR,  # U-235
            2.342e7 * YEAR,  # U-236
            4.468e9 * YEAR,  # U-238
            2.356 * DAY,  # Np-239 -> Pu-239
            2.411e4 * YEAR,  # Pu-239
            6.561e3 * YEAR,  # Pu-240
            14.329 * YEAR,  # Pu-241 -> Am-241 (untracked)
            6.57 * HOUR,  # I-135  -> Xe-135
            9.14 * HOUR,  # Xe-135 -> Cs-135 (untracked)
            np.inf,  # Sm-149
            np.inf,  # H-1
            np.inf,  # O-16
        ]
    )

    reaction_product = {ch: np.full(m, -1, dtype=np.int64) for ch in REACTIONS}
    capture = reaction_product["(n,gamma)"]
    capture[idx["U235"]] = idx["U236"]
    capture[idx["U238"]] = idx["Np239"]
    capture[idx["Pu239"]] = idx["Pu240"]
    capture[idx["Pu240"]] = idx["Pu241"]
    n2n = reaction_product["(n,2n)"]
    n2n[idx["U236"]] = idx["U235"]
    n2n[idx["Pu240"]] = idx["Pu239"]
    n2n[idx["Pu241"]] = idx["Pu240"]

    yields = {
        "U235": {"I135": 0.0639, "Xe135": 0.00237, "Sm149": 0.0113},
        "U238": {"I135": 0.0610, "Xe135": 0.0010, "Sm149": 0.0090},
        "Pu239": {"I135": 0.0654, "Xe135": 0.0110, "Sm149": 0.0125},
        "Pu241": {"I135": 0.0693, "Xe135": 0.0070, "Sm149": 0.0140},
    }
    fy_parent, fy_product, fy_yield = [], [], []
    for parent, products in yields.items():
        for product, value in products.items():
            fy_parent.append(idx[parent])
            fy_product.append(idx[product])
            fy_yield.append(value)

    kappa = np.zeros(m)
    kappa[idx["U235"]] = 3.24e-11
    kappa[idx["U236"]] = 3.2e-11
    kappa[idx["U238"]] = 3.35e-11
    kappa[idx["Np239"]] = 3.2e-11
    kappa[idx["Pu239"]] = 3.33e-11
    kappa[idx["Pu240"]] = 3.2e-11
    kappa[idx["Pu241"]] = 3.44e-11

    return DepletionChain(
        names=list(EXAMPLE_ISOTOPES),
        half_life=half_life,
        decay_from=np.array([idx["Np239"], idx["I135"]], dtype=np.int64),
        decay_to=np.array([idx["Pu239"], idx["Xe135"]], dtype=np.int64),
        decay_branch=np.array([1.0, 1.0]),
        reaction_product=reaction_product,
        fy_parent=np.array(fy_parent, dtype=np.int64),
        fy_product=np.array(fy_product, dtype=np.int64),
        fy_yield=np.array(fy_yield, dtype=np.float64),
        kappa=kappa,
    )


def _library_for(chain, groups, seed=0):
    """Attach reproducible pseudo-random cross sections to a chain."""
    rng = np.random.default_rng(seed)
    m = chain.n_nuclides
    return library_from_chain(
        chain,
        groups,
        fission_xs=rng.random((m, groups)) * 5.0,
        xs_total=np.ones((m, groups)),
        xs_scatter=np.ones((m, groups, groups)),
        nu_fission=np.ones((m, groups)),
        chi=np.full(groups, 1.0 / groups),
        reaction_xs={
            "(n,gamma)": rng.random((m, groups)) * 3.0,
            "(n,2n)": rng.random((m, groups)) * 0.1,
        },
    )


def test_preset_subset_matches_example_chain():
    reference = _example_chain()
    selected = load_chain("lwr-actinides").subset(EXAMPLE_ISOTOPES)

    assert selected.names == reference.names
    assert np.array_equal(selected.half_life, reference.half_life)
    assert np.array_equal(selected.kappa, reference.kappa)

    # The couplings need not be stored in the same COO order -- what has to
    # match is the matrix they assemble. (The preset carries Pu-241 -> Am-241
    # and Xe-135 -> Cs-135 with a -1 product after selection, where the
    # reference simply has no entry; both contribute nothing.)
    groups = 4
    rng = np.random.default_rng(12345)
    flux = rng.random(groups) * 1.0e14
    left = _matrix(_library_for(reference, groups), flux)
    right = _matrix(_library_for(selected, groups), flux)
    assert np.array_equal(left, right)


def test_dropped_daughter_coupling_contributes_nothing():
    """A `-1` product entry must not change the matrix at all."""
    chain = load_chain("lwr-actinides").subset(EXAMPLE_ISOTOPES)
    assert -1 in chain.decay_to.tolist()  # Pu-241 -> Am-241, Xe-135 -> Cs-135

    library = _library_for(chain, 2)
    flux = np.array([1.0e14, 5.0e13])
    with_sentinels = _matrix(library, flux)

    # Physically deleting those entries must give the same matrix.
    keep = chain.decay_to >= 0
    library.decay_from = chain.decay_from[keep]
    library.decay_to = chain.decay_to[keep]
    library.decay_branch = chain.decay_branch[keep]
    library.validate()
    assert np.array_equal(_matrix(library, flux), with_sentinels)


########################################################################
# library_from_chain
########################################################################


def test_library_from_chain_round_trips_through_chain():
    library = synthetic_library(groups=2)
    rebuilt = library_from_chain(
        library.chain(),
        library.groups,
        library.fission_xs,
        library.xs_total,
        library.xs_scatter,
        library.nu_fission,
        library.chi,
        reaction_xs=library.reaction_xs,
    )
    assert rebuilt.names == library.names
    flux = np.array([1.0e14, 5.0e13])
    assert np.array_equal(_matrix(rebuilt, flux), _matrix(library, flux))


def test_library_from_chain_copies_chain_arrays():
    chain = synthetic_chain()
    library = synthetic_library(groups=1)
    built = library_from_chain(
        chain,
        1,
        library.fission_xs,
        library.xs_total,
        library.xs_scatter,
        library.nu_fission,
        library.chi,
    )
    built.half_life[0] = 1.0
    assert np.isinf(chain.half_life[0])


def test_library_from_chain_validates_shapes():
    chain = synthetic_chain()
    library = synthetic_library(groups=1)
    with pytest.raises(ValueError, match="fission_xs"):
        library_from_chain(
            chain,
            2,  # claims two groups, arrays are one
            library.fission_xs,
            library.xs_total,
            library.xs_scatter,
            library.nu_fission,
            library.chi,
        )


def test_library_from_chain_requires_a_product_for_each_channel():
    chain = synthetic_chain()
    library = synthetic_library(groups=1)
    unmapped = next(ch for ch in REACTIONS if ch not in chain.reaction_product)
    with pytest.raises(ValueError, match="missing"):
        library_from_chain(
            chain,
            1,
            library.fission_xs,
            library.xs_total,
            library.xs_scatter,
            library.nu_fission,
            library.chi,
            reaction_xs={unmapped: np.zeros((chain.n_nuclides, 1))},
        )
