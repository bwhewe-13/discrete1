"""Unit tests for the NuclideLibrary schema, validation, and npz round-trip."""

import numpy as np
import pytest

from discrete1.nuclides import load_library, save_library, synthetic_library

pytestmark = pytest.mark.depletion


def test_decay_constant_stable_is_zero():
    lib = synthetic_library(groups=1)
    lam = lib.decay_constant
    assert lam[lib.index["fuel"]] == 0.0
    assert lam[lib.index["fp"]] == pytest.approx(np.log(2.0) / 3.0e4)


def test_validate_rejects_nonpositive_half_life():
    lib = synthetic_library(groups=1)
    lib.half_life = lib.half_life.copy()
    lib.half_life[0] = 0.0
    with pytest.raises(ValueError, match="half_life"):
        lib.validate()


def test_validate_rejects_out_of_range_decay_index():
    lib = synthetic_library(groups=1)
    lib.decay_to = np.array([lib.n_nuclides], dtype=np.int64)
    with pytest.raises(ValueError, match="decay_to"):
        lib.validate()


def test_validate_rejects_negative_parent_index():
    lib = synthetic_library(groups=1)
    lib.fy_parent = np.array([-1], dtype=np.int64)
    with pytest.raises(ValueError, match="fy_parent"):
        lib.validate()


def test_validate_allows_untracked_products():
    # -1 marks a decay daughter / fission product outside the tracked set.
    lib = synthetic_library(groups=1)
    lib.decay_to = np.array([-1], dtype=np.int64)
    lib.fy_product = np.array([-1], dtype=np.int64)
    lib.validate()


def test_validate_rejects_negative_branch_and_yield():
    lib = synthetic_library(groups=1)
    lib.decay_branch = np.array([-0.1])
    with pytest.raises(ValueError, match="decay_branch"):
        lib.validate()

    lib = synthetic_library(groups=1)
    lib.fy_yield = np.array([-2.0])
    with pytest.raises(ValueError, match="fy_yield"):
        lib.validate()


def test_validate_rejects_duplicate_names():
    lib = synthetic_library(groups=1)
    lib.names = ["fuel", "fuel", "fp", "fp_stable"]
    with pytest.raises(ValueError, match="unique"):
        lib.validate()


def test_save_load_round_trip(tmp_path):
    # Loads without pickle; names come back as plain str.
    lib = synthetic_library(groups=2)
    path = tmp_path / "library.npz"
    save_library(lib, path)
    loaded = load_library(path)

    assert loaded.names == lib.names
    assert all(type(name) is str for name in loaded.names)
    assert loaded.groups == lib.groups
    assert np.array_equal(loaded.half_life, lib.half_life)
    assert np.array_equal(loaded.decay_to, lib.decay_to)
    assert np.array_equal(loaded.fy_yield, lib.fy_yield)
    assert loaded.reaction_xs.keys() == lib.reaction_xs.keys()
    for channel, xs in lib.reaction_xs.items():
        assert np.array_equal(loaded.reaction_xs[channel], xs)
        assert np.array_equal(
            loaded.reaction_product[channel], lib.reaction_product[channel]
        )
    assert np.array_equal(loaded.xs_scatter, lib.xs_scatter)
    assert loaded.index == lib.index
