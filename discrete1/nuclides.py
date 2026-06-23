"""Nuclide library schema for burnup / activation calculations.

This module defines :class:`NuclideLibrary`, the data container consumed by
the depletion machinery (:mod:`discrete1.depletion`, :mod:`discrete1.burnup1d`).
It collects, for a fixed set of ``M`` tracked nuclides on a ``G``-group energy
grid:

- radioactive decay data (half-lives and branching to daughter nuclides),
- neutron reaction channels (microscopic multigroup cross sections plus the
  product nuclide of each channel),
- fission yields and energy released per fission,
- the microscopic transport cross sections used to rebuild macroscopic
  material cross sections as composition evolves.

The schema is deliberately index-based: every nuclide has a fixed integer
index ``0..M-1`` (see :attr:`NuclideLibrary.names`), and all couplings (decay
branches, reaction products, fission yields) are stored as parallel arrays of
those indices. This keeps the burnup-matrix assembly in
:func:`discrete1.depletion.build_burnup_matrix` a series of vectorized index
operations.

Sparse coupling arrays use a coordinate (COO) layout: a triple of
``(from_index, to_index, value)`` arrays. A self-contained
:func:`synthetic_library` builds a small valid library for tests and examples;
:func:`load_library` reads one back from a ``.npz`` archive.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

# Canonical neutron reaction channels that transmute one nuclide into another.
# Fission is handled separately because it produces many products via yields.
REACTIONS = ("(n,gamma)", "(n,2n)", "(n,3n)", "(n,p)", "(n,a)")

LN2 = math.log(2.0)


@dataclass
class NuclideLibrary:
    """Container of decay, reaction, fission, and transport data per nuclide.

    All per-nuclide arrays are indexed by the nuclide's position in
    :attr:`names`. Couplings between nuclides are stored as COO triples.

    Attributes
    ----------
    names : list of str
        Nuclide identifiers (e.g. ``"u-235"``), length ``M``. Index order
        defines the canonical nuclide ordering used everywhere else.
    groups : int
        Number of energy groups ``G``.
    half_life : numpy.ndarray, shape (M,)
        Half-lives in seconds. Use ``numpy.inf`` for stable nuclides.
    decay_from, decay_to : numpy.ndarray of int
        Parallel COO index arrays: parent ``decay_from[k]`` decays to
        daughter ``decay_to[k]``.
    decay_branch : numpy.ndarray, shape (n_decay,)
        Branching ratio for each decay coupling (per parent decay).
    reaction_xs : dict[str, numpy.ndarray]
        Maps a channel in :data:`REACTIONS` to a microscopic multigroup
        cross section array of shape ``(M, G)`` in barns.
    reaction_product : dict[str, numpy.ndarray]
        Maps a channel to an int array of shape ``(M,)`` giving the product
        nuclide index for each parent (``-1`` if the channel is absent or the
        product is not tracked).
    fy_parent, fy_product : numpy.ndarray of int
        Parallel COO index arrays for fission yields: fissioning nuclide
        ``fy_parent[k]`` produces ``fy_product[k]``.
    fy_yield : numpy.ndarray, shape (n_fy,)
        Number of ``fy_product`` atoms produced per fission of ``fy_parent``.
    fission_xs : numpy.ndarray, shape (M, G)
        Microscopic fission cross section in barns (drives fission-product
        production and burnup loss).
    kappa : numpy.ndarray, shape (M,)
        Energy released per fission in joules (for power normalization).
    xs_total : numpy.ndarray, shape (M, G)
        Microscopic total cross section in barns (transport coupling).
    xs_scatter : numpy.ndarray, shape (M, G, G)
        Microscopic scattering matrix in barns (transport coupling).
    nu_fission : numpy.ndarray, shape (M, G)
        Microscopic ``nu * sigma_f`` in barns (transport coupling).
    chi : numpy.ndarray, shape (G,)
        Fission neutron emission spectrum (sums to 1).
    """

    names: List[str]
    groups: int
    half_life: np.ndarray
    decay_from: np.ndarray
    decay_to: np.ndarray
    decay_branch: np.ndarray
    reaction_xs: Dict[str, np.ndarray]
    reaction_product: Dict[str, np.ndarray]
    fy_parent: np.ndarray
    fy_product: np.ndarray
    fy_yield: np.ndarray
    fission_xs: np.ndarray
    kappa: np.ndarray
    xs_total: np.ndarray
    xs_scatter: np.ndarray
    nu_fission: np.ndarray
    chi: np.ndarray
    index: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        """Build the name -> index lookup and validate array shapes."""
        if not self.index:
            self.index = {name: idx for idx, name in enumerate(self.names)}
        self.validate()

    @property
    def n_nuclides(self) -> int:
        """Number of tracked nuclides ``M``."""
        return len(self.names)

    @property
    def decay_constant(self) -> np.ndarray:
        """Decay constants ``lambda = ln(2) / half_life`` (1/s), 0 if stable."""
        with np.errstate(divide="ignore"):
            lam = LN2 / self.half_life
        lam[~np.isfinite(lam)] = 0.0
        return lam

    def validate(self):
        """Check internal shape and index consistency; raise on error."""
        m, g = self.n_nuclides, self.groups
        assert self.half_life.shape == (m,), "half_life must have shape (M,)"
        for arr, name in (
            (self.fission_xs, "fission_xs"),
            (self.nu_fission, "nu_fission"),
            (self.xs_total, "xs_total"),
        ):
            assert arr.shape == (m, g), f"{name} must have shape (M, G)"
        assert self.xs_scatter.shape == (m, g, g), "xs_scatter must be (M, G, G)"
        assert self.chi.shape == (g,), "chi must have shape (G,)"
        assert self.kappa.shape == (m,), "kappa must have shape (M,)"
        for channel in REACTIONS:
            if channel in self.reaction_xs:
                assert self.reaction_xs[channel].shape == (
                    m,
                    g,
                ), f"reaction_xs[{channel}] must have shape (M, G)"
                assert self.reaction_product[channel].shape == (
                    m,
                ), f"reaction_product[{channel}] must have shape (M,)"
        for arr in (self.decay_from, self.decay_to, self.decay_branch):
            assert arr.shape == self.decay_from.shape, "decay COO arrays misaligned"
        for arr in (self.fy_parent, self.fy_product, self.fy_yield):
            assert arr.shape == self.fy_parent.shape, "fission-yield COO misaligned"


def load_library(path):
    """Load a :class:`NuclideLibrary` from a ``.npz`` archive.

    Parameters
    ----------
    path : str
        Path to a ``.npz`` file written by :func:`save_library` (or with the
        same key layout).

    Returns
    -------
    NuclideLibrary
        Reconstructed library.
    """
    data = np.load(path, allow_pickle=True)
    names = list(data["names"])
    reaction_xs = {}
    reaction_product = {}
    for channel in REACTIONS:
        key = _channel_key(channel)
        if f"rxs_{key}" in data.files:
            reaction_xs[channel] = data[f"rxs_{key}"]
            reaction_product[channel] = data[f"rprod_{key}"]
    return NuclideLibrary(
        names=names,
        groups=int(data["groups"]),
        half_life=data["half_life"],
        decay_from=data["decay_from"],
        decay_to=data["decay_to"],
        decay_branch=data["decay_branch"],
        reaction_xs=reaction_xs,
        reaction_product=reaction_product,
        fy_parent=data["fy_parent"],
        fy_product=data["fy_product"],
        fy_yield=data["fy_yield"],
        fission_xs=data["fission_xs"],
        kappa=data["kappa"],
        xs_total=data["xs_total"],
        xs_scatter=data["xs_scatter"],
        nu_fission=data["nu_fission"],
        chi=data["chi"],
    )


def save_library(library, path):
    """Write a :class:`NuclideLibrary` to a ``.npz`` archive."""
    arrays = {
        "names": np.array(library.names),
        "groups": library.groups,
        "half_life": library.half_life,
        "decay_from": library.decay_from,
        "decay_to": library.decay_to,
        "decay_branch": library.decay_branch,
        "fy_parent": library.fy_parent,
        "fy_product": library.fy_product,
        "fy_yield": library.fy_yield,
        "fission_xs": library.fission_xs,
        "kappa": library.kappa,
        "xs_total": library.xs_total,
        "xs_scatter": library.xs_scatter,
        "nu_fission": library.nu_fission,
        "chi": library.chi,
    }
    for channel, xs in library.reaction_xs.items():
        key = _channel_key(channel)
        arrays[f"rxs_{key}"] = xs
        arrays[f"rprod_{key}"] = library.reaction_product[channel]
    np.savez(path, **arrays)


def _channel_key(channel):
    """Turn a reaction channel label into an npz-safe key fragment."""
    return channel.replace("(", "").replace(")", "").replace(",", "_")


def synthetic_library(groups=1):
    """Build a small, self-contained library for tests and examples.

    Models a four-nuclide system on ``groups`` energy groups:

    ``fuel`` (a fissile, capturing nuclide) captures a neutron to ``fuel2``
    and fissions into a ``fp`` (fission product) that beta-decays to a stable
    ``fp_stable``. This exercises decay, ``(n,gamma)`` capture, fission yields,
    and the transport-coupling cross sections without external data.

    Parameters
    ----------
    groups : int, optional
        Number of energy groups (default 1).

    Returns
    -------
    NuclideLibrary
        A valid, small depletion library.
    """
    names = ["fuel", "fuel2", "fp", "fp_stable"]
    m = len(names)
    g = groups
    idx = {name: i for i, name in enumerate(names)}

    half_life = np.array([np.inf, np.inf, 3.0e4, np.inf])  # fp decays, rest stable

    # fp -> fp_stable, full branching
    decay_from = np.array([idx["fp"]], dtype=np.int64)
    decay_to = np.array([idx["fp_stable"]], dtype=np.int64)
    decay_branch = np.array([1.0])

    # (n,gamma): fuel -> fuel2 captures; others none
    cap_xs = np.zeros((m, g))
    cap_xs[idx["fuel"]] = 5.0
    cap_product = np.full(m, -1, dtype=np.int64)
    cap_product[idx["fuel"]] = idx["fuel2"]
    reaction_xs = {"(n,gamma)": cap_xs}
    reaction_product = {"(n,gamma)": cap_product}

    # Fission: fuel fissions, yielding ~2 fp atoms per fission
    fission_xs = np.zeros((m, g))
    fission_xs[idx["fuel"]] = 10.0
    fy_parent = np.array([idx["fuel"]], dtype=np.int64)
    fy_product = np.array([idx["fp"]], dtype=np.int64)
    fy_yield = np.array([2.0])
    kappa = np.zeros(m)
    kappa[idx["fuel"]] = 3.2e-11  # ~200 MeV in joules

    # Transport coupling cross sections (barns)
    nu_fission = np.zeros((m, g))
    nu_fission[idx["fuel"]] = 2.4 * 10.0  # nu * sigma_f
    xs_total = np.zeros((m, g))
    xs_total[idx["fuel"]] = 20.0
    xs_total[idx["fuel2"]] = 8.0
    xs_total[idx["fp"]] = 4.0
    xs_total[idx["fp_stable"]] = 2.0
    xs_scatter = np.zeros((m, g, g))
    for i, sigma_s in enumerate([6.0, 6.0, 3.0, 1.5]):
        xs_scatter[i] = np.full((g, g), sigma_s / g)
    chi = np.full(g, 1.0 / g)

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
