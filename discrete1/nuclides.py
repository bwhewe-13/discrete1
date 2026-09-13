"""Nuclide library schema for burnup / activation calculations.

This module defines the two data containers consumed by the depletion
machinery (:mod:`discrete1.depletion`, :mod:`discrete1.burnup1d`), split along
the same seam as the Bateman matrix itself:

- :class:`DepletionChain`, the *flux-independent* half: half-lives, decay
  branching, reaction products, fission yields, energy per fission. Constant
  for all time and reusable across problems; preset chains ship as packaged
  data (see :mod:`discrete1.chains`).
- :class:`NuclideLibrary`, a chain plus the microscopic multigroup cross
  sections that make the other half of the matrix flux-dependent. Build one
  from a chain with :func:`library_from_chain`.

:class:`NuclideLibrary` collects, for a fixed set of ``M`` tracked nuclides on
a ``G``-group energy grid:

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

Both containers support ``subset(names)``, which restricts the data to a
chosen inventory and renumbers every coupling accordingly (the way a problem
picks its nuclides out of a large master library).
"""

import math
from dataclasses import dataclass, field

import numpy as np

# Canonical neutron reaction channels that transmute one nuclide into another.
# Fission is handled separately because it produces many products via yields.
REACTIONS = ("(n,gamma)", "(n,2n)", "(n,3n)", "(n,p)", "(n,a)")

LN2 = math.log(2.0)


def _check_indices(indices, name, n_nuclides, lower=0):
    """Verify an index array stays within ``[lower, M)``."""
    if indices.size and (indices.min() < lower or indices.max() >= n_nuclides):
        raise ValueError(f"{name} indices must be in [{lower}, M)")


def _validate_topology(chain):
    """Check the flux-independent data of a chain or library; raise on error.

    Shared by :meth:`DepletionChain.validate` and
    :meth:`NuclideLibrary.validate`; the latter adds the cross-section shape
    checks on top.
    """
    m = chain.n_nuclides
    if len(set(chain.names)) != m:
        raise ValueError("names must be unique")
    if chain.half_life.shape != (m,):
        raise ValueError("half_life must have shape (M,)")
    if not np.all(chain.half_life > 0.0):
        raise ValueError("half_life must be positive (numpy.inf if stable)")
    if chain.kappa.shape != (m,):
        raise ValueError("kappa must have shape (M,)")
    for channel, product in chain.reaction_product.items():
        if product.shape != (m,):
            raise ValueError(f"reaction_product[{channel}] must be (M,)")
        # -1 flags an untracked product; anything else must index names
        _check_indices(product, f"reaction_product[{channel}]", m, lower=-1)
    for arr in (chain.decay_to, chain.decay_branch):
        if arr.shape != chain.decay_from.shape:
            raise ValueError("decay COO arrays misaligned")
    for arr in (chain.fy_product, chain.fy_yield):
        if arr.shape != chain.fy_parent.shape:
            raise ValueError("fission-yield COO arrays misaligned")
    _check_indices(chain.decay_from, "decay_from", m)
    _check_indices(chain.decay_to, "decay_to", m, lower=-1)
    _check_indices(chain.fy_parent, "fy_parent", m)
    _check_indices(chain.fy_product, "fy_product", m, lower=-1)
    if np.any(chain.decay_branch < 0.0):
        raise ValueError("decay_branch must be non-negative")
    if np.any(chain.fy_yield < 0.0):
        raise ValueError("fy_yield must be non-negative")


def _selection(index, names):
    """Resolve a name selection against ``index`` into keep/renumber arrays.

    ``keep`` gives the old indices in the new order; ``mapping`` is old index
    -> new index with -1 for anything dropped, reusing the same sentinel the
    rest of the schema already uses for an untracked product.
    """
    names = list(names)
    if len(set(names)) != len(names):
        raise ValueError("subset names must be unique")
    keep = np.empty(len(names), dtype=np.int64)
    mapping = np.full(len(index), -1, dtype=np.int64)
    for new, name in enumerate(names):
        if name not in index:
            raise ValueError(f"unknown nuclide {name!r}")
        keep[new] = index[name]
        mapping[index[name]] = new
    return keep, mapping


def _remap_products(product, keep, mapping):
    """Row-select a per-nuclide product array and renumber its values."""
    out = np.asarray(product)[keep].astype(np.int64)
    tracked = out >= 0
    out[tracked] = mapping[out[tracked]]
    return out


def _remap_coo(from_index, to_index, values, mapping):
    """Restrict a COO coupling to the selected parents and renumber it.

    Couplings whose *parent* is dropped disappear entirely. Couplings whose
    *product* is dropped are kept with a ``-1`` product: the parent still
    loses atoms down that channel, it just produces nothing tracked.
    """
    parents = mapping[from_index]
    keep = parents >= 0
    products = to_index[keep]
    # np.maximum guards the gather; -1 entries stay -1 via the where().
    products = np.where(products >= 0, mapping[np.maximum(products, 0)], -1)
    return parents[keep], products.astype(np.int64), np.asarray(values)[keep].copy()


def _subset_topology(chain, keep, mapping):
    """Remap every flux-independent array onto a nuclide selection."""
    decay_from, decay_to, decay_branch = _remap_coo(
        chain.decay_from, chain.decay_to, chain.decay_branch, mapping
    )
    fy_parent, fy_product, fy_yield = _remap_coo(
        chain.fy_parent, chain.fy_product, chain.fy_yield, mapping
    )
    return {
        "half_life": chain.half_life[keep].copy(),
        "decay_from": decay_from,
        "decay_to": decay_to,
        "decay_branch": decay_branch,
        "reaction_product": {
            channel: _remap_products(product, keep, mapping)
            for channel, product in chain.reaction_product.items()
        },
        "fy_parent": fy_parent,
        "fy_product": fy_product,
        "fy_yield": fy_yield,
        "kappa": chain.kappa[keep].copy(),
    }


@dataclass
class DepletionChain:
    """Flux-independent depletion topology: decay, products, and yields.

    This is the half of the Bateman matrix that never changes: half-lives,
    branching, which nuclide each reaction channel produces, fission yields,
    and energy per fission. The other half (the ``sum_g sigma_g phi_g``
    reaction rates) depends on the flux and is rebuilt every burnup step by
    :func:`discrete1.depletion.build_burnup_matrix`, so it is deliberately
    *not* stored here.

    Because a chain carries no cross sections it is problem-independent:
    pair one with microscopic multigroup data via :func:`library_from_chain`
    to get the :class:`NuclideLibrary` the solvers consume. Preset chains
    ship as packaged data, see :func:`discrete1.chains.load_chain`.

    Attributes
    ----------
    names : list of str
        Nuclide identifiers, length ``M``. Index order is canonical.
    half_life : numpy.ndarray, shape (M,)
        Half-lives in seconds (``numpy.inf`` for stable).
    decay_from, decay_to : numpy.ndarray of int
        COO decay couplings; ``-1`` marks an untracked daughter.
    decay_branch : numpy.ndarray, shape (n_decay,)
        Branching ratio per decay coupling.
    reaction_product : dict[str, numpy.ndarray]
        Channel in :data:`REACTIONS` -> product index per parent, shape
        ``(M,)``, ``-1`` where absent or untracked.
    fy_parent, fy_product : numpy.ndarray of int
        COO fission-yield couplings; ``-1`` marks an untracked product.
    fy_yield : numpy.ndarray, shape (n_fy,)
        Atoms of ``fy_product`` per fission of ``fy_parent``.
    kappa : numpy.ndarray, shape (M,)
        Energy released per fission in joules.
    """

    names: list[str]
    half_life: np.ndarray
    decay_from: np.ndarray
    decay_to: np.ndarray
    decay_branch: np.ndarray
    reaction_product: dict[str, np.ndarray]
    fy_parent: np.ndarray
    fy_product: np.ndarray
    fy_yield: np.ndarray
    kappa: np.ndarray
    index: dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        """Build the name -> index lookup and validate the topology."""
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
        return LN2 / self.half_life

    def validate(self):
        """Check shape, index, and value consistency; raise on error.

        Raises
        ------
        ValueError
            If any array shape, nuclide index, or physical value (half-life,
            branching ratio, fission yield) is inconsistent or invalid.
        """
        _validate_topology(self)

    def subset(self, names):
        """Restrict the chain to ``names``, renumbering every coupling.

        See :meth:`NuclideLibrary.subset` for the truncation semantics; in
        particular, dropping a product keeps the parent's loss term.

        Parameters
        ----------
        names : sequence of str
            Nuclides to keep, in the order they should be indexed.

        Returns
        -------
        DepletionChain
            A new chain over ``names`` only.

        Raises
        ------
        ValueError
            If ``names`` repeats a nuclide or names one absent from the chain.
        """
        keep, mapping = _selection(self.index, names)
        return DepletionChain(
            names=list(names), **_subset_topology(self, keep, mapping)
        )


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
        daughter ``decay_to[k]`` (``-1`` if the daughter is not tracked;
        the parent still loses atoms).
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
        ``fy_parent[k]`` produces ``fy_product[k]`` (``-1`` if the product
        is not tracked).
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
    chi : numpy.ndarray, shape (G,) or (G, G)
        Fission neutron emission spectrum. If 1D, one spectrum shared across
        all incident (fissioning) groups (sums to 1). If 2D, indexed
        ``[g_in, g_out]`` -- the outgoing-group spectrum as a function of the
        incident group ``g_in`` that induced fission (each row sums to 1).
    """

    names: list[str]
    groups: int
    half_life: np.ndarray
    decay_from: np.ndarray
    decay_to: np.ndarray
    decay_branch: np.ndarray
    reaction_xs: dict[str, np.ndarray]
    reaction_product: dict[str, np.ndarray]
    fy_parent: np.ndarray
    fy_product: np.ndarray
    fy_yield: np.ndarray
    fission_xs: np.ndarray
    kappa: np.ndarray
    xs_total: np.ndarray
    xs_scatter: np.ndarray
    nu_fission: np.ndarray
    chi: np.ndarray
    index: dict[str, int] = field(default_factory=dict)

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
        """Decay constants ``lambda = ln(2) / half_life`` (1/s), 0 if stable.

        :meth:`validate` guarantees positive half-lives, so stable nuclides
        (``half_life = inf``) map exactly to zero.
        """
        return LN2 / self.half_life

    def validate(self):
        """Check internal shape, index, and value consistency; raise on error.

        Raises
        ------
        ValueError
            If any array shape, nuclide index, or physical value (half-life,
            branching ratio, fission yield) is inconsistent or invalid.
        """
        _validate_topology(self)
        m, g = self.n_nuclides, self.groups
        for arr, name in (
            (self.fission_xs, "fission_xs"),
            (self.nu_fission, "nu_fission"),
            (self.xs_total, "xs_total"),
        ):
            if arr.shape != (m, g):
                raise ValueError(f"{name} must have shape (M, G)")
        if self.xs_scatter.shape != (m, g, g):
            raise ValueError("xs_scatter must be (M, G, G)")
        if self.chi.shape not in ((g,), (g, g)):
            raise ValueError("chi must have shape (G,) or (G, G)")
        for channel in REACTIONS:
            if channel in self.reaction_xs:
                if self.reaction_xs[channel].shape != (m, g):
                    raise ValueError(f"reaction_xs[{channel}] must have shape (M, G)")
                if channel not in self.reaction_product:
                    raise ValueError(f"reaction_product[{channel}] missing")

    def chain(self):
        """Extract the flux-independent :class:`DepletionChain` half."""
        return DepletionChain(
            names=list(self.names),
            half_life=self.half_life.copy(),
            decay_from=self.decay_from.copy(),
            decay_to=self.decay_to.copy(),
            decay_branch=self.decay_branch.copy(),
            reaction_product={
                channel: product.copy()
                for channel, product in self.reaction_product.items()
            },
            fy_parent=self.fy_parent.copy(),
            fy_product=self.fy_product.copy(),
            fy_yield=self.fy_yield.copy(),
            kappa=self.kappa.copy(),
        )

    def subset(self, names):
        """Restrict the library to ``names``, renumbering every coupling.

        This is how a problem selects its inventory out of a large master
        library: cross sections are row-selected and every decay, reaction,
        and fission-yield coupling is renumbered onto the new indices.

        Dropping a nuclide that other nuclides produce does *not* remove the
        parent's loss of atoms down that channel: the coupling is kept with a
        ``-1`` product, so :func:`discrete1.depletion.build_burnup_matrix`
        still charges the parent for the absorption/decay but creates
        nothing. That's the physically correct truncation, but it also means
        a selection can silently change answers by cutting a production path
        (dropping Xe-135 while keeping I-135 discards the poison, not the
        I-135 decay), so pick the inventory with that in mind.

        Parameters
        ----------
        names : sequence of str
            Nuclides to keep, in the order they should be indexed. May
            reorder as well as restrict.

        Returns
        -------
        NuclideLibrary
            A new library over ``names`` only, sharing no arrays with ``self``.

        Raises
        ------
        ValueError
            If ``names`` repeats a nuclide or names one absent from the
            library.
        """
        keep, mapping = _selection(self.index, names)
        return NuclideLibrary(
            names=list(names),
            groups=self.groups,
            reaction_xs={
                channel: xs[keep].copy() for channel, xs in self.reaction_xs.items()
            },
            fission_xs=self.fission_xs[keep].copy(),
            xs_total=self.xs_total[keep].copy(),
            xs_scatter=self.xs_scatter[keep].copy(),
            nu_fission=self.nu_fission[keep].copy(),
            chi=self.chi.copy(),
            **_subset_topology(self, keep, mapping),
        )


def library_from_chain(
    chain,
    groups,
    fission_xs,
    xs_total,
    xs_scatter,
    nu_fission,
    chi,
    reaction_xs=None,
):
    """Pair a :class:`DepletionChain` with microscopic cross sections.

    The chain supplies the flux-independent topology; the cross-section
    arguments supply the multigroup data that makes reaction rates
    flux-dependent. Every array must be ordered to match ``chain.names``.

    Parameters
    ----------
    chain : DepletionChain
        Depletion topology. Restrict it with :meth:`DepletionChain.subset`
        first if the cross sections cover only part of the chain.
    groups : int
        Number of energy groups ``G``.
    fission_xs, xs_total, nu_fission : numpy.ndarray, shape (M, G)
        Microscopic fission, total, and ``nu * sigma_f`` cross sections (barns).
    xs_scatter : numpy.ndarray, shape (M, G, G)
        Microscopic scattering matrix (barns).
    chi : numpy.ndarray, shape (G,) or (G, G)
        Fission emission spectrum. See :class:`NuclideLibrary` for the (G, G)
        ``[g_in, g_out]`` convention.
    reaction_xs : dict[str, numpy.ndarray], optional
        Channel -> microscopic multigroup cross section, shape ``(M, G)``.
        Each channel present must also appear in ``chain.reaction_product``.

    Returns
    -------
    NuclideLibrary
        Validated library, copying the chain's arrays so the two stay
        independent.

    Raises
    ------
    ValueError
        If any array shape disagrees with ``chain.names`` / ``groups``, or a
        channel in ``reaction_xs`` has no product mapping on the chain.
    """
    return NuclideLibrary(
        names=list(chain.names),
        groups=groups,
        half_life=chain.half_life.copy(),
        decay_from=chain.decay_from.copy(),
        decay_to=chain.decay_to.copy(),
        decay_branch=chain.decay_branch.copy(),
        reaction_xs={} if reaction_xs is None else dict(reaction_xs),
        reaction_product={
            channel: product.copy()
            for channel, product in chain.reaction_product.items()
        },
        fy_parent=chain.fy_parent.copy(),
        fy_product=chain.fy_product.copy(),
        fy_yield=chain.fy_yield.copy(),
        fission_xs=fission_xs,
        kappa=chain.kappa.copy(),
        xs_total=xs_total,
        xs_scatter=xs_scatter,
        nu_fission=nu_fission,
        chi=chi,
    )


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
    data = np.load(path)
    names = [str(name) for name in data["names"]]
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


SYNTHETIC_NAMES = ["fuel", "fuel2", "fp", "fp_stable"]


def synthetic_chain():
    """Build the topology of the small self-contained test system.

    Four nuclides: ``fuel`` (a fissile, capturing nuclide) captures a neutron
    to ``fuel2`` and fissions into a ``fp`` (fission product) that beta-decays
    to a stable ``fp_stable``. This exercises decay, ``(n,gamma)`` capture,
    and fission yields without external data.

    Returns
    -------
    DepletionChain
        A valid, small depletion chain.
    """
    names = list(SYNTHETIC_NAMES)
    m = len(names)
    idx = {name: i for i, name in enumerate(names)}

    half_life = np.array([np.inf, np.inf, 3.0e4, np.inf])  # fp decays, rest stable

    # fp -> fp_stable, full branching
    decay_from = np.array([idx["fp"]], dtype=np.int64)
    decay_to = np.array([idx["fp_stable"]], dtype=np.int64)
    decay_branch = np.array([1.0])

    # (n,gamma): fuel -> fuel2 captures; others none
    cap_product = np.full(m, -1, dtype=np.int64)
    cap_product[idx["fuel"]] = idx["fuel2"]

    # Fission: fuel fissions, yielding ~2 fp atoms per fission
    fy_parent = np.array([idx["fuel"]], dtype=np.int64)
    fy_product = np.array([idx["fp"]], dtype=np.int64)
    fy_yield = np.array([2.0])
    kappa = np.zeros(m)
    kappa[idx["fuel"]] = 3.2e-11  # ~200 MeV in joules

    return DepletionChain(
        names=names,
        half_life=half_life,
        decay_from=decay_from,
        decay_to=decay_to,
        decay_branch=decay_branch,
        reaction_product={"(n,gamma)": cap_product},
        fy_parent=fy_parent,
        fy_product=fy_product,
        fy_yield=fy_yield,
        kappa=kappa,
    )


def synthetic_library(groups=1):
    """Build a small, self-contained library for tests and examples.

    Attaches flat multigroup cross sections to :func:`synthetic_chain`, so the
    result also exercises the transport-coupling arrays.

    Parameters
    ----------
    groups : int, optional
        Number of energy groups (default 1).

    Returns
    -------
    NuclideLibrary
        A valid, small depletion library.
    """
    chain = synthetic_chain()
    m, g = chain.n_nuclides, groups
    idx = chain.index

    cap_xs = np.zeros((m, g))
    cap_xs[idx["fuel"]] = 5.0

    fission_xs = np.zeros((m, g))
    fission_xs[idx["fuel"]] = 10.0

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

    return library_from_chain(
        chain,
        g,
        fission_xs,
        xs_total,
        xs_scatter,
        nu_fission,
        chi,
        reaction_xs={"(n,gamma)": cap_xs},
    )
