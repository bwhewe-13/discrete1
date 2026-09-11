"""Preset depletion chains.

A :class:`~discrete1.nuclides.DepletionChain` is the flux-independent half of
the Bateman matrix: half-lives, decay branching, reaction products, fission
yields, energy per fission. It carries no cross sections, so one chain serves
every problem: pick the nuclides you track with
:meth:`~discrete1.nuclides.DepletionChain.subset`, then attach multigroup
microscopic data with :func:`~discrete1.nuclides.library_from_chain`::

    from discrete1.chains import load_chain

    chain = load_chain("lwr-actinides").subset(["U235", "U238", "Xe135"])

The presets are defined below as plain Python literals rather than packaged
``.npz`` data: a chain is small (tens of nuclides and couplings), and keeping
the nuclear data in the source means the values and their provenance are
reviewable in a diff. :func:`load_chain` still reads a ``.npz`` path for
chains you build yourself, and :func:`save_chain` writes one.

The flux-*dependent* half of the matrix (the ``sum_g sigma_g phi_g`` reaction
rates) is deliberately not preset: it is rebuilt from the live flux every
burnup step by :func:`discrete1.depletion.build_burnup_matrix`, which is what
makes the burnup transport-coupled.

Data provenance
---------------
Half-lives: ENDF/B-VIII.0 decay sublibrary / NNDC (Chart of Nuclides).
Thermal fission yields: England & Rider (ENDF/B-VI) as tabulated in standard
reactor-physics texts. Energy per fission: ~200-208 MeV per actinide.
"""

import numpy as np

from discrete1.nuclides import REACTIONS, DepletionChain, _channel_key

__all__ = ["available_chains", "load_chain", "save_chain"]

DAY = 86400.0  # s
YEAR = 3.1557e7  # s
HOUR = 3600.0  # s


########################################################################
# lwr-actinides
########################################################################

# The U/Np/Pu actinide line, the I-135/Xe-135 and Sm-149 poison chains, the
# boron burnable-poison channel, and light-water moderator backgrounds.
#
# Standard chain simplifications (each collapses a short-lived intermediate
# whose equilibrium is reached far inside a burnup step):
#   U-238  (n,gamma) -> Np-239   skips U-239   (23.5 min)
#   Np-237 (n,gamma) -> Pu-238   skips Np-238  (2.1 d)
#   Sm-149 fission yields are lumped *cumulative* chain yields (the
#          Nd-149 / Pm-149 precursors are not tracked)
#   I-135 -> Xe-135 lumps the Xe-135m branch

_LWR_NAMES = [
    # --- actinides ---
    "U234",
    "U235",
    "U236",
    "U237",
    "U238",
    "Np237",
    "Np239",
    "Pu238",
    "Pu239",
    "Pu240",
    "Pu241",
    "Pu242",
    "Am241",
    # --- fission-product poisons ---
    "I135",
    "Xe135",
    "Xe136",
    "Cs135",
    "Sm149",
    # --- burnable poison and moderator backgrounds ---
    "B10",
    "B11",
    "H1",
    "O16",
]

# Nuclides omitted here are treated as stable.
_LWR_HALF_LIFE = {
    "U234": 2.455e5 * YEAR,
    "U235": 7.04e8 * YEAR,
    "U236": 2.342e7 * YEAR,
    "U237": 6.75 * DAY,  # -> Np-237
    "U238": 4.468e9 * YEAR,
    "Np237": 2.144e6 * YEAR,
    "Np239": 2.356 * DAY,  # -> Pu-239
    "Pu238": 87.7 * YEAR,
    "Pu239": 2.411e4 * YEAR,
    "Pu240": 6.561e3 * YEAR,
    "Pu241": 14.329 * YEAR,  # -> Am-241
    "Pu242": 3.75e5 * YEAR,
    "Am241": 432.6 * YEAR,
    "I135": 6.57 * HOUR,  # -> Xe-135
    "Xe135": 9.14 * HOUR,  # -> Cs-135
    "Cs135": 2.3e6 * YEAR,
}

# Decay couplings into the tracked set, as (parent, daughter, branch). A
# nuclide with a half-life but no coupling here still loses atoms, its
# daughter is simply not tracked. The alpha decays of the long-lived
# actinides are negligible on burnup timescales and carry no coupling.
_LWR_DECAY = [
    ("U237", "Np237", 1.0),
    ("Np239", "Pu239", 1.0),
    ("Pu241", "Am241", 1.0),  # beta-, 99.998%; the alpha branch is 2.4e-5
    ("I135", "Xe135", 1.0),
    ("Xe135", "Cs135", 1.0),
]

# (n,gamma) products; anything not listed is -1 (channel absent, or the
# product is outside the tracked set).
_LWR_CAPTURE = {
    "U234": "U235",
    "U235": "U236",
    "U236": "U237",
    "U238": "Np239",  # via U-239, skipped
    "Np237": "Pu238",  # via Np-238, skipped
    "Pu238": "Pu239",
    "Pu239": "Pu240",
    "Pu240": "Pu241",
    "Pu241": "Pu242",
    "Xe135": "Xe136",
}

# (n,2n) channels that stay inside the tracked set.
_LWR_N2N = {
    "U236": "U235",
    "U238": "U237",
    "Pu240": "Pu239",
    "Pu241": "Pu240",
}

# Thermal fission yields (U-238: fast). I-135 and Sm-149 are cumulative chain
# yields; Xe-135 is the independent yield -- the rest of the Xe-135 arrives
# through the I-135 decay coupling above.
_LWR_YIELDS = {
    "U235": {"I135": 0.0639, "Xe135": 0.00237, "Sm149": 0.0113},
    "U238": {"I135": 0.0610, "Xe135": 0.0010, "Sm149": 0.0090},
    "Pu239": {"I135": 0.0654, "Xe135": 0.0110, "Sm149": 0.0125},
    "Pu241": {"I135": 0.0693, "Xe135": 0.0070, "Sm149": 0.0140},
}

# Energy per fission (J). Every actinide gets a value so power normalization
# also counts threshold fission in the minor actinides; 3.2e-11 J (~200 MeV)
# is the nominal stand-in where a measured value is not warranted.
_LWR_KAPPA = {
    "U234": 3.2e-11,
    "U235": 3.24e-11,  # ~202 MeV
    "U236": 3.2e-11,
    "U237": 3.2e-11,
    "U238": 3.35e-11,
    "Np237": 3.2e-11,
    "Np239": 3.2e-11,
    "Pu238": 3.2e-11,
    "Pu239": 3.33e-11,  # ~208 MeV
    "Pu240": 3.2e-11,
    "Pu241": 3.44e-11,
    "Pu242": 3.2e-11,
    "Am241": 3.2e-11,
}


def _build_chain(names, half_life, decay, products, yields, kappa):
    """Turn the literal tables above into a :class:`DepletionChain`.

    Parameters
    ----------
    names : list of str
        Tracked nuclides, in canonical index order.
    half_life : dict[str, float]
        Half-life in seconds; nuclides omitted are stable.
    decay : list of (str, str, float)
        ``(parent, daughter, branch)`` couplings into the tracked set.
    products : dict[str, dict[str, str]]
        Reaction channel -> ``{parent: product}``. Parents absent from a
        channel get ``-1``.
    yields : dict[str, dict[str, float]]
        Fissioning nuclide -> ``{fission product: atoms per fission}``.
    kappa : dict[str, float]
        Energy released per fission in joules; omitted nuclides get 0.

    Returns
    -------
    DepletionChain
        Validated chain.
    """
    idx = {name: i for i, name in enumerate(names)}
    m = len(names)

    reaction_product = {
        channel: np.full(m, -1, dtype=np.int64) for channel in REACTIONS
    }
    for channel, couplings in products.items():
        for parent, product in couplings.items():
            reaction_product[channel][idx[parent]] = idx[product]

    fy_parent, fy_product, fy_yield = [], [], []
    for parent, fission_products in yields.items():
        for product, value in fission_products.items():
            fy_parent.append(idx[parent])
            fy_product.append(idx[product])
            fy_yield.append(value)

    return DepletionChain(
        names=list(names),
        half_life=np.array([half_life.get(name, np.inf) for name in names]),
        decay_from=np.array([idx[parent] for parent, _, _ in decay], dtype=np.int64),
        decay_to=np.array([idx[child] for _, child, _ in decay], dtype=np.int64),
        decay_branch=np.array([branch for _, _, branch in decay]),
        reaction_product=reaction_product,
        fy_parent=np.array(fy_parent, dtype=np.int64),
        fy_product=np.array(fy_product, dtype=np.int64),
        fy_yield=np.array(fy_yield, dtype=np.float64),
        kappa=np.array([kappa.get(name, 0.0) for name in names]),
    )


def _lwr_actinides():
    """Build the ``lwr-actinides`` preset."""
    return _build_chain(
        _LWR_NAMES,
        _LWR_HALF_LIFE,
        _LWR_DECAY,
        # B-10 (n,alpha) -> Li-7 + He-4 is the burnable-poison burnout.
        # Neither product is tracked, so it is loss-only -- which is all it
        # needs to be; the -1 still charges B-10 for the absorption.
        {"(n,gamma)": _LWR_CAPTURE, "(n,2n)": _LWR_N2N},
        _LWR_YIELDS,
        _LWR_KAPPA,
    )


PRESETS = {"lwr-actinides": _lwr_actinides}


########################################################################
# Public API
########################################################################


def available_chains():
    """List the names of the built-in preset chains.

    Returns
    -------
    list of str
        Preset names accepted by :func:`load_chain`, sorted.
    """
    return sorted(PRESETS)


def load_chain(name_or_path):
    """Load a :class:`~discrete1.nuclides.DepletionChain`.

    Parameters
    ----------
    name_or_path : str
        Either a preset name from :func:`available_chains` (e.g.
        ``"lwr-actinides"``) or a path to a ``.npz`` archive written by
        :func:`save_chain`.

    Returns
    -------
    DepletionChain
        Validated chain. Presets are rebuilt on each call, so mutating the
        result cannot corrupt later loads.

    Raises
    ------
    ValueError
        If ``name_or_path`` is neither a known preset nor a ``.npz`` path.
    """
    name = str(name_or_path)
    if name in PRESETS:
        return PRESETS[name]()
    if not name.endswith(".npz"):
        raise ValueError(
            f"unknown chain {name!r}; available presets: "
            f"{', '.join(available_chains())} (or pass a path to a .npz archive)"
        )

    data = np.load(name)
    reaction_product = {}
    for channel in REACTIONS:
        key = f"rprod_{_channel_key(channel)}"
        if key in data.files:
            reaction_product[channel] = data[key]
    return DepletionChain(
        names=[str(item) for item in data["names"]],
        half_life=data["half_life"],
        decay_from=data["decay_from"],
        decay_to=data["decay_to"],
        decay_branch=data["decay_branch"],
        reaction_product=reaction_product,
        fy_parent=data["fy_parent"],
        fy_product=data["fy_product"],
        fy_yield=data["fy_yield"],
        kappa=data["kappa"],
    )


def save_chain(chain, path):
    """Write a :class:`~discrete1.nuclides.DepletionChain` to a ``.npz``."""
    arrays = {
        "names": np.array(chain.names),
        "half_life": chain.half_life,
        "decay_from": chain.decay_from,
        "decay_to": chain.decay_to,
        "decay_branch": chain.decay_branch,
        "fy_parent": chain.fy_parent,
        "fy_product": chain.fy_product,
        "fy_yield": chain.fy_yield,
        "kappa": chain.kappa,
    }
    for channel, product in chain.reaction_product.items():
        arrays[f"rprod_{_channel_key(channel)}"] = product
    np.savez(path, **arrays)
