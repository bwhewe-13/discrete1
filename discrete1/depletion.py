"""Bateman burnup-matrix assembly and single-region depletion.

This module turns a :class:`~discrete1.nuclides.NuclideLibrary` and a region's
one-group neutron flux into the transmutation (burnup) matrix ``A`` of the
Bateman system ``dN/dt = A N``, then advances number densities over a time
step with CRAM (:func:`discrete1.cram.cram_expm`).

The burnup matrix has units of 1/s. Column ``i`` holds all the rates by which
nuclide ``i`` is lost or transmutes into other nuclides; row ``j`` collects all
production of nuclide ``j``::

    A[i, i] -= lambda_i + sum_r rate(i, r)        # decay + reaction losses
    A[t, i] += lambda_i * branch                  # decay production
    A[p, i] += rate(i, r)                         # reaction production (r -> p)
    A[p, i] += rate_fission(i) * yield(i -> p)    # fission-product production

Reaction rates collapse the multigroup microscopic cross sections against the
group flux: ``rate = sum_g sigma_g * phi_g``. Unit convention (consistent set):

- ``phi_g`` in n/cm^2/s,
- microscopic ``sigma_g`` stored in barns and converted with
  :data:`discrete1.constants.CM_TO_BARNS` (1 barn = 1e-24 cm^2),
- number densities in atoms/(barn*cm) (equivalently 1e24 atoms/cm^3),

which yields reaction rates and ``A`` entries in 1/s.
"""

import numpy as np
import scipy.sparse as sp

import discrete1.constants as const
from discrete1 import cram
from discrete1.nuclides import REACTIONS

__all__ = ["reaction_rate_1g", "build_burnup_matrix", "deplete"]


def reaction_rate_1g(xs_mg, flux_g):
    """Collapse multigroup microscopic cross sections to one-group rates.

    Computes the per-nuclide reaction rate ``sum_g sigma_g * phi_g`` (in 1/s)
    given barn-valued cross sections and a group flux in n/cm^2/s.

    Parameters
    ----------
    xs_mg : numpy.ndarray, shape (M, G)
        Microscopic multigroup cross section in barns.
    flux_g : numpy.ndarray, shape (G,)
        One-region scalar flux per energy group in n/cm^2/s.

    Returns
    -------
    numpy.ndarray, shape (M,)
        One-group reaction rate per nuclide in 1/s.
    """
    return (xs_mg @ flux_g) * const.CM_TO_BARNS


def build_burnup_matrix(library, flux_g, densities=None):
    """Assemble the sparse Bateman burnup matrix for one region.

    Parameters
    ----------
    library : NuclideLibrary
        Decay, reaction, and fission data.
    flux_g : numpy.ndarray, shape (G,)
        Region scalar flux per energy group in n/cm^2/s. Pass zeros for a
        pure decay/activation-free calculation.
    densities : numpy.ndarray, optional
        Unused for matrix assembly (the burnup matrix is independent of
        composition); accepted for API symmetry and future extensions.

    Returns
    -------
    scipy.sparse.csc_matrix, shape (M, M)
        Burnup matrix ``A`` (1/s) such that ``dN/dt = A N``.

    Notes
    -----
    The matrix is built in COO form (row, col, value triplets) and converted
    to CSC for the CRAM linear solves. Column sums of a correctly built matrix
    are zero for channels that conserve atoms (decay, capture) and can be
    positive for fission (multiple products per absorption).
    """
    m = library.n_nuclides
    rows = []
    cols = []
    vals = []

    # Diagonal losses accumulated per nuclide, then emitted as (i, i) entries.
    loss = np.zeros(m)

    # --- Radioactive decay ---
    lam = library.decay_constant  # (M,)
    loss += lam
    for parent, daughter, branch in zip(
        library.decay_from, library.decay_to, library.decay_branch
    ):
        rows.append(daughter)
        cols.append(parent)
        vals.append(lam[parent] * branch)

    # --- Neutron reactions (transmutation channels) ---
    for channel in REACTIONS:
        if channel not in library.reaction_xs:
            continue
        rate = reaction_rate_1g(library.reaction_xs[channel], flux_g)  # (M,)
        loss += rate
        product = library.reaction_product[channel]
        for parent in range(m):
            target = product[parent]
            if target >= 0 and rate[parent] != 0.0:
                rows.append(target)
                cols.append(parent)
                vals.append(rate[parent])

    # --- Fission (loss of fissile nuclide, production of fission products) ---
    fission_rate = reaction_rate_1g(library.fission_xs, flux_g)  # (M,)
    loss += fission_rate
    for parent, product, yld in zip(
        library.fy_parent, library.fy_product, library.fy_yield
    ):
        if fission_rate[parent] != 0.0:
            rows.append(product)
            cols.append(parent)
            vals.append(fission_rate[parent] * yld)

    # --- Diagonal loss terms ---
    for i in range(m):
        if loss[i] != 0.0:
            rows.append(i)
            cols.append(i)
            vals.append(-loss[i])

    matrix = sp.coo_matrix((vals, (rows, cols)), shape=(m, m), dtype=np.float64).tocsc()
    matrix.sum_duplicates()
    return matrix


def deplete(library, densities, flux_g, dt, order=48, substeps=1):
    """Advance number densities over ``dt`` for a single region.

    Builds the burnup matrix from ``flux_g`` and applies CRAM.

    Parameters
    ----------
    library : NuclideLibrary
        Depletion data.
    densities : numpy.ndarray, shape (M,)
        Initial number densities in atoms/(barn*cm).
    flux_g : numpy.ndarray, shape (G,)
        Region scalar flux per energy group in n/cm^2/s.
    dt : float
        Time step in seconds.
    order : int, optional
        CRAM order (16 or 48; default 48).
    substeps : int, optional
        CRAM substeps over ``dt`` (default 1).

    Returns
    -------
    numpy.ndarray, shape (M,)
        Number densities after ``dt``.
    """
    matrix = build_burnup_matrix(library, flux_g, densities)
    return cram.cram_expm(matrix, densities, dt, order=order, substeps=substeps)
