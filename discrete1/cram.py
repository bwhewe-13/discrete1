"""Chebyshev Rational Approximation Method (CRAM) matrix exponential.

This module evaluates ``n(t + dt) = exp(A * dt) @ n0`` for the depletion
(Bateman) system ``dn/dt = A n`` using the Chebyshev Rational Approximation
Method. CRAM is well suited to burnup matrices because their eigenvalues
cluster near the negative real axis, where the Chebyshev rational
approximation of ``exp(x)`` is near-optimal. This makes it accurate and
stable for the stiff matrices that arise from mixing short- and long-lived
nuclides, without pruning the chain.

The approximation is evaluated through its incomplete-partial-fraction (IPF)
form, applying the half-set of conjugate poles ``theta_k`` sequentially::

    y <- n0
    for each (alpha_k, theta_k):
        y <- y + 2 * Re[ alpha_k * (A dt - theta_k I)^-1 y ]
    y <- alpha0 * y

where ``alpha_k`` are the residues and ``alpha0`` is a final multiplicative
factor. Each pole requires a single complex sparse linear solve. This is the
form used by OpenMC/Serpent; note the running vector ``y`` (not ``n0``) is
fed into each successive solve.

Two orders are provided, selectable via the ``order`` argument:

- ``order=16``: CRAM-16 (Pusa, 2010).
- ``order=48``: CRAM-48 incomplete partial fractions (Pusa, 2016).

The order-48 set is more accurate and robust for large time steps; order-16
is cheaper. Coefficient values are transcribed from the canonical Pusa
references as used by OpenMC/Serpent.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

__all__ = ["cram_expm", "VALID_ORDERS"]

VALID_ORDERS = (16, 48)

########################################################################
# CRAM coefficients (half-set of conjugate poles)
########################################################################

# CRAM-16 (Pusa, 2010)
_THETA16 = np.array(
    [
        +3.509103608414918 + 8.436198985884374j,
        +5.948152268951177 + 3.587457362018322j,
        -5.264971343442647 + 16.22022147316793j,
        +1.419375897185666 + 10.92536348449672j,
        +6.416177699099435 + 1.194122393370139j,
        +4.993174737717997 + 5.996881713603942j,
        -1.413928462488886 + 13.49772569889275j,
        -10.84391707869699 + 19.27744616718165j,
    ],
    dtype=np.complex128,
)

_ALPHA16 = np.array(
    [
        +5.464930576870210e3 - 3.797983575308356e4j,
        +9.045112476907548e1 - 1.115537522430261e3j,
        +2.344818070467641e2 - 4.228020157070496e2j,
        +9.453304067358312e1 - 2.951294291446048e2j,
        +7.283792954673409e2 - 1.205646080220011e5j,
        +3.648229059594851e1 - 1.155509621409682e2j,
        +2.547321630156819e1 - 2.639500283021502e1j,
        +2.394538338734709e1 - 5.650522971778156e0j,
    ],
    dtype=np.complex128,
)

_ALPHA0_16 = 2.124853710495224e-16

# CRAM-48 incomplete partial fractions (Pusa, 2016)
_THETA48 = np.array(
    [
        -4.465731934165702e1 + 6.233225190695437e1j,
        -5.284616241568964e0 + 4.057499381311059e1j,
        -8.867715667624458e0 + 4.325515754166724e1j,
        +3.493013124279215e0 + 3.281615453173585e1j,
        +1.564102508858634e1 + 1.558061616372237e1j,
        +1.742097597385893e1 + 1.076629305714420e1j,
        -2.834466755180654e1 + 5.492841024648724e1j,
        +1.661569367939544e1 + 1.316994930024688e1j,
        +8.011836167974721e0 + 2.780232111309410e1j,
        -2.056267541998229e0 + 3.794824788914354e1j,
        +1.449208170441839e1 + 1.799988210051809e1j,
        +1.853807176907916e1 + 5.974332563100539e0j,
        +9.932562704505182e0 + 2.532823409972962e1j,
        -2.244223871767187e1 + 5.179633600312162e1j,
        +8.590014121680897e-1 + 3.536456194294350e1j,
        -1.286192925744479e1 + 4.600304902833652e1j,
        +1.164596909542055e1 + 2.287153304140217e1j,
        +1.806076684783089e1 + 8.368200580099821e0j,
        +5.870672154659249e0 + 3.029700159040121e1j,
        -3.542938819659747e1 + 5.834381701800013e1j,
        +1.901323489060250e1 + 1.194282058271408e0j,
        +1.885508331552577e1 + 3.583428564427879e0j,
        -1.734689708174982e1 + 4.883941101108207e1j,
        +1.316284237125190e1 + 2.042951874827759e1j,
    ],
    dtype=np.complex128,
)

_ALPHA48 = np.array(
    [
        +6.387380733878774e2 - 6.743912502859256e2j,
        +1.909896179065730e2 - 3.973203432721332e2j,
        +4.236195226571914e2 - 2.041233768918671e3j,
        +4.645770595258726e2 - 1.652917287299683e3j,
        +7.765163276752433e2 - 1.783617639907328e4j,
        +1.907115136768522e3 - 5.887068595142284e4j,
        +2.909892685603256e3 - 9.953255345514560e3j,
        +1.944772206620450e2 - 1.427131226068449e3j,
        +1.382799786972332e5 - 3.256885197214938e6j,
        +5.628442079602433e3 - 2.924284515884309e4j,
        +2.151681283794220e2 - 1.121774011188224e3j,
        +1.324720240514420e3 - 6.370088443140973e4j,
        +1.617548476343347e4 - 1.008798413156542e6j,
        +1.112729040439685e2 - 8.837109731680418e1j,
        +1.074624783191125e2 - 1.457246116408180e2j,
        +8.835727765158191e1 - 6.388286188419360e1j,
        +9.354078136054179e1 - 2.195424319460237e2j,
        +9.418142823531573e1 - 6.719055740098035e2j,
        +1.040012390717851e2 - 1.693747595553868e2j,
        +6.861882624343235e1 - 1.177598523430493e1j,
        +8.766654491283722e1 - 4.596464999363902e3j,
        +1.056007619389650e2 - 1.738294585524067e3j,
        +7.738987569039419e1 - 4.311715386228984e1j,
        +1.041366366475571e2 - 2.777743732451969e2j,
    ],
    dtype=np.complex128,
)

_ALPHA0_48 = 2.258038182743983e-47

_COEFFS = {
    16: (_THETA16, _ALPHA16, _ALPHA0_16),
    48: (_THETA48, _ALPHA48, _ALPHA0_48),
}


########################################################################
# Solver
########################################################################


def cram_expm(matrix, n0, dt, order=48, substeps=1):
    """Evaluate ``exp(matrix * dt) @ n0`` with CRAM.

    Parameters
    ----------
    matrix : scipy.sparse matrix or numpy.ndarray, shape (M, M)
        Burnup (transmutation) matrix ``A`` with units of 1/s. Sparse
        input is recommended; dense input is converted internally.
    n0 : numpy.ndarray, shape (M,)
        Initial nuclide number densities.
    dt : float
        Time step in seconds.
    order : int, optional
        CRAM order, either 16 or 48 (default 48).
    substeps : int, optional
        Number of equal substeps over ``dt`` (default 1). Substeps reuse
        sparse LU factorizations and reduce error on stiff chains.

    Returns
    -------
    numpy.ndarray, shape (M,)
        Number densities after ``dt``, i.e. ``exp(A dt) n0``. The result
        is real; the tiny imaginary residue from the complex solves is
        discarded.

    Notes
    -----
    Uses the incomplete-partial-fraction (IPF) form: the half-set of
    conjugate poles is applied sequentially to the running vector, each
    pole solving the complex sparse system ``(A dt - theta_k I) y = y``
    via a sparse LU factorization, with a final multiply by ``alpha0``.
    """
    if order not in _COEFFS:
        raise ValueError(f"CRAM order must be one of {VALID_ORDERS}, got {order}")
    if substeps < 1:
        raise ValueError(f"substeps must be a positive integer, got {substeps}")

    theta, alpha, alpha0 = _COEFFS[order]

    # Scale by the (sub)step and work in CSC for efficient sparse LU solves
    step_dt = dt / substeps
    A = sp.csc_matrix(matrix, dtype=np.complex128) * step_dt
    identity = sp.eye(A.shape[0], format="csc", dtype=np.complex128)

    # Pre-factorize each shifted system once and reuse across substeps
    solvers = [spla.splu(A - theta_k * identity).solve for theta_k in theta]

    y = n0.astype(np.float64)
    for _ in range(substeps):
        for alpha_k, solve in zip(alpha, solvers):
            y = y + 2.0 * np.real(alpha_k * solve(y))
        y = alpha0 * y
    return y
