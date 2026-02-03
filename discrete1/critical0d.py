"""Zero-dimensional criticality helpers.

This module provides small helpers used for 0D (spatially independent)
criticality calculations such as a simple power iteration driver.
"""

import numpy as np

import discrete1
from discrete1 import multi_group_0d as mg
from discrete1 import tools

count_kk = 100
change_kk = 1e-06


def power_iteration(angles, xs_total, xs_scatter, xs_fission, chi=None):
    """Run a simple power iteration for the 0D problem.

    Parameters
    ----------
    angles : int
        Number of angular ordinates to construct.
    xs_total, xs_scatter, xs_fission : numpy.ndarray
        Cross section arrays used by the multigroup solver.
    chi : numpy.ndarray, optional
        Fission Neutron Distribution. Must be included if xs_fission is nusigf.

    Returns
    -------
    tuple
        (flux, keff) where ``flux`` is the converged group flux and
        ``keff`` is the estimated multiplication factor.
    """

    angle_x, angle_w = discrete1.angular_x(angles)

    # Initialize and normalize flux
    flux_old = np.random.rand(xs_total.shape[0])
    keff = np.linalg.norm(flux_old)
    flux_old /= np.linalg.norm(keff)

    # Initialize power source
    source = np.zeros((1, xs_total.shape[0]))

    converged = False
    count = 0
    change = 0.0

    while not (converged):
        # Update power source term
        if chi is None:
            tools._fission_mat_source_0d(flux_old, xs_fission, source, keff)
        else:
            tools._fission_vec_source_0d(flux_old, chi, xs_fission, source, keff)

        # Solve for scalar flux
        flux = mg.source_iteration(
            flux_old, xs_total, xs_scatter, source, angle_x, angle_w
        )

        # Update keffective
        if chi is None:
            keff = tools._update_keff_mat_0d(flux, flux_old, xs_fission, keff)
        else:
            keff = tools._update_keff_vec_0d(flux, flux_old, chi, xs_fission, keff)

        # Normalize flux
        flux /= np.linalg.norm(flux)

        # Check for convergence
        change = np.linalg.norm((flux - flux_old) / flux)
        print(f"Count: {count:>2}\tKeff: {keff:.8f}", end="\r")
        converged = (change < change_kk) or (count >= count_kk)
        count += 1

        flux_old = flux.copy()

    print(f"\nConvergence: {change:2.6e}")
    return flux, keff
