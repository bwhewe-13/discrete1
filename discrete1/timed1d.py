"""Time-dependent 1D transport problems.

This module provides solvers for time-dependent (transient) 1D transport
problems using discrete ordinates methods. It implements Backward Euler (BDF1)
and second-order Backward Differentiation Formula (BDF2) time integration schemes.
"""

import numpy as np
from tqdm import tqdm

from discrete1 import multi_group as mg
from discrete1 import tools


def backward_euler(
    flux_last,
    xs_total,
    xs_scatter,
    xs_fission,
    velocity,
    external,
    boundary,
    medium_map,
    delta_x,
    angle_x,
    angle_w,
    bc_x,
    steps,
    dt,
    geometry=1,
):
    """Solve time-dependent transport using first-order Backward Differentiation.

    Advances solution through multiple time steps using implicit BDF1 (Backward Euler)
    time integration with source iteration for the spatial problem.

    Parameters
    ----------
    flux_last : numpy.ndarray
        Angular flux at previous time step (cells, angles, groups).
    xs_total : numpy.ndarray
        Total macroscopic cross section (cells, groups).
    xs_scatter : numpy.ndarray
        Scattering cross section (cells, groups).
    xs_fission : numpy.ndarray
        Fission cross section (cells, groups).
    velocity : numpy.ndarray
        Particle velocity (groups,).
    external : numpy.ndarray
        External source (steps/1, cells, angles, groups).
    boundary : numpy.ndarray
        Boundary conditions (steps/1, 2, angles, groups).
    medium_map : numpy.ndarray
        Material region map (cells,).
    delta_x : numpy.ndarray
        Cell widths (cells,).
    angle_x : numpy.ndarray
        Discrete ordinates (angles,).
    angle_w : numpy.ndarray
        Quadrature weights (angles,).
    bc_x : int
        Boundary condition type (0=vacuum, 1=reflective).
    steps : int
        Number of time steps.
    dt : float
        Time step size.
    geometry : int, optional
        Geometry type (1=slab, 2=cylindrical, 3=spherical).

    Returns
    -------
    numpy.ndarray
        Scalar flux at all time steps (steps, cells, groups).
    """

    # Scalar flux approximation
    flux_old = np.sum(flux_last * angle_w[None, :, None], axis=1)

    # Scalar flux for every time step
    flux_time = np.zeros((steps,) + flux_old.shape)

    # Combine scattering and fission
    xs_matrix = xs_scatter + xs_fission

    # Create xs_total_star
    xs_total_v = xs_total + 1 / (velocity * dt)

    # Iterate over time steps
    for step in tqdm(range(steps), desc="BDF1 ", ascii=True):
        # Determine dimensions of external and boundary sources
        qq = 0 if external.shape[0] == 1 else step
        bb = 0 if boundary.shape[0] == 1 else step

        # Update q_star
        q_star = external[qq] + 1 / (velocity * dt) * flux_last

        # Run source iteration for scalar flux centers
        flux_time[step] = mg.source_iteration(
            flux_old,
            xs_total_v,
            xs_matrix,
            q_star,
            boundary[bb],
            medium_map,
            delta_x,
            angle_x,
            angle_w,
            bc_x,
            geometry,
        )

        # Create (sigma_s + sigma_f) * phi^{\ell} + Q*
        flux_old = flux_time[step].copy()
        tools._time_right_side(q_star, flux_old, xs_matrix, medium_map)

        # Solve for angular flux
        flux_last = mg.known_source_angular(
            xs_total_v,
            q_star,
            boundary[bb],
            medium_map,
            delta_x,
            angle_x,
            angle_w,
            bc_x,
            geometry,
            edges=0,
        )

    return flux_time


def bdf2(
    flux_last_1,
    xs_total,
    xs_scatter,
    xs_fission,
    velocity,
    external,
    boundary,
    medium_map,
    delta_x,
    angle_x,
    angle_w,
    bc_x,
    steps,
    dt,
    geometry=1,
):
    """Solve time-dependent transport using second-order Backward Differentiation.

    Advances solution through multiple time steps using implicit BDF2 time integration,
    starting with BDF1 on the first step and switching to BDF2 thereafter.

    Parameters
    ----------
    flux_last_1 : numpy.ndarray
        Angular flux at previous time step (cells, angles, groups).
    xs_total : numpy.ndarray
        Total macroscopic cross section (cells, groups).
    xs_scatter : numpy.ndarray
        Scattering cross section (cells, groups).
    xs_fission : numpy.ndarray
        Fission cross section (cells, groups).
    velocity : numpy.ndarray
        Particle velocity (groups,).
    external : numpy.ndarray
        External source (steps/1, cells, angles, groups).
    boundary : numpy.ndarray
        Boundary conditions (steps/1, 2, angles, groups).
    medium_map : numpy.ndarray
        Material region map (cells,).
    delta_x : numpy.ndarray
        Cell widths (cells,).
    angle_x : numpy.ndarray
        Discrete ordinates (angles,).
    angle_w : numpy.ndarray
        Quadrature weights (angles,).
    bc_x : int
        Boundary condition type (0=vacuum, 1=reflective).
    steps : int
        Number of time steps.
    dt : float
        Time step size.
    geometry : int, optional
        Geometry type (1=slab, 2=cylindrical, 3=spherical).

    Returns
    -------
    numpy.ndarray
        Scalar flux at all time steps (steps, cells, groups).
    """
    # Scalar flux approximation
    flux_old = np.sum(flux_last_1 * angle_w[None, :, None], axis=1)

    # Angular flux of step \ell - 2
    flux_last_2 = np.zeros(flux_last_1.shape)

    # Scalar flux for every time step
    flux_time = np.zeros((steps,) + flux_old.shape)

    # Combine scattering and fission
    xs_matrix = xs_scatter + xs_fission

    # Create xs_total_star (for BDF1 step)
    xs_total_v = xs_total + 1 / (velocity * dt)

    # Iterate over time steps
    for step in tqdm(range(steps), desc="BDF2 ", ascii=True):
        # Determine dimensions of external and boundary sources
        qq = 0 if external.shape[0] == 1 else step
        bb = 0 if boundary.shape[0] == 1 else step

        # BDF1 on first time step
        if step == 0:
            # Update q_star
            q_star = external[qq] + 1 / (velocity * dt) * flux_last_1

        else:
            q_star = (
                external[qq]
                + 2 / (velocity * dt) * flux_last_1
                - 1 / (2 * velocity * dt) * flux_last_2
            )

        # Run source iteration for scalar flux centers
        flux_time[step] = mg.source_iteration(
            flux_old,
            xs_total_v,
            xs_matrix,
            q_star,
            boundary[bb],
            medium_map,
            delta_x,
            angle_x,
            angle_w,
            bc_x,
            geometry,
        )

        # Create (sigma_s + sigma_f) * phi^{\ell} + Q*
        flux_old = flux_time[step].copy()
        tools._time_right_side(q_star, flux_old, xs_matrix, medium_map)

        # Solve for angular flux
        flux_last_2 = flux_last_1.copy()
        flux_last_1 = mg.known_source_angular(
            xs_total_v,
            q_star,
            boundary[bb],
            medium_map,
            delta_x,
            angle_x,
            angle_w,
            bc_x,
            geometry,
            edges=0,
        )

        # Update xs_totat_star (for BDF2 steps)
        if step == 0:
            xs_total_v = xs_total + 1.5 / (velocity * dt)

    return flux_time
