# Called for time dependent source problems
import numpy as np
from tqdm import tqdm

import discrete1
from discrete1 import multi_group as mg
from discrete1 import tools
from discrete1.utils import hybrid as hytools


def backward_euler(
    groups_c,
    angles_c,
    flux_last,
    xs_total_u,
    xs_scatter_u,
    xs_fission_u,
    velocity_u,
    external_u,
    boundary_u,
    medium_map,
    delta_x,
    angle_xu,
    angle_wu,
    bc_x,
    steps,
    dt,
    geometry=1,
    energy_grid=87,
):

    # Scalar flux approximation
    flux_u = np.sum(flux_last * angle_wu[None, :, None], axis=1)

    # Initialize time step flux, collided boundary
    flux_time = np.zeros((steps,) + flux_u.shape)
    boundary_c = np.zeros((2, 1, 1))

    # Combine scattering and fission
    xs_matrix_u = xs_scatter_u + xs_fission_u

    # Create star coef
    star_coef_u = 1 / (velocity_u * dt)

    # Check if collided energy groups don't change with time
    if isinstance(groups_c, int):
        # Calculate new grid
        coarse_idx, factor, edges_gidx_c, edges_g = hytools.energy_grid_change(
            energy_grid, flux_u.shape[1], groups_c
        )

        # Initialize collided flux and source
        flux_c = np.zeros((medium_map.shape[0], groups_c))
        source_c = np.zeros((medium_map.shape[0], 1, groups_c))

        # Calculate star_coef
        star_coef_c = 1 / (hytools.coarsen_velocity(velocity_u, edges_gidx_c) * dt)

    # Check if discrete ordinates don't change with time
    if isinstance(angles_c, int):
        angle_c = angles_c

    # Iterate over time steps
    for step in tqdm(range(steps), desc="vBDF1*", ascii=True):
        # Determine dimensions of external and boundary sources
        qq = 0 if external_u.shape[0] == 1 else step
        bb = 0 if boundary_u.shape[0] == 1 else step

        # Check if the number of collided energy groups change with time
        if isinstance(groups_c, np.ndarray):
            # Calculate new grid
            coarse_idx, factor, edges_gidx_c, edges_g = hytools.energy_grid_change(
                energy_grid, flux_u.shape[1], groups_c[step]
            )

            # Initialize collided flux and source
            flux_c = np.zeros((medium_map.shape[0], groups_c[step]))
            source_c = np.zeros((medium_map.shape[0], 1, groups_c[step]))

            # Calculate star_coef
            star_coef_c = 1 / (hytools.coarsen_velocity(velocity_u, edges_gidx_c) * dt)

        if isinstance(angles_c, np.ndarray):
            angle_c = angles_c[step]

        # Update q_star
        q_star = external_u[qq] + star_coef_u * flux_last

        # Run hybrid method
        _variable_hybrid_method(
            angle_c,
            flux_u,
            flux_c,
            xs_total_u,
            star_coef_u,
            star_coef_c,
            xs_matrix_u,
            q_star,
            source_c,
            boundary_u[bb],
            boundary_c,
            medium_map,
            delta_x,
            angle_xu,
            angle_wu,
            bc_x,
            coarse_idx,
            factor,
            edges_gidx_c,
            edges_g,
            geometry,
        )

        # Solve for angular flux
        flux_last = mg.known_source_angular(
            xs_total_u + star_coef_u,
            q_star,
            boundary_u[bb],
            medium_map,
            delta_x,
            angle_xu,
            angle_wu,
            bc_x,
            geometry,
            edges=0,
        )

        # Step 5: Update and repeat
        flux_time[step] = np.sum(flux_last * angle_wu[None, :, None], axis=1)

    return flux_time


def bdf2(
    groups_c,
    angles_c,
    flux_last_1,
    xs_total_u,
    xs_scatter_u,
    xs_fission_u,
    velocity_u,
    external_u,
    boundary_u,
    medium_map,
    delta_x,
    angle_xu,
    angle_wu,
    bc_x,
    steps,
    dt,
    geometry=1,
    energy_grid=87,
):

    # Scalar flux approximation
    flux_u = np.sum(flux_last_1 * angle_wu[None, :, None], axis=1)

    # Initialize time step flux, angular flux of step \ell - 2, collided boundary
    flux_time = np.zeros((steps,) + flux_u.shape)
    flux_last_2 = np.zeros(flux_last_1.shape)
    boundary_c = np.zeros((2, 1, 1))

    # Combine scattering and fission
    xs_matrix_u = xs_scatter_u + xs_fission_u

    # Create xs_total_star
    star_coef_u = 1 / (velocity_u * dt)

    # Check if collided energy groups don't change with time
    if isinstance(groups_c, int):
        # Calculate new grid
        coarse_idx, factor, edges_gidx_c, edges_g = hytools.energy_grid_change(
            energy_grid, flux_u.shape[1], groups_c
        )

        # Initialize collided flux and source
        flux_c = np.zeros((medium_map.shape[0], groups_c))
        source_c = np.zeros((medium_map.shape[0], 1, groups_c))

        # Calculate star_coef
        star_coef_c = 1 / (hytools.coarsen_velocity(velocity_u, edges_gidx_c) * dt)

    # Check if discrete ordinates don't change with time
    if isinstance(angles_c, int):
        angle_c = angles_c

    # Iterate over time steps
    for step in tqdm(range(steps), desc="vBDF2*", ascii=True):
        # Determine dimensions of external and boundary sources
        qq = 0 if external_u.shape[0] == 1 else step
        bb = 0 if boundary_u.shape[0] == 1 else step

        # Check if the number of collided energy groups change with time
        if isinstance(groups_c, np.ndarray):
            # Calculate new grid
            coarse_idx, factor, edges_gidx_c, edges_g = hytools.energy_grid_change(
                energy_grid, flux_u.shape[1], groups_c[step]
            )

            # Initialize collided flux and source
            flux_c = np.zeros((medium_map.shape[0], groups_c[step]))
            source_c = np.zeros((medium_map.shape[0], 1, groups_c[step]))

            # Calculate star_coef
            numerator = 1.0 if step == 0 else 1.5
            star_coef_c = numerator / (
                hytools.coarsen_velocity(velocity_u, edges_gidx_c) * dt
            )

        if isinstance(angles_c, np.ndarray):
            angle_c = angles_c[step]

        # BDF1 on first time step
        if step == 0:
            # Update q_star
            q_star = external_u[qq] + star_coef_u * flux_last_1

        else:
            q_star = (
                external_u[qq]
                + 4 / 3.0 * star_coef_u * flux_last_1
                - 1 / 3.0 * star_coef_u * flux_last_2
            )

        # Run hybrid method
        _variable_hybrid_method(
            angle_c,
            flux_u,
            flux_c,
            xs_total_u,
            star_coef_u,
            star_coef_c,
            xs_matrix_u,
            q_star,
            source_c,
            boundary_u[bb],
            boundary_c,
            medium_map,
            delta_x,
            angle_xu,
            angle_wu,
            bc_x,
            coarse_idx,
            factor,
            edges_gidx_c,
            edges_g,
            geometry,
        )

        # Solve for angular flux
        flux_last_2 = flux_last_1.copy()
        flux_last_1 = mg.known_source_angular(
            xs_total_u + star_coef_u,
            q_star,
            boundary_u[bb],
            medium_map,
            delta_x,
            angle_xu,
            angle_wu,
            bc_x,
            geometry,
            edges=0,
        )

        # Step 5: Update and repeat
        flux_time[step] = np.sum(flux_last_1 * angle_wu[None, :, None], axis=1)

        # Update xs_totat_star (for BDF2 steps)
        if step == 0:
            star_coef_u = 1.5 / (velocity_u * dt)
            if isinstance(groups_c, int):
                star_coef_c = 1.5 / (
                    hytools.coarsen_velocity(velocity_u, edges_gidx_c) * dt
                )

    return flux_time


def _variable_hybrid_method(
    angles_c,
    flux_u,
    flux_c,
    xs_total_u,
    star_coef_u,
    star_coef_c,
    xs_scatter_u,
    q_star,
    source_c,
    boundary_u,
    boundary_c,
    medium_map,
    delta_x,
    angle_xu,
    angle_wu,
    bc_x,
    coarse_idx,
    factor,
    edges_gidx_c,
    edges_g,
    geometry,
):

    # Step 1: Solve Uncollided Equation known_source (I x N x G) -> (I x G)
    flux_u = mg.known_source_scalar(
        xs_total_u + star_coef_u,
        q_star,
        boundary_u,
        medium_map,
        delta_x,
        angle_xu,
        angle_wu,
        bc_x,
        geometry,
        edges=0,
    )

    # Step 2: Compute collided source (I x G')
    tools._hybrid_source_collided(
        flux_u, xs_scatter_u, source_c, medium_map, coarse_idx
    )

    # Step 3: Solve Collided Equation (I x G')
    angle_xc, angle_wc = discrete1.angular_x(angles_c, bc_x)
    delta_coarse = np.diff(np.asarray(edges_g)[edges_gidx_c])
    delta_fine = np.diff(edges_g)

    # Give good estimate for time step
    tools._coarsen_flux(flux_u, flux_c, edges_gidx_c)

    flux_c = mg.variable_source_iteration(
        flux_c,
        xs_total_u,
        star_coef_c,
        xs_scatter_u,
        source_c,
        boundary_c,
        medium_map,
        delta_x,
        angle_xc,
        angle_wc,
        bc_x,
        delta_coarse,
        delta_fine,
        edges_gidx_c,
        geometry,
    )

    # Step 4: Create a new source and solve for angular flux
    tools._hybrid_source_total(
        flux_u, flux_c, xs_scatter_u, q_star, medium_map, coarse_idx, factor
    )
