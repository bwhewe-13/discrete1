# Called for time dependent source problems

import numpy as np
from tqdm import tqdm

import discrete1
from discrete1 import multi_group as mg
from discrete1 import tools
from discrete1.utils import hybrid as hytools


def auto_bdf1(
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
    angles_c,
    groups_c,
    energy_grid,
    steps,
    dt,
    geometry=1,
):

    # Get hybrid parameters
    edges_g, edges_gidx_u, edges_gidx_c = energy_grid
    fine_idx, coarse_idx, factor = hytools.indexing(*energy_grid)

    # Check for same number of energy groups
    if groups_c == flux_last.shape[2]:
        xs_total_c = xs_total_u.copy()
        xs_scatter_c = xs_scatter_u.copy()
        xs_fission_c = xs_fission_u.copy()
        velocity_c = velocity_u.copy()
    else:
        xs_collided = hytools.coarsen_materials(
            xs_total_u, xs_scatter_u, xs_fission_u, edges_g[edges_gidx_u], edges_gidx_c
        )
        xs_total_c, xs_scatter_c, xs_fission_c = xs_collided
        velocity_c = hytools.coarsen_velocity(velocity_u, edges_gidx_c)

    # Check for same number of angles
    if angles_c == flux_last.shape[1]:
        angle_xc = angle_xu.copy()
        angle_wc = angle_wu.copy()
    else:
        angle_xc, angle_wc = discrete1.angular_x(angles_c, bc_x)

    return backward_euler(
        flux_last,
        xs_total_u,
        xs_total_c,
        xs_scatter_u,
        xs_scatter_c,
        xs_fission_u,
        xs_fission_c,
        velocity_u,
        velocity_c,
        external_u,
        boundary_u,
        medium_map,
        delta_x,
        angle_xu,
        angle_xc,
        angle_wu,
        angle_wc,
        bc_x,
        fine_idx,
        coarse_idx,
        factor,
        steps,
        dt,
        geometry,
    )


def backward_euler(
    flux_last,
    xs_total_u,
    xs_total_c,
    xs_scatter_u,
    xs_scatter_c,
    xs_fission_u,
    xs_fission_c,
    velocity_u,
    velocity_c,
    external_u,
    boundary_u,
    medium_map,
    delta_x,
    angle_xu,
    angle_xc,
    angle_wu,
    angle_wc,
    bc_x,
    fine_idx,
    coarse_idx,
    factor,
    steps,
    dt,
    geometry=1,
):

    # Scalar flux approximation
    flux_u = np.sum(flux_last * angle_wu[None, :, None], axis=1)
    flux_c = np.zeros((medium_map.shape[0], xs_total_c.shape[1]))

    # Scalar flux for every time step
    flux_time = np.zeros((steps,) + flux_u.shape)

    # Combine scattering and fission
    xs_matrix_u = xs_scatter_u + xs_fission_u
    xs_matrix_c = xs_scatter_c + xs_fission_c

    # Create xs_total_star
    xs_total_u += 1 / (velocity_u * dt)
    xs_total_c += 1 / (velocity_c * dt)

    # Initialize collided source and boundary
    source_c = np.zeros((medium_map.shape[0], 1, xs_total_c.shape[1]))
    boundary_c = np.zeros((2, 1, 1))

    # Iterate over time steps
    for step in tqdm(range(steps), desc="BDF1*", ascii=True):
        # Determine dimensions of external and boundary sources
        qq = 0 if external_u.shape[0] == 1 else step
        bb = 0 if boundary_u.shape[0] == 1 else step

        # Update q_star
        q_star = external_u[qq] + 1 / (velocity_u * dt) * flux_last

        # Run hybrid method
        _hybrid_method(
            flux_u,
            flux_c,
            xs_total_u,
            xs_total_c,
            xs_matrix_u,
            xs_matrix_c,
            q_star,
            source_c,
            boundary_u[bb],
            boundary_c,
            medium_map,
            delta_x,
            angle_xu,
            angle_xc,
            angle_wu,
            angle_wc,
            bc_x,
            fine_idx,
            coarse_idx,
            factor,
            geometry,
        )

        # Solve for angular flux
        flux_last = mg.known_source_angular(
            xs_total_u,
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


def auto_bdf2(
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
    angles_c,
    groups_c,
    energy_grid,
    steps,
    dt,
    geometry=1,
):

    # Get hybrid parameters
    edges_g, edges_gidx_u, edges_gidx_c = energy_grid
    fine_idx, coarse_idx, factor = hytools.indexing(*energy_grid)

    # Check for same number of energy groups
    if groups_c == flux_last_1.shape[2]:
        xs_total_c = xs_total_u.copy()
        xs_scatter_c = xs_scatter_u.copy()
        xs_fission_c = xs_fission_u.copy()
        velocity_c = velocity_u.copy()
    else:
        xs_collided = hytools.coarsen_materials(
            xs_total_u, xs_scatter_u, xs_fission_u, edges_g[edges_gidx_u], edges_gidx_c
        )
        xs_total_c, xs_scatter_c, xs_fission_c = xs_collided
        velocity_c = hytools.coarsen_velocity(velocity_u, edges_gidx_c)

    # Check for same number of angles
    if angles_c == flux_last_1.shape[1]:
        angle_xc = angle_xu.copy()
        angle_wc = angle_wu.copy()
    else:
        angle_xc, angle_wc = discrete1.angular_x(angles_c, bc_x)

    return bdf2(
        flux_last_1,
        xs_total_u,
        xs_total_c,
        xs_scatter_u,
        xs_scatter_c,
        xs_fission_u,
        xs_fission_c,
        velocity_u,
        velocity_c,
        external_u,
        boundary_u,
        medium_map,
        delta_x,
        angle_xu,
        angle_xc,
        angle_wu,
        angle_wc,
        bc_x,
        fine_idx,
        coarse_idx,
        factor,
        steps,
        dt,
        geometry,
    )


def bdf2(
    flux_last_1,
    xs_total_u,
    xs_total_c,
    xs_scatter_u,
    xs_scatter_c,
    xs_fission_u,
    xs_fission_c,
    velocity_u,
    velocity_c,
    external_u,
    boundary_u,
    medium_map,
    delta_x,
    angle_xu,
    angle_xc,
    angle_wu,
    angle_wc,
    bc_x,
    fine_idx,
    coarse_idx,
    factor,
    steps,
    dt,
    geometry=1,
):

    # Scalar flux approximation
    flux_u = np.sum(flux_last_1 * angle_wu[None, :, None], axis=1)
    flux_c = np.zeros((medium_map.shape[0], xs_total_c.shape[1]))

    # Angular flux of step \ell - 2
    flux_last_2 = np.zeros(flux_last_1.shape)

    # Scalar flux for every time step
    flux_time = np.zeros((steps,) + flux_u.shape)

    # Combine scattering and fission
    xs_matrix_u = xs_scatter_u + xs_fission_u
    xs_matrix_c = xs_scatter_c + xs_fission_c

    # Create xs_total_star
    xs_total_vu = xs_total_u + 1 / (velocity_u * dt)
    xs_total_vc = xs_total_c + 1 / (velocity_c * dt)

    # Initialize collided source and boundary
    source_c = np.zeros((medium_map.shape[0], 1, xs_total_c.shape[1]))
    boundary_c = np.zeros((2, 1, 1))

    # Iterate over time steps
    for step in tqdm(range(steps), desc="BDF2*", ascii=True):
        # Determine dimensions of external and boundary sources
        qq = 0 if external_u.shape[0] == 1 else step
        bb = 0 if boundary_u.shape[0] == 1 else step

        # BDF1 on first time step
        if step == 0:
            # Update q_star
            q_star = external_u[qq] + 1 / (velocity_u * dt) * flux_last_1

        else:
            q_star = (
                external_u[qq]
                + 2 / (velocity_u * dt) * flux_last_1
                - 1 / (2 * velocity_u * dt) * flux_last_2
            )

        # Run hybrid method
        _hybrid_method(
            flux_u,
            flux_c,
            xs_total_vu,
            xs_total_vc,
            xs_matrix_u,
            xs_matrix_c,
            q_star,
            source_c,
            boundary_u[bb],
            boundary_c,
            medium_map,
            delta_x,
            angle_xu,
            angle_xc,
            angle_wu,
            angle_wc,
            bc_x,
            fine_idx,
            coarse_idx,
            factor,
            geometry,
        )

        # Solve for angular flux
        flux_last_2 = flux_last_1.copy()
        flux_last_1 = mg.known_source_angular(
            xs_total_vu,
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
            xs_total_vu = xs_total_u + 1.5 / (velocity_u * dt)
            xs_total_vc = xs_total_c + 1.5 / (velocity_c * dt)

    return flux_time


def _hybrid_method(
    flux_u,
    flux_c,
    xs_total_u,
    xs_total_c,
    xs_scatter_u,
    xs_scatter_c,
    q_star,
    source_c,
    boundary_u,
    boundary_c,
    medium_map,
    delta_x,
    angle_xu,
    angle_xc,
    angle_wu,
    angle_wc,
    bc_x,
    fine_idx,
    coarse_idx,
    factor,
    geometry,
):

    # Step 1: Solve Uncollided Equation known_source (I x N x G) -> (I x G)
    flux_u = mg.known_source_scalar(
        xs_total_u,
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
    flux_c = mg.source_iteration(
        flux_c,
        xs_total_c,
        xs_scatter_c,
        source_c,
        boundary_c,
        medium_map,
        delta_x,
        angle_xc,
        angle_wc,
        bc_x,
        geometry,
    )

    # Step 4: Create a new source and solve for angular flux
    tools._hybrid_source_total(
        flux_u, flux_c, xs_scatter_u, q_star, medium_map, coarse_idx, factor
    )
