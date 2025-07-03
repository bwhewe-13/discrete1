import numpy as np

import discrete1
from discrete1 import hybrid1d, vhybrid1d
from discrete1.utils import hybrid as hytools

cells_x = 1000
angles_u = 8
angles_c = 4
groups_u = 87
groups_c = 43

steps = 5
dt = 1e-8
bc_x = [0, 0]


# Spatial
length_x = 10.0
delta_x = np.repeat(length_x / cells_x, cells_x)
edges_x = np.linspace(0, length_x, cells_x + 1)
centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

# Energy Grid
energy_grid = discrete1.energy_grid(87, groups_u, groups_c)
edges_g, edges_gidx_u, edges_gidx_c = energy_grid
velocity_u = discrete1.energy_velocity(groups_u, edges_g)

# Angular
angle_xu, angle_wu = discrete1.angular_x(angles_u, bc_x)

# Layout and Materials
layout = [[0, "stainless-steel-440", "0-4, 6-10"], [1, "uranium-%20%", "4-6"]]
medium_map = discrete1.spatial1d(layout, edges_x)

# Cross Sections - Uncollided
materials = np.array(layout)[:, 1]
xs_total_u, xs_scatter_u, xs_fission_u = discrete1.materials(87, materials)

# Get hybrid parameters
fine_idx, coarse_idx, factor = hytools.indexing(*energy_grid)

# Check for same number of energy groups
if groups_c == groups_u:
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
if angles_c == angles_u:
    angle_xc = angle_xu.copy()
    angle_wc = angle_wu.copy()
else:
    angle_xc, angle_wc = discrete1.angular_x(angles_c, bc_x)

# External and boundary sources
external_u = np.zeros((1, cells_x, 1, 1))
boundary_u = discrete1.boundary1d.deuterium_tritium(0, edges_g)
edges_t = np.linspace(0, steps * dt, steps + 1)
boundary_u = discrete1.boundary1d.time_dependence_decay_02(boundary_u, edges_t)


flux_last = np.zeros((cells_x, angles_u, groups_u))

hy_flux = hybrid1d.backward_euler(
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
)

vhy_flux = vhybrid1d.backward_euler(
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
    coarse_idx,
    factor,
    edges_gidx_c,
    edges_g,
    steps,
    dt,
    geometry=1,
)

# print(np.sum(hy_flux, axis=(1, 2)))
# print(np.sum(vhy_flux, axis=(1, 2)))
