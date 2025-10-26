"""Hybrid (multi-grid) slab example with uranium materials.

This example demonstrates time-dependent and hybrid transport
integration for an enriched uranium configuration. It compares a
multi-group time integrator (timed1d) to a hybrid reduced-order
integration (hybrid1d) and prints simple integral diagnostics.

Usage: run this file with a Python interpreter. It requires the
`discrete1` package and NumPy.
"""

import numpy as np

import discrete1
from discrete1 import hybrid1d, timed1d

cells_x = 1000
angles_u = 8
angles_c = 8
groups_u = 87
groups_c = 87

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
velocity = discrete1.energy_velocity(groups_u, edges_g)

# Angular
angle_x, angle_w = discrete1.angular_x(angles_u, bc_x)

# Layout and Materials
layout = [[0, "stainless-steel-440", "0-4, 6-10"], [1, "uranium-%20%", "4-6"]]
medium_map = discrete1.spatial1d(layout, edges_x)

# Cross Sections - Uncollided
materials = np.array(layout)[:, 1]
xs_total, xs_scatter, xs_fission = discrete1.materials(87, materials)

# External and boundary sources
external = np.zeros((1, cells_x, 1, 1))
boundary = discrete1.boundary1d.deuterium_tritium(0, edges_g)
edges_t = np.linspace(0, steps * dt, steps + 1)
boundary = discrete1.boundary1d.time_dependence_decay_02(boundary, edges_t)


flux_last = np.zeros((cells_x, angles_u, groups_u))

mg_flux = timed1d.backward_euler(
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
    steps=steps,
    dt=dt,
    geometry=1,
)


hy_flux = hybrid1d.auto_bdf1(
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
    angles_c,
    groups_c,
    energy_grid,
    steps=steps,
    dt=dt,
    geometry=1,
)

print(np.sum(mg_flux, axis=(1, 2)))
print(np.sum(hy_flux, axis=(1, 2)))
