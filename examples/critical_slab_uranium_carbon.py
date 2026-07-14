"""k-Eigenvalue Criticality Problem for 1D Slab."""

import matplotlib.pyplot as plt
import numpy as np

import discrete1
from discrete1.critical1d import hybrid_power_iteration, power_iteration

cells_x = 200
angles = 8
groups = 87
bc_x = [0, 0]

# Hybrid Parameters
angles_c = 8
groups_c = 87
energy_grid = discrete1.energy_grid(87, groups, groups_c)

# Spatial
length_x = 40.0
delta_x = np.repeat(length_x / cells_x, cells_x)
edges_x = np.linspace(0.0, length_x, cells_x + 1)
centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

# Angular
angle_x, angle_w = discrete1.angular_x(angles, bc_x)

# Layout
layout = [[0, "carbon", "0-10, 14-26, 30-40"], [1, "uranium-%25%", "10-14, 26-30"]]

# Medium Map and Materials
medium_map = discrete1.spatial1d(layout, edges_x)

materials = np.array(layout)[:, 1]
xs_total, xs_scatter, xs_fission = discrete1.materials(groups, materials)

flux, keff = hybrid_power_iteration(
    xs_total,
    xs_scatter,
    xs_fission,
    medium_map,
    delta_x,
    angle_x,
    angle_w,
    bc_x,
    angles_c,
    groups_c,
    energy_grid,
    geometry=1,
)

# flux, keff = power_iteration(
#     xs_total,
#     xs_scatter,
#     xs_fission,
#     medium_map,
#     delta_x,
#     angle_x,
#     angle_w,
#     bc_x,
#     geometry=1,
# )

# data = {"flux": flux, "keff": keff}
# np.savez("reference-uranium-carbon.npz", **data)

ref = np.load("reference-uranium-carbon.npz")  # S8

fig, ax = plt.subplots(1, 2)
label = f"Reference {np.round(ref['keff'], 8)}"
ax[0].plot(np.sum(ref["flux"], axis=1), c="k", ls=":", label=label)
ax[0].plot(np.sum(flux, axis=1), c="r", alpha=0.7, label=f"Approx {np.round(keff, 8)}")
ax[0].set_title("Axis = 1")

ax[1].plot(
    np.sum(ref["flux"], axis=0),
    c="k",
    ls=":",
    label=f"Reference {np.round(ref['keff'], 8)}",
)
ax[1].plot(np.sum(flux, axis=0), c="r", alpha=0.7, label=f"Approx {np.round(keff, 8)}")
ax[1].set_title("Axis = 0")

for ii in range(2):
    ax[ii].legend(loc=0, framealpha=1)
    ax[ii].grid(which="both")

plt.show()
