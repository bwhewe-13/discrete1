import numpy as np

import discrete1
from discrete1.critical1d import power_iteration

cells_x = 1000
angles = 8
groups = 87
bc_x = [0, 1]

# Spatial
length_x = 100.0
delta_x = np.repeat(length_x / cells_x, cells_x)
edges_x = np.linspace(0.0, length_x, cells_x + 1)
centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

# Angular
angle_x, angle_w = discrete1.angular_x(angles, bc_x)


# Original Layout
layout = [
    [0, "high-density-polyethyene-087", "0-45"],
    [1, "uranium-hydride-%20%", "45-80"],
    [2, "uranium-hydride-%00%", "80-100"],
]

# # Multizone Layout
# layout = [[0, "high-density-polyethyene-087", "0-10, 20-30, 40-50, 60-70"], \
#           [1, "uranium-hydride-%20%", "10-20, 30-40, 50-60, 70-80"], \
#           [2, "uranium-hydride-%00%", "80-100"]]

# # Mixed Enrich Layout
# layout = [[0, "high-density-polyethyene-087", "0-45"], \
#           [1, "uranium-hydride-%12%", "45-50, 75-80"], \
#           [2, "uranium-hydride-%27%", "50-75"], \
#           [3, "uranium-hydride-%00%", "80-100"]]

# Medium Map and Materials
medium_map = discrete1.spatial1d(layout, edges_x)

materials = np.array(layout)[:, 1]
xs_total, xs_scatter, xs_fission = discrete1.materials(groups, materials)


flux, keff = power_iteration(
    xs_total,
    xs_scatter,
    xs_fission,
    medium_map,
    delta_x,
    angle_x,
    angle_w,
    bc_x,
    geometry=1,
)

data = {"flux": flux, "keff": keff}
# np.savez("uranium-hydride.npz", **data)
