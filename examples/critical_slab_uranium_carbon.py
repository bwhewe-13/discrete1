import numpy as np

import discrete1
from discrete1.critical1d import power_iteration

cells_x = 1000
angles = 8
groups = 87
bc_x = [0, 0]

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
# np.savez("uranium-carbon.npz", **data)
