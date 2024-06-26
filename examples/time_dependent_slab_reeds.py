
import numpy as np
import matplotlib.pyplot as plt


import discrete1
from discrete1.fixed1d import source_iteration
from discrete1.timed1d import backward_euler, bdf2

cells_x = 160
angles = 4
groups = 1
bc_x = [0, 1]


# Spatial
length_x = 8. if np.sum(bc_x) > 0.0 else 16.
delta_x = np.repeat(length_x / cells_x, cells_x)
edges_x = np.linspace(0, length_x, cells_x+1)
centers_x = 0.5 * (edges_x[1:] + edges_x[:-1])

# Angular
angle_x, angle_w = discrete1.angular_x(angles, bc_x)

# Layout and Materials
if bc_x == [0, 0]:
    layout = [[0, "scattering", "0-4, 12-16"], [1, "vacuum", "4-5, 11-12"], \
              [2, "absorber", "5-6, 10-11"], [3, "source", "6-10"]]
elif bc_x == [0, 1]:
    layout = [[0, "scattering", "0-4"], [1, "vacuum", "4-5"], \
              [2, "absorber", "5-6"], [3, "source", "6-8"]]
elif bc_x == [1, 0]:
    layout = [[0, "scattering", "4-8"], [1, "vacuum", "3-4"], \
              [2, "absorber", "2-3"], [3, "source", "0-2"]]
medium_map = discrete1.spatial1d(layout, edges_x)

xs_total = np.array([[1.0], [0.0], [5.0], [50.0]])
xs_scatter = np.array([[[0.9]], [[0.0]], [[0.0]], [[0.0]]])
xs_fission = np.array([[[0.0]], [[0.0]], [[0.0]], [[0.0]]])

# Sources
external = discrete1.external1d.reeds(edges_x, bc_x)
boundary = np.zeros((2, 1, 1))

ss_flux = source_iteration(xs_total, xs_scatter, xs_fission, external, boundary, \
                    medium_map, delta_x, angle_x, angle_w, bc_x, geometry=1)

# Make boundary and external time-dependent
external = external[None,:,:,:].copy()
boundary = boundary[None,:,:,:].copy()

flux_last = np.zeros((cells_x, angles, groups))
velocity = np.ones((groups,))
td_flux = backward_euler(flux_last, xs_total, xs_scatter, xs_fission, velocity, \
                        external, boundary, medium_map, delta_x, angle_x, \
                        angle_w, bc_x, steps=100, dt=1, geometry=1)


fig, ax = plt.subplots()
ax.plot(centers_x, ss_flux.flatten(), c="k", ls=":", label="Steady State")
ax.plot(centers_x, td_flux[-1,:,0], c="r", alpha=0.6, label="Backward Euler")
ax.grid(which="both")
ax.legend(loc=0, framealpha=1)
plt.show()