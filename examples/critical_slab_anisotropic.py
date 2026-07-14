"""Anisotropic one-group critical slab via Legendre-moment scattering."""

import numpy as np

import discrete1
from discrete1.critical1d import power_iteration

cells_x = 250
angles = 16
bc_x = [0, 0]
angle_x, angle_w = discrete1.angular_x(angles, bc_x)
xs_total = np.array([[1.0]])  # (n_materials=1, n_groups=1)
xs_fission = np.array([[[2.5 * 0.266667]]])  # (n_materials=1, n_groups=1)

# xs_scatter shape: (n_materials, n_groups, n_groups, L+1)
# Last axis holds Legendre moments [sigma_s0, sigma_s1, sigma_s2]
xs_scatter = np.array([[[[0.733333, 0.2, 0.075]]]])  # (1, 1, 1, 3)

medium_map = np.zeros((cells_x,), dtype=np.int32)

# Critical half-length for PUa-1-2-SL from Sood et al. (1999) Table 25
length = 0.76378 * 2
delta_x = np.repeat(length / cells_x, cells_x)

flux, keff = power_iteration(
    xs_total, xs_scatter, xs_fission, medium_map, delta_x, angle_x, angle_w, bc_x
)
print(f"Final Keff: {keff:.8f}")


np.savez("anisotropic_critical.npz", flux=flux, keff=keff)
