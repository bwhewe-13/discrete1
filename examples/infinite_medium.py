"""Solve an infinite medium criticality example using power iteration."""

import numpy as np

from discrete1.critical0d import power_iteration

angles = 16

# One Group
xs_total = np.array([0.32640])
xs_scatter = np.array([[0.225216]])
xs_fission = np.array([[3.24 * 0.0816]])
chi = None

# # Two Group
# xs_total = np.array([0.3360, 0.2208])
# xs_scatter = np.array([[0.23616, 0.0432], [0.0, 0.0792]])
# xs_fission = np.array([2.93, 3.1]) * np.array([0.08544, 0.0936])
# chi = np.array([0.425, 0.575])


flux, keff = power_iteration(angles, xs_total, xs_scatter, xs_fission, chi)
print(f"Converged keff: {keff:.8f}")
