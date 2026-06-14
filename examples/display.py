"""Compare and plot saved anisotropic flux/keff results."""

import matplotlib.pyplot as plt
import numpy as np

data1 = np.load("test-anisotropic-06.npz")
data2 = np.load("anisotropic_critical.npz")

print(data1["keff"] - data2["keff"])

fig, ax = plt.subplots()
ax.plot(data1["flux"] - data2["flux"], label="Test Anisotropic 06")
# ax.plot(, label="Anisotropic Critical")
ax.grid(which="both")
ax.legend()
plt.show()
