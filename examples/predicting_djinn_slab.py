"""Example using DJINN models to accelerate scatter/fission estimation.

This script demonstrates plugging pre-trained DJINN models into the
power-iteration eigenvalue solver to accelerate or replace parts of the
transport operator (scatter and/or fission kernels). It shows how to
load models, label materials, update cross sections, and run a
transport eigenvalue estimate.

Usage: adapt model paths and labels, then run with a Python interpreter.
Requires `discrete1`, NumPy, and trained DJINN model checkpoints.
"""

import numpy as np

import discrete1
from discrete1.constants import HDPE_MM
from discrete1.djinn1d import power_iteration
from discrete1.utils import machine_learning as ml

cells_x = 1000
angles = 8
groups = 87
bc_x = [0, 1]

enrich = "15"


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
    [1, f"uranium-hydride-%{enrich}%", "45-80"],
    [2, "uranium-hydride-%00%", "80-100"],
]

# Medium Map and Materials
medium_map = discrete1.spatial1d(layout, edges_x)

materials = np.array(layout)[:, 1]
xs_total, xs_scatter, xs_fission = discrete1.materials(groups, materials)

initial = np.load("reference-data/reference-enrich-15pct.npz")["flux"]
print("Reference Keff", np.load("reference-data/reference-enrich-15pct.npz")["keff"])

########################################################################
# Fission models
########################################################################
# # Load fission models
# fpath = "models/fission_uh3_label/"
# fmodel_paths = [0, fpath + "model_002003", fpath + "model_002003"]
# fmodels = ml.load_djinn_models(fmodel_paths)

# # Get reflective label
# hdpe_mm = np.round(HDPE_MM, 2)
# label_layout = [[hdpe_mm, "high-density-polyethyene-087", "0-45"], \
#                 [0.01 * float(enrich), f"uranium-hydride-%{enrich}%", "45-80"], \
#                 [0.0, "uranium-hydride-%00%", "80-100"]]
# flabels = discrete1.spatial1d(label_layout, edges_x, labels=True)

# # Update cross sections
# xs_fission = ml.update_cross_sections(xs_fission, [1, 2])


# flux, keff = power_iteration(initial, xs_total, xs_scatter, xs_fission, \
#                 medium_map, delta_x, angle_x, angle_w, bc_x, geometry=1, \
#                 fission_models=fmodels, scatter_models=[], \
#                 fission_labels=flabels, scatter_labels=None)

########################################################################
# Scatter models
########################################################################
# Load scatter models
spath_hdpe = "models/scatter_hdpe_label/"
spath_uh3 = "models/scatter_uh3_label/"
smodel_paths = [
    spath_hdpe + "model_001004",
    spath_uh3 + "model_001004",
    spath_uh3 + "model_001004",
]
smodels = ml.load_djinn_models(smodel_paths)

# Get reflective label
hdpe_mm = np.round(HDPE_MM, 2)
label_layout = [
    [hdpe_mm, "high-density-polyethyene-087", "0-45"],
    [0.01 * float(enrich), f"uranium-hydride-%{enrich}%", "45-80"],
    [0.0, "uranium-hydride-%00%", "80-100"],
]
slabels = discrete1.spatial1d(label_layout, edges_x, labels=True)

# Update cross sections
xs_scatter = ml.update_cross_sections(xs_scatter, [0, 1, 2])


flux, keff = power_iteration(
    initial,
    xs_total,
    xs_scatter,
    xs_fission,
    medium_map,
    delta_x,
    angle_x,
    angle_w,
    bc_x,
    geometry=1,
    fission_models=[],
    scatter_models=smodels,
    fission_labels=None,
    scatter_labels=slabels,
)


########################################################################
# Save data
########################################################################

# data = {"flux": flux, "keff": keff}
# np.savez(f"estimation-{enrich}pct.npz", **data)
