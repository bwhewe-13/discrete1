"""Material cross-section generation and management.

This module provides tools for generating and managing neutron cross-sections
for various materials used in transport calculations. It supports both
enriched (uranium, plutonium, uranium-hydride) and non-enriched materials,
with cross-sections loaded from precomputed data files.

The module handles:
- Cross-section lookup for standard materials
- Enrichment calculations for fissile materials
- Special composition calculations (e.g., uranium hydride)
- Vacuum/void material properties
"""

import numpy as np
import pkg_resources

import discrete1.constants as const

DATA_PATH = pkg_resources.resource_filename("discrete1", "sources/")

########################################################################
# Material Cross Sections
########################################################################

__enrichment_materials = ("uranium", "uranium-hydride", "plutonium")

__nonenrichment_materials = (
    "stainless-steel-440",
    "hydrogen",
    "high-density-polyethyene-618",
    "high-density-polyethyene-087",
    "carbon",
    "uranium-235",
    "uranium-238",
    "water-uranium-dioxide",
    "plutonium-239",
    "plutonium-240",
    "vacuum",
)

__materials = __enrichment_materials + __nonenrichment_materials


def materials(groups, materials, key=False):
    """Create cross sections for different materials.

    Generates total, scatter, and fission cross-sections for a list of
    materials. Handles both enriched and non-enriched materials, with
    proper composition calculations for mixtures.

    Parameters
    ----------
    groups : int
        Number of energy groups.
    materials : list
        List of material names (str). Each name can include enrichment
        percentage using '-X%' suffix.
    key : bool, optional
        If True, return a mapping of indices to material names.

    Returns
    -------
    numpy.ndarray
        Total cross-sections (n_materials, n_groups).
    numpy.ndarray
        Scatter cross-sections (n_materials, n_groups, n_groups).
    numpy.ndarray
        Fission cross-sections (n_materials, n_groups, n_groups).
    dict, optional
        Material index to name mapping if key=True.

    Notes
    -----
    Supported materials are defined in __materials tuple. Enriched
    materials support percentage specification (e.g., 'uranium-5%').
    """
    material_key = {}
    xs_total = []
    xs_scatter = []
    xs_fission = []
    for idx, material in enumerate(materials):
        # Verify it is possible
        assert (
            material.split("-%")[0] in __materials
        ), "Material not recognized, use:\n{}".format(__materials)
        # Calculate cross section
        total, scatter, fission = _generate_cross_section(groups, material)
        xs_total.append(total)
        xs_scatter.append(scatter)
        xs_fission.append(fission)
        material_key[idx] = material
    xs_total = np.array(xs_total)
    xs_scatter = np.array(xs_scatter)
    xs_fission = np.array(xs_fission)
    if key:
        return xs_total, xs_scatter, xs_fission, material_key
    return xs_total, xs_scatter, xs_fission


def _generate_cross_section(groups, material):
    data = {}
    if "%" in material:
        material, enrichment = material.split("-%")
        enrichment = float(enrichment.strip("%")) * 0.01

    if material == "vacuum":
        return (
            np.zeros((groups)),
            np.zeros((groups, groups)),
            np.zeros((groups, groups)),
        )
    elif material in __nonenrichment_materials:
        data = np.load(DATA_PATH + "materials/" + material + ".npz")
    elif material == "uranium":
        u235 = np.load(DATA_PATH + "materials/uranium-235.npz")
        u238 = np.load(DATA_PATH + "materials/uranium-238.npz")
        for xs in u235.files:
            data[xs] = u235[xs] * enrichment + u238[xs] * (1 - enrichment)
    elif material == "plutonium":
        pu239 = np.load(DATA_PATH + "materials/plutonium-239.npz")
        pu240 = np.load(DATA_PATH + "materials/plutonium-240.npz")
        for xs in pu239.files:
            data[xs] = pu239[xs] * enrichment + pu240[xs] * (1 - enrichment)
    elif material == "uranium-hydride":
        return _generate_uranium_hydride(enrichment)

    return data["total"], data["scatter"], data["fission"]


def _generate_uranium_hydride(enrichment):
    molar = enrichment * const.URANIUM_235_MM + (1 - enrichment) * const.URANIUM_238_MM
    rho = const.URANIUM_HYDRIDE_RHO / const.URANIUM_RHO

    n235 = (enrichment * rho * molar) / (molar + 3 * const.HYDROGEN_MM)
    n238 = ((1 - enrichment) * rho * molar) / (molar + 3 * const.HYDROGEN_MM)
    n1 = const.URANIUM_HYDRIDE_RHO * const.AVAGADRO / (molar + 3 * const.HYDROGEN_MM)
    n1 *= const.CM_TO_BARNS * 3

    u235 = np.load(DATA_PATH + "materials/uranium-235.npz")
    u238 = np.load(DATA_PATH + "materials/uranium-238.npz")
    h1 = np.load(DATA_PATH + "materials/hydrogen.npz")

    total = n235 * u235["total"] + n238 * u238["total"] + n1 * h1["total"]
    scatter = n235 * u235["scatter"] + n238 * u238["scatter"] + n1 * h1["scatter"]
    fission = n235 * u235["fission"] + n238 * u238["fission"] + n1 * h1["fission"]
    return total, scatter, fission
