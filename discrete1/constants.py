# Module constants for discrete1
"""Physical and numerical constants used across the discrete1 package.

This module exposes numeric constants (unit conversions, material molar
masses, densities, and convergence tolerances) so they can be imported
from other modules without duplication.
"""

# Convergence parameters - iterations
COUNT_ANGULAR = 100
COUNT_ENERGY = 100
COUNT_POWER = 100

# Convergence parameters - difference
CHANGE_ANGULAR = 1e-12
CHANGE_ENERGY = 1e-08
CHANGE_POWER = 1e-06  # For Power Iterations

# Conversion Between Units
MASS_NEUTRON = 1.67493e-27
EV_TO_JOULES = 1.60218e-19
LIGHT_SPEED = 2.9979246e8
AVAGADRO = 6.022e23
CM_TO_BARNS = 1e-24


# Isotope and Material Molar Mass
URANIUM_MM = 238.0289
URANIUM_235_MM = 235.04393
URANIUM_238_MM = 238.0289
URANIUM_HYDRIDE_MM = 240.60467449999996

HYDROGEN_MM = 1.00784
CARBON_MM = 12.0116
HDPE_MM = 15.03512
MANGANESE_MM = 54.9380471
OXYGEN_MM = 15.9994
STAINLESS_440_MM = 52.68213573619713

CHROMIUM_50_MM = 49.9460464
CHROMIUM_52_MM = 51.9405098
CHROMIUM_53_MM = 52.9406513
CHROMIUM_54_MM = 53.9388825
COPPER_63_MM = 62.929597
COPPER_65_MM = 64.927789
IRON_54_MM = 53.9396127
IRON_56_MM = 55.9349393
IRON_57_MM = 56.9353958
SILICON_28_MM = 27.9769271
SILICON_29_MM = 28.9764949
SILICON_30_MM = 29.9737707


# Isotope and Material Densities
URANIUM_RHO = 19.1
URANIUM_235_RHO = 18.8
URANIUM_238_RHO = 18.95
URANIUM_HYDRIDE_RHO = 10.95

HYDROGEN_RHO = 0.07
CARBON_RHO = 2.26
HDPE_RHO = 0.97
MANGANESE_RHO = 7.3
OXYGEN_RHO = 1.429
STAINLESS_440_RHO = 7.85
STAINLESS_STEEL_RHO = 7.85
CHROMIUM_RHO = 7.19
COPPER_RHO = 8.96
IRON_RHO = 7.86
SILICON_RHO = 2.329


# Isotope Abundance
CHROMIUM_50 = 0.04345
CHROMIUM_52 = 0.83789
CHROMIUM_53 = 0.09501
CHROMIUM_54 = 0.02365
COPPER_63 = 0.6917
COPPER_65 = 0.3083
IRON_54 = 0.05845
IRON_56 = 0.92036
IRON_57 = 0.02119
SILICON_28 = 0.922297
SILICON_29 = 0.046832
SILICON_30 = 0.030872


# Stainless Compound
SS_440_IRON = 0.79
SS_440_CHROMIUM = 0.18
SS_440_CARBON = 0.01
SS_440_SILICON = 0.01
SS_440_MANGANESE = 0.01

# Metal Element Dictionary
STAINLESS_ELEMENTS = {
    "fe-54": {"molar_mass": IRON_54_MM, "abundance": IRON_54},
    "fe-56": {"molar_mass": IRON_56_MM, "abundance": IRON_56},
    "fe-57": {"molar_mass": IRON_57_MM, "abundance": IRON_57},
    "cr-50": {"molar_mass": CHROMIUM_50_MM, "abundance": CHROMIUM_50},
    "cr-52": {"molar_mass": CHROMIUM_52_MM, "abundance": CHROMIUM_52},
    "cr-53": {"molar_mass": CHROMIUM_53_MM, "abundance": CHROMIUM_53},
    "cr-54": {"molar_mass": CHROMIUM_54_MM, "abundance": CHROMIUM_54},
    "si-28": {"molar_mass": SILICON_28_MM, "abundance": SILICON_28},
    "si-29": {"molar_mass": SILICON_29_MM, "abundance": SILICON_29},
    "si-30": {"molar_mass": SILICON_30_MM, "abundance": SILICON_30},
    "mn-55": {"molar_mass": MANGANESE_MM, "abundance": 1.0},
    "c-12": {"molar_mass": CARBON_MM, "abundance": 1.0},
}
