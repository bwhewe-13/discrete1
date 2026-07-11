r"""Transport-coupled depletion of a reactor slab with PyCNiC nuclear data.

End-to-end example coupling **PyCNiC** (ENDF -> NJOY -> 30-group multigroup
library) to **discrete1** (S_N transport + CRAM burnup) for the ten most
common depletion isotopes

    U-235, U-236, U-238, Np-239, Pu-239, Pu-240, Pu-241, I-135, Xe-135, Sm-149

plus H-1 and O-16 as non-depleting moderator background. The problem is a
1D slab: a homogenized research-reactor-style LEU + water core flanked by
light-water reflectors, burned at constant power over ~150 days.

The script runs in two cached phases:

1. **Generate** (needs NJOY; PyCNiC auto-detects the binary). Runs
   RECONR/BROADR/GROUPR per isotope on the NJOY IGN 3 (30-group) structure,
   merges everything into one ``MultiGroupLibrary`` (saved to
   ``depletion_pycnic_30g.h5``), attaches a hand-built depletion topology
   (decay constants, capture products, thermal fission yields), and exports
   ``depletion_pycnic_30g.npz`` in the discrete1 ``NuclideLibrary`` layout via
   :func:`pycnic.depletion.to_nuclide_library_npz`. For example, inside the
   PyCNiC Docker image (NJOY + ENDF/B-VIII.0 tapes included)::

       docker run --rm -v $PWD:/work -w /work \
           -v /path/to/PyCNiC/src:/pycnic-src -e PYTHONPATH=/pycnic-src \
           pycnic:full python depletion_pycnic_slab.py --generate

2. **Burnup** (needs discrete1; no NJOY). Loads the ``.npz`` with
   :func:`discrete1.nuclides.load_library` and runs
   :func:`discrete1.burnup1d.burnup` (power iteration per step, CE/LI
   predictor-corrector, CRAM-48 depletion), printing a step table and saving
   a two-panel figure::

       python depletion_pycnic_slab.py

Each phase is skipped automatically when its cached artifact already exists,
so the pair of commands above works even though NJOY and discrete1 live in
different environments.
"""

import argparse
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
H5_FILE = HERE / "depletion_pycnic_30g.h5"
NPZ_FILE = HERE / "depletion_pycnic_30g.npz"
PNG_FILE = HERE / "depletion_pycnic_slab.png"

TEMPERATURE = 293.6  # K
N_LEGENDRE = 2  # P0-P1 moments from GROUPR (export keeps P0)
DAY = 86400.0  # s
YEAR = 3.1557e7  # s

# The tracked inventory: 10 depletion isotopes + 2 moderator backgrounds.
# GND-form names (index order is the canonical nuclide ordering everywhere).
ISOTOPES = [
    "U-235",
    "U-236",
    "U-238",
    "Np-239",
    "Pu-239",
    "Pu-240",
    "Pu-241",
    "I-135",
    "Xe-135",
    "Sm-149",
    "H-1",
    "O-16",
]


########################################################################
# ENDF tape discovery
########################################################################

# (Z, element symbol, A) for the local ENDF/B-VIII.0 naming n-ZZZ_El_AAA.endf
_ZA = {
    "U-235": (92, "U", 235),
    "U-236": (92, "U", 236),
    "U-238": (92, "U", 238),
    "Np-239": (93, "Np", 239),
    "Pu-239": (94, "Pu", 239),
    "Pu-240": (94, "Pu", 240),
    "Pu-241": (94, "Pu", 241),
    "I-135": (53, "I", 135),
    "Xe-135": (54, "Xe", 135),
    "Sm-149": (62, "Sm", 149),
    "H-1": (1, "H", 1),
    "O-16": (8, "O", 16),
}


def find_endf_tape(isotope):
    """Locate the ENDF evaluation for *isotope*, trying common layouts.

    Search order: ``$ENDF_DIR`` (either naming), the PyCNiC Docker convention
    ``/data/endf/tapes/<iso>.txt``, and a local ENDF/B-VIII.0 download
    ``~/Downloads/ENDF-B-VIII.0/neutrons/n-ZZZ_El_AAA.endf``.
    """
    z, element, a = _ZA[isotope]
    flat = isotope.lower().replace("-", "")  # u235
    endfb = f"n-{z:03d}_{element}_{a:03d}.endf"  # n-092_U_235.endf

    candidates = []
    endf_dir = os.environ.get("ENDF_DIR")
    if endf_dir:
        candidates += [Path(endf_dir) / f"{flat}.txt", Path(endf_dir) / endfb]
    candidates += [
        Path("/data/endf/tapes") / f"{flat}.txt",
        Path.home() / "Downloads" / "ENDF-B-VIII.0" / "neutrons" / endfb,
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"No ENDF tape found for {isotope}; searched: "
        + ", ".join(str(c) for c in candidates)
    )


########################################################################
# Phase 1a: 30-group multigroup library (NJOY IGN 3) via PyCNiC
########################################################################


def build_multigroup_library():
    """Run NJOY per isotope on the IGN 3 structure and merge into one library."""
    import dataclasses

    from pycnic.benchmarks.structures import get_structure
    from pycnic.library import MultiGroupLibrary
    from pycnic.pipeline import CrossSectionPipeline

    try:
        structure = get_structure("njoy_ign3")
    except KeyError:  # older PyCNiC releases key this as "njoy-ign3"
        structure = get_structure("njoy-ign3")
    # The export keeps only the P0 scattering moment; P0-P1 from NJOY is
    # plenty and must match the structure's declared moment count.
    structure = dataclasses.replace(structure, n_legendre=N_LEGENDRE)
    print(f"NJOY IGN 3 structure: {structure.n_groups} groups")

    merged = MultiGroupLibrary(structure)
    merged.metadata["source"] = "ENDF/B-VIII.0 via NJOY GROUPR (PyCNiC)"
    merged.metadata["reference"] = "discrete1 examples/depletion_pycnic_slab.py"

    for isotope in ISOTOPES:
        tape = find_endf_tape(isotope)
        print(f"[{isotope}] NJOY <- {tape}")
        pipeline = CrossSectionPipeline(
            endf_file=str(tape),
            isotope=isotope,
            group_structure=structure,
            temperatures=[TEMPERATURE],
            n_legendre=N_LEGENDRE,
            weight_function="lwr",
            output_dir=str(HERE / "njoy_work"),
        )
        lib = pipeline.run()
        for mat in lib.materials.values():
            merged.add_material(mat)

    merged.save_hdf5(str(H5_FILE))
    print(f"Saved multigroup library -> {H5_FILE.name}")
    return merged


########################################################################
# Phase 1b: depletion topology (textbook decay / capture / yield data)
########################################################################


def build_depletion_chain():
    """Hand-build the PyCNiC ``DepletionChain`` for the tracked inventory.

    Half-lives, branching, and thermal fission yields are standard published
    values (ENDF/B decay sublibrary; England & Rider yields). Couplings whose
    product falls outside the tracked set are encoded as ``-1`` (the parent
    still loses atoms). Two standard chain simplifications: U-238 capture goes
    directly to Np-239 (the 23.5-min U-239 step is skipped), and Sm-149 gets a
    lumped cumulative yield (the Nd-149/Pm-149 precursors are skipped).
    """
    from pycnic.depletion import REACTIONS, DepletionChain

    names = [iso.replace("-", "") for iso in ISOTOPES]  # GND form: U235, ...
    idx = {name: i for i, name in enumerate(names)}
    m = len(names)

    half_life = np.array(
        [
            7.04e8 * YEAR,  # U-235
            2.342e7 * YEAR,  # U-236
            4.468e9 * YEAR,  # U-238
            2.356 * DAY,  # Np-239 -> Pu-239
            2.411e4 * YEAR,  # Pu-239
            6.561e3 * YEAR,  # Pu-240
            14.329 * YEAR,  # Pu-241 -> Am-241 (untracked)
            6.57 * 3600.0,  # I-135  -> Xe-135
            9.14 * 3600.0,  # Xe-135 -> Cs-135 (untracked)
            np.inf,  # Sm-149
            np.inf,  # H-1
            np.inf,  # O-16
        ]
    )

    # Decay couplings into the tracked set. Xe-135 and Pu-241 decay to
    # untracked daughters: no coupling entry, but their half-life above still
    # removes atoms. The alpha decays of the long-lived actinides are
    # negligible on burnup timescales and likewise carry no coupling.
    decay_from = np.array([idx["Np239"], idx["I135"]], dtype=np.int64)
    decay_to = np.array([idx["Pu239"], idx["Xe135"]], dtype=np.int64)
    decay_branch = np.array([1.0, 1.0])

    # (n,gamma) products; -1 = product not tracked (U-237, Np-240, Pu-242,
    # I-136, Xe-136, Sm-150, H-2, O-17).
    reaction_product = {ch: np.full(m, -1, dtype=np.int64) for ch in REACTIONS}
    capture = reaction_product["(n,gamma)"]
    capture[idx["U235"]] = idx["U236"]
    capture[idx["U238"]] = idx["Np239"]  # via 23.5-min U-239, skipped
    capture[idx["Pu239"]] = idx["Pu240"]
    capture[idx["Pu240"]] = idx["Pu241"]

    # (n,2n) channels that stay inside the tracked set.
    n2n = reaction_product["(n,2n)"]
    n2n[idx["U236"]] = idx["U235"]
    n2n[idx["Pu240"]] = idx["Pu239"]
    n2n[idx["Pu241"]] = idx["Pu240"]

    # Thermal fission yields (U-238: fast). I-135 and Sm-149 use cumulative
    # chain yields; Xe-135 the independent yield (the rest arrives via I-135).
    yields = {
        "U235": {"I135": 0.0639, "Xe135": 0.00237, "Sm149": 0.0113},
        "U238": {"I135": 0.0610, "Xe135": 0.0010, "Sm149": 0.0090},
        "Pu239": {"I135": 0.0654, "Xe135": 0.0110, "Sm149": 0.0125},
        "Pu241": {"I135": 0.0693, "Xe135": 0.0070, "Sm149": 0.0140},
    }
    fy_parent, fy_product, fy_yield = [], [], []
    for parent, products in yields.items():
        for product, value in products.items():
            fy_parent.append(idx[parent])
            fy_product.append(idx[product])
            fy_yield.append(value)

    # Energy per fission (J). All actinides get a value so the power
    # normalization also counts threshold fission in U-236/Np-239/Pu-240.
    kappa = np.zeros(m)
    kappa[idx["U235"]] = 3.24e-11  # ~202 MeV
    kappa[idx["U236"]] = 3.2e-11
    kappa[idx["U238"]] = 3.35e-11
    kappa[idx["Np239"]] = 3.2e-11
    kappa[idx["Pu239"]] = 3.33e-11  # ~208 MeV
    kappa[idx["Pu240"]] = 3.2e-11
    kappa[idx["Pu241"]] = 3.44e-11

    return DepletionChain(
        names=names,
        half_life=half_life,
        decay_from=decay_from,
        decay_to=decay_to,
        decay_branch=decay_branch,
        reaction_product=reaction_product,
        fy_parent=np.array(fy_parent, dtype=np.int64),
        fy_product=np.array(fy_product, dtype=np.int64),
        fy_yield=np.array(fy_yield, dtype=np.float64),
        kappa=kappa,
    )


def generate():
    """Phase 1: build (or load) the 30-group library and export the ``.npz``."""
    from pycnic.depletion import to_nuclide_library_npz
    from pycnic.library import MultiGroupLibrary

    if H5_FILE.exists():
        print(f"Using cached multigroup library {H5_FILE.name}")
        mglib = MultiGroupLibrary.load_hdf5(str(H5_FILE))
    else:
        mglib = build_multigroup_library()

    chain = build_depletion_chain()
    tracked = to_nuclide_library_npz(mglib, chain, str(NPZ_FILE), chi_from="U-235")
    print(f"Exported depletion library -> {NPZ_FILE.name}: {tracked}")


########################################################################
# Phase 2: transport-coupled burnup in discrete1
########################################################################

# Homogenized research-reactor-style (MTR) core: 19.75 w/o LEU dispersed in
# light water (H/U = 15), plus a pure light-water reflector, in atoms/(barn
# cm). The high enrichment/dilution is deliberate: PyCNiC's GROUPR decks run
# at infinite dilution (sigma0 = 1e10 b), so U-238 resonance capture carries
# no self-shielding — a dilute, highly moderated core is the configuration
# for which that approximation is actually accurate. (A 3-4 w/o PWR lattice
# would need shielded resonance data to be anywhere near critical.)
N_U = 4.0e-3
ENRICH = 0.1975
N_FUEL = {
    "U235": ENRICH * N_U,
    "U238": (1.0 - ENRICH) * N_U,
    "H1": 6.0e-2,
    "O16": 3.0e-2 + 2.0 * N_U,
}
N_WATER = {"H1": 6.68e-2, "O16": 3.34e-2}

REFLECTOR = 20.0  # cm of water on each side
FUEL = 40.0  # cm of homogenized fuel
CELL = 0.5  # cm mesh
ANGLES = 8
POWER = 8.0e3  # W/cm^2 of slab cross section (~200 W/cm^3 in the fuel)
DT_DAYS = [1.0, 4.0, 25.0, 30.0, 30.0, 30.0, 30.0]  # short steps resolve Xe


def burnup_problem(library):
    """Run the slab burnup and return (days, density_history, keff_history)."""
    import discrete1
    from discrete1 import burnup1d

    length = FUEL + 2 * REFLECTOR
    cells = int(round(length / CELL))
    edges = np.linspace(0.0, length, cells + 1)
    centers = 0.5 * (edges[1:] + edges[:-1])
    delta_x = np.diff(edges)
    # Region 0 = fuel, region 1 = water reflector
    medium_map = np.where(
        (centers > REFLECTOR) & (centers < REFLECTOR + FUEL), 0, 1
    ).astype(np.int32)

    bc_x = [0, 0]
    angle_x, angle_w = discrete1.angular_x(ANGLES, bc_x)

    densities = np.zeros((2, library.n_nuclides))
    for name, value in N_FUEL.items():
        densities[0, library.index[name]] = value
    for name, value in N_WATER.items():
        densities[1, library.index[name]] = value

    dt_steps = np.array(DT_DAYS) * DAY
    history, keff = burnup1d.burnup(
        library,
        densities,
        medium_map,
        delta_x,
        angle_x,
        angle_w,
        bc_x,
        dt_steps,
        power=POWER,
        geometry=1,
        order=48,
        predictor_corrector=True,
    )
    days = np.concatenate(([0.0], np.cumsum(DT_DAYS)))
    return days, history, keff


def report(library, days, history, keff):
    """Print the burnup step table (fuel-region densities)."""
    i = library.index
    print("\n  day     keff     N(U-235)     N(Pu-239)    N(Xe-135)    N(Sm-149)")
    print("  " + "-" * 68)
    for s, day in enumerate(days):
        fuel = history[s, 0]
        print(
            f"  {day:5.0f}  {keff[s]:.5f}  {fuel[i['U235']]:.5e}"
            f"  {fuel[i['Pu239']]:.5e}  {fuel[i['Xe135']]:.5e}"
            f"  {fuel[i['Sm149']]:.5e}"
        )


# Fixed categorical color per nuclide (colorblind-validated ordering).
SERIES = [
    ("U235", "U-235", "#2a78d6"),
    ("Pu239", "Pu-239", "#1baf7a"),
    ("Pu240", "Pu-240", "#eda100"),
    ("Pu241", "Pu-241", "#008300"),
    ("Xe135", "Xe-135", "#4a3aa7"),
    ("Sm149", "Sm-149", "#e34948"),
    ("I135", "I-135", "#e87ba4"),
    ("Np239", "Np-239", "#eb6834"),
]


def plot(library, days, history, keff):
    """Save a two-panel figure: keff vs time and fuel nuclide densities."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax in (ax1, ax2):
        ax.grid(color="0.92", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_xlabel("Time (days)")

    ax1.plot(days, keff, color="#2a78d6", linewidth=2, marker="o", markersize=5)
    ax1.set_ylabel("k-effective")
    ax1.set_title("Reactivity over burnup", loc="left")

    for name, label, color in SERIES:
        ax2.plot(
            days,
            history[:, 0, library.index[name]],
            color=color,
            linewidth=2,
            label=label,
        )
    ax2.set_yscale("log")
    ax2.set_ylabel("Number density (atoms/b-cm)")
    ax2.set_title("Fuel-region nuclide inventory", loc="left")
    ax2.legend(frameon=False, fontsize=8, ncol=2)

    fig.suptitle(
        "LEU slab burnup: ENDF/B-VIII.0 -> NJOY IGN 3 (PyCNiC) -> discrete1",
        x=0.01,
        ha="left",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(PNG_FILE, dpi=150)
    print(f"Saved figure -> {PNG_FILE.name}")


def run_burnup():
    """Phase 2: load the exported library and burn the slab."""
    from discrete1.nuclides import load_library

    library = load_library(str(NPZ_FILE))
    print(
        f"Loaded {NPZ_FILE.name}: {library.n_nuclides} nuclides, "
        f"{library.groups} groups"
    )
    days, history, keff = burnup_problem(library)
    report(library, days, history, keff)
    plot(library, days, history, keff)


def main():
    """Regenerate missing artifacts, then run the burnup if discrete1 exists."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--generate",
        action="store_true",
        help="force phase 1 (NJOY library generation + npz export)",
    )
    args = parser.parse_args()

    if args.generate or not NPZ_FILE.exists():
        generate()

    try:
        import discrete1  # noqa: F401
    except ImportError:
        print("discrete1 not installed here; npz ready — run phase 2 locally.")
        return
    run_burnup()


if __name__ == "__main__":
    main()
