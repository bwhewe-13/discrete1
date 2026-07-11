"""Transport-coupled burnup / depletion driver for 1D problems.

This module couples the neutron transport solvers to the depletion engine to
evolve material composition over a sequence of burnup steps. Each depletable
region (one per ``medium_map`` material index) carries its own vector of
nuclide number densities; as those densities change, macroscopic cross
sections are rebuilt and the transport problem is re-solved.

The default inner solve is a k-eigenvalue power iteration
(:func:`discrete1.critical1d.power_iteration`); the flux is renormalized to a
target fission power (or to a specified flux level for activation studies)
before reaction rates are computed. Depletion of each region then uses CRAM
(:func:`discrete1.depletion.deplete`).

Time integration uses a predictor-corrector scheme defaulting to **CE/LI**
(Constant-Extrapolation predictor / Linear-Interpolation corrector), mirroring
the Serpent ``set pcc`` default:

1. Predictor (CE): build the burnup matrix from the beginning-of-step flux,
   hold it constant over the step, and deplete.
2. Corrector (LI): rebuild macroscopic cross sections from the predicted
   composition, re-solve transport for the end-of-step flux, and re-deplete
   with the step-averaged burnup matrix.

Set ``predictor_corrector=False`` for the cheaper predictor-only (CE) scheme.

Unit convention (see :mod:`discrete1.depletion`): number densities in
atoms/(barn*cm), microscopic cross sections in barns, flux in n/cm^2/s, so
macroscopic cross sections come out in 1/cm and reaction rates in 1/s.
"""

import numpy as np
from tqdm import tqdm

from discrete1 import cram, critical1d
from discrete1.depletion import build_burnup_matrix

__all__ = ["macroscopic_xs", "region_flux", "burnup"]


def macroscopic_xs(library, densities):
    """Build macroscopic cross sections per region from number densities.

    Macroscopic cross section ``Sigma = sum_m N_m * sigma_m`` for each region,
    using the microscopic transport data on the library. With densities in
    atoms/(barn*cm) and microscopic cross sections in barns, the result is in
    1/cm.

    Parameters
    ----------
    library : NuclideLibrary
        Source of microscopic transport cross sections.
    densities : numpy.ndarray, shape (R, M)
        Number densities for each of ``R`` regions and ``M`` nuclides.

    Returns
    -------
    xs_total : numpy.ndarray, shape (R, G)
        Macroscopic total cross section.
    xs_scatter : numpy.ndarray, shape (R, G, G)
        Macroscopic scattering matrix.
    nu_fission : numpy.ndarray, shape (R, G)
        Macroscopic ``nu * Sigma_f`` (pair with ``library.chi``).
    """
    xs_total = densities @ library.xs_total
    nu_fission = densities @ library.nu_fission
    xs_scatter = np.einsum("rm,mgh->rgh", densities, library.xs_scatter)
    return xs_total, xs_scatter, nu_fission


def _cell_volumes(delta_x, geometry):
    """Cell volume weights: widths for slabs, shell volumes for spheres.

    Matches the sphere sweep convention (cell ``i`` spans radii built by
    accumulating ``delta_x``): ``V_i = 4/3 pi (r_{i+1}^3 - r_i^3)``.
    """
    if geometry == 2:
        edges = np.concatenate(([0.0], np.cumsum(delta_x)))
        return 4.0 / 3.0 * np.pi * np.diff(edges**3)
    return np.asarray(delta_x, dtype=np.float64)


def region_flux(flux, medium_map, delta_x, n_regions):
    """Volume-average the cell flux over the cells of each region.

    Parameters
    ----------
    flux : numpy.ndarray, shape (cells_x, G)
        Cell-centered scalar flux.
    medium_map : numpy.ndarray, shape (cells_x,)
        Region (material) index per cell.
    delta_x : numpy.ndarray, shape (cells_x,)
        Cell volume weights: cell widths in slab geometry, shell volumes
        (``4/3 pi (r_out^3 - r_in^3)``) in spherical geometry.
    n_regions : int
        Number of regions ``R``.

    Returns
    -------
    numpy.ndarray, shape (R, G)
        Volume-averaged flux per region. Regions with no cells get zero.
    """
    groups = flux.shape[1]
    rflux = np.zeros((n_regions, groups))
    volume = np.zeros(n_regions)
    np.add.at(rflux, medium_map, delta_x[:, None] * flux)
    np.add.at(volume, medium_map, delta_x)
    nonzero = volume > 0.0
    rflux[nonzero] /= volume[nonzero, None]
    return rflux


def _power_density(flux, densities, library, medium_map, volumes):
    """Total fission power for the current flux and composition.

    ``P = sum_cells V_c * sum_g (sum_m N_m kappa_m sigma_f_mg) phi_cg``.
    The barn/cm^3 unit factors cancel under the atoms/(barn*cm) convention,
    so no explicit conversion constant is needed.
    """
    # kappa-fission macroscopic cross section per region (R, G)
    kappa_fission = (densities * library.kappa[None, :]) @ library.fission_xs
    return np.sum(volumes[:, None] * kappa_fission[medium_map] * flux)


def _normalize(flux, densities, library, medium_map, volumes, power, flux_level):
    """Scale the eigenvector flux to a target power or flux level."""
    if power is not None:
        total = _power_density(flux, densities, library, medium_map, volumes)
        if total <= 0.0:
            raise ValueError("Cannot power-normalize: zero fission power in system.")
        return flux * (power / total)
    if flux_level is not None:
        # Scale so the volume-averaged total (group-summed) flux matches.
        total_flux = np.sum(flux, axis=1)
        average = np.sum(total_flux * volumes) / np.sum(volumes)
        if average <= 0.0:
            raise ValueError("Cannot flux-normalize: zero average flux.")
        return flux * (flux_level / average)
    raise ValueError("Provide exactly one of `power` or `flux_level`.")


def _solve_transport(
    library,
    densities,
    medium_map,
    delta_x,
    volumes,
    angle_x,
    angle_w,
    bc_x,
    geometry,
    power,
    flux_level,
):
    """Rebuild macroscopic xs, run power iteration, and normalize the flux."""
    xs_total, xs_scatter, nu_fission = macroscopic_xs(library, densities)
    # power_iteration expects chi shaped (materials, groups); the emission
    # spectrum is shared across regions, so tile it per region (writable copy
    # required by the numba kernel signature).
    chi = np.tile(library.chi, (densities.shape[0], 1))
    flux, keff = critical1d.power_iteration(
        xs_total,
        xs_scatter,
        nu_fission,
        medium_map,
        delta_x,
        angle_x,
        angle_w,
        bc_x,
        chi=chi,
        geometry=geometry,
    )
    flux = _normalize(flux, densities, library, medium_map, volumes, power, flux_level)
    return flux, keff


def burnup(
    library,
    densities,
    medium_map,
    delta_x,
    angle_x,
    angle_w,
    bc_x,
    dt_steps,
    power=None,
    flux_level=None,
    geometry=1,
    order=48,
    substeps=1,
    predictor_corrector=True,
):
    """Run a transport-coupled burnup calculation over a sequence of steps.

    Parameters
    ----------
    library : NuclideLibrary
        Depletion and transport data.
    densities : numpy.ndarray, shape (R, M)
        Initial number densities for ``R`` regions, ``M`` nuclides, in
        atoms/(barn*cm). Region index aligns with ``medium_map`` values.
    medium_map : numpy.ndarray, shape (cells_x,)
        Region index per spatial cell.
    delta_x : numpy.ndarray, shape (cells_x,)
        Cell widths.
    angle_x, angle_w : numpy.ndarray
        Angular ordinates and weights.
    bc_x : list-like
        Boundary condition indicators [left, right].
    dt_steps : array_like, shape (n_steps,)
        Burnup step durations in seconds.
    power : float, optional
        Target total fission power (power normalization). In spherical
        geometry this is true watts; in slab geometry the transverse extent
        is implicit, so it is power per unit cross-sectional area (W/cm^2).
        Provide exactly one of ``power`` or ``flux_level``.
    flux_level : float, optional
        Target volume-averaged scalar flux in n/cm^2/s (flux normalization).
    geometry : int, optional
        Geometry selector (1=slab, 2=sphere). Default 1.
    order : int, optional
        CRAM order (16 or 48; default 48).
    substeps : int, optional
        CRAM substeps within each predictor/corrector depletion (default 1).
    predictor_corrector : bool, optional
        If True (default) use CE/LI; if False use predictor-only CE.

    Returns
    -------
    density_history : numpy.ndarray, shape (n_steps + 1, R, M)
        Number densities at the start of each step and after the last step.
    keff_history : numpy.ndarray, shape (n_steps + 1,)
        k-effective of the composition in ``density_history[s]``; the final
        entry is the end-of-life eigenvalue after the last burnup step.

    Notes
    -----
    Tiny negative number densities from the CRAM solves are clamped to zero
    before they feed back into the macroscopic cross sections.
    """
    dt_steps = np.atleast_1d(np.asarray(dt_steps, dtype=np.float64))
    densities = np.array(densities, dtype=np.float64)
    n_regions = densities.shape[0]
    volumes = _cell_volumes(delta_x, geometry)

    density_history = np.zeros((dt_steps.shape[0] + 1, *densities.shape))
    density_history[0] = densities
    keff_history = np.zeros(dt_steps.shape[0] + 1)

    for step, dt in enumerate(tqdm(dt_steps, desc="Burnup", ascii=True)):
        # --- Beginning-of-step transport solve (predictor flux) ---
        flux0, keff0 = _solve_transport(
            library,
            densities,
            medium_map,
            delta_x,
            volumes,
            angle_x,
            angle_w,
            bc_x,
            geometry,
            power,
            flux_level,
        )
        keff_history[step] = keff0
        rflux0 = region_flux(flux0, medium_map, volumes, n_regions)

        # --- Predictor (CE): per-region burnup matrices held constant ---
        matrices0 = [build_burnup_matrix(library, rflux0[r]) for r in range(n_regions)]
        predicted = np.array(
            [
                cram.cram_expm(
                    matrices0[r], densities[r], dt, order=order, substeps=substeps
                )
                for r in range(n_regions)
            ]
        )
        predicted = np.maximum(predicted, 0.0)

        if not predictor_corrector:
            densities = predicted
            density_history[step + 1] = densities
            continue

        # --- Corrector (LI): end-of-step flux, step-averaged matrices ---
        flux1, _ = _solve_transport(
            library,
            predicted,
            medium_map,
            delta_x,
            volumes,
            angle_x,
            angle_w,
            bc_x,
            geometry,
            power,
            flux_level,
        )
        rflux1 = region_flux(flux1, medium_map, volumes, n_regions)
        corrected = np.zeros_like(densities)
        for r in range(n_regions):
            matrix1 = build_burnup_matrix(library, rflux1[r])
            avg = 0.5 * (matrices0[r] + matrix1)
            corrected[r] = cram.cram_expm(
                avg, densities[r], dt, order=order, substeps=substeps
            )

        densities = np.maximum(corrected, 0.0)
        density_history[step + 1] = densities

    # --- End-of-life eigenvalue for the final composition ---
    _, keff_history[-1] = _solve_transport(
        library,
        densities,
        medium_map,
        delta_x,
        volumes,
        angle_x,
        angle_w,
        bc_x,
        geometry,
        power,
        flux_level,
    )

    return density_history, keff_history
