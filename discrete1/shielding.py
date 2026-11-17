"""Bondarenko self-shielding as a burnup cross-section update.

A library shields its resonances at the background cross sections of the
composition it was made for. As the fuel burns those backgrounds change:
Pu-239 and Pu-240 build up from trace amounts, where infinite dilution holds,
to densities where their 0.3 and 1 eV resonances shield strongly.
:func:`bondarenko_update` turns Bondarenko tables into an ``xs_update`` for
:func:`discrete1.burnup1d.burnup`, re-shielding each region from its own
densities before every transport solve and burnup matrix.

Table layout
------------
Read from the ``shield_*`` keys of a nuclide-library ``.npz``, e.g. from NJOY
GROUPR run at several background cross sections. With ``M`` nuclides
(``names``), ``K`` of them tabled, ``G`` groups and ``S`` finite backgrounds:

- ``shield_sigma0`` ``(S,)``: background cross sections (barns), ascending;
  ``shield_sigma0_infinite``: the infinite-dilution node appended to them.
- ``shield_index`` ``(K,)``: indices into ``names`` of the tabled nuclides.
- ``shield_inf_total``, ``shield_inf_fission``, ``shield_inf_nu_fission``,
  ``shield_inf_capture`` ``(K, G)`` and ``shield_inf_scatter`` ``(K, G, G)``
  (``[g_out, g_in]``): the infinite-dilution data the factors multiply.
- ``shield_f_total``, ``shield_f_fission``, ``shield_f_capture`` ``(K, G, S)``:
  shielded over infinite-dilution ratios (``nu_fission`` takes the fission
  factor; the ``(n,gamma)`` channel takes the capture factor).
- ``shield_scatter_k``, ``shield_scatter_g`` ``(n,)`` and
  ``shield_scatter_delta`` ``(n, G, S)``: shielded minus infinite-dilution
  scattering into each outgoing group from incident group ``g`` of tabled
  nuclide ``k``.
- ``shield_sigma_p`` ``(M,)``: potential scattering (barns) per nuclide.
- ``shield_escape``: lattice escape cross section (1/cm) for fuel regions.

The background of tabled nuclide ``i`` in a region is
``(sum_{j != i} N_j sigma_p,j + Sigma_e) / N_i`` (infinite where ``N_i`` is
zero), and factors and differences are interpolated linearly in
``log(sigma0)`` over the grid plus the infinite-dilution node (factor 1,
difference 0), held flat below the grid.
"""

import dataclasses

import numpy as np

__all__ = ["bondarenko_update"]

_INF_KEYS = (
    "shield_inf_total",
    "shield_inf_fission",
    "shield_inf_nu_fission",
    "shield_inf_capture",
    "shield_inf_scatter",
)


class _BondarenkoUpdate:
    """``xs_update`` callable re-shielding the tabled nuclides of a library."""

    def __init__(self, source, escape=None):
        if isinstance(source, (str, bytes)) or hasattr(source, "__fspath__"):
            source = np.load(source)
        data = source
        self.names = [str(name) for name in data["names"]]
        self.grid = np.log(
            np.append(
                np.asarray(data["shield_sigma0"], dtype=np.float64),
                float(data["shield_sigma0_infinite"]),
            )
        )
        self.tabled = [self.names[i] for i in np.asarray(data["shield_index"])]
        self.sigma_p = dict(
            zip(self.names, np.asarray(data["shield_sigma_p"], dtype=np.float64))
        )
        self.inf = {key: np.asarray(data[key], dtype=np.float64) for key in _INF_KEYS}
        self.factors = {
            key: _append(np.asarray(data[f"shield_f_{key}"], dtype=np.float64), 1.0)
            for key in ("total", "fission", "capture")
        }
        self.scatter_k = np.asarray(data["shield_scatter_k"], dtype=np.int64)
        self.scatter_g = np.asarray(data["shield_scatter_g"], dtype=np.int64)
        self.scatter_delta = _append(
            np.asarray(data["shield_scatter_delta"], dtype=np.float64), 0.0
        )
        stored = float(data["shield_escape"])
        self.escape = np.atleast_1d(
            np.asarray(stored if escape is None else escape, dtype=np.float64)
        )

        if np.any(np.diff(self.grid) <= 0.0):
            raise ValueError(
                "shield_sigma0 must be ascending and below shield_sigma0_infinite"
            )
        shape = (len(self.tabled), self.inf["shield_inf_total"].shape[1])
        for key, table in self.factors.items():
            if table.shape != shape + (self.grid.size,):
                raise ValueError(f"shield_f_{key} must have shape (K, G, S)")
        if self.scatter_delta.shape[1:] != (shape[1], self.grid.size):
            raise ValueError("shield_scatter_delta must have shape (n, G, S)")
        if np.any(self.escape < 0.0):
            raise ValueError("escape must be non-negative")

    def _escape(self, region):
        """Return the escape term for ``region``: one shared value, or its own."""
        return float(self.escape[0] if self.escape.size == 1 else self.escape[region])

    def _weights(self, sigma0):
        """Interpolation nodes and weights for each background."""
        with np.errstate(divide="ignore"):
            x = np.clip(np.log(sigma0), self.grid[0], self.grid[-1])
        upper = np.clip(np.searchsorted(self.grid, x), 1, self.grid.size - 1)
        lower = upper - 1
        weight = (x - self.grid[lower]) / (self.grid[upper] - self.grid[lower])
        return lower, upper, np.where(x <= self.grid[0], 0.0, weight)

    def __call__(self, library, densities, region):
        present = [name for name in self.tabled if name in library.index]
        if not present or self.grid.size < 2:
            return library
        rows = np.array([self.tabled.index(name) for name in present])
        at = np.array([library.index[name] for name in present])

        densities = np.asarray(densities, dtype=np.float64)
        sigma_p = np.array([self.sigma_p.get(name, 0.0) for name in library.names])
        others = densities @ sigma_p - densities[at] * sigma_p[at]
        others = np.maximum(others, 0.0) + self._escape(region)
        # With no other scatterers and no escape, sigma0 = 0 (fully shielded)
        with np.errstate(divide="ignore", invalid="ignore"):
            sigma0 = np.where(densities[at] > 0.0, others / densities[at], np.inf)
        lower, upper, weight = self._weights(sigma0)

        def factor(key):
            table = self.factors[key][rows]
            k = np.arange(rows.size)
            return (1.0 - weight)[:, None] * table[k, :, lower] + weight[
                :, None
            ] * table[k, :, upper]

        f_fission = factor("fission")
        xs_total = library.xs_total.copy()
        fission_xs = library.fission_xs.copy()
        nu_fission = library.nu_fission.copy()
        xs_scatter = library.xs_scatter.copy()
        reaction_xs = dict(library.reaction_xs)
        xs_total[at] = self.inf["shield_inf_total"][rows] * factor("total")
        fission_xs[at] = self.inf["shield_inf_fission"][rows] * f_fission
        nu_fission[at] = self.inf["shield_inf_nu_fission"][rows] * f_fission
        if "(n,gamma)" in reaction_xs:
            capture = reaction_xs["(n,gamma)"].copy()
            capture[at] = self.inf["shield_inf_capture"][rows] * factor("capture")
            reaction_xs["(n,gamma)"] = capture

        scatter = self.inf["shield_inf_scatter"][rows].copy()
        for slot, row in enumerate(rows):
            mine = np.nonzero(self.scatter_k == row)[0]
            if mine.size == 0:
                continue
            delta = self.scatter_delta[mine]
            n = np.arange(mine.size)
            change = (1.0 - weight[slot]) * delta[n, :, lower[slot]] + weight[
                slot
            ] * delta[n, :, upper[slot]]
            # Column g of [g_out, g_in]: scattering out of incident group g.
            scatter[slot][:, self.scatter_g[mine]] += change.T
        xs_scatter[at] = scatter

        return dataclasses.replace(
            library,
            xs_total=xs_total,
            fission_xs=fission_xs,
            nu_fission=nu_fission,
            xs_scatter=xs_scatter,
            reaction_xs=reaction_xs,
        )


def _append(table, value):
    """``table`` with a constant column appended on its last axis."""
    tail = np.full(table.shape[:-1] + (1,), value)
    return np.concatenate([table, tail], axis=-1)


def bondarenko_update(source, escape=None):
    """Build an ``xs_update`` that re-shields tabled nuclides per region.

    Pass the result to :func:`discrete1.burnup1d.burnup` or
    :func:`discrete1.burnup1d.ml_burnup` as ``xs_update``.

    Parameters
    ----------
    source : str, path-like or mapping
        A nuclide-library ``.npz`` (or its loaded contents) carrying the
        ``shield_*`` keys described in this module.
    escape : float or array_like, optional
        Escape cross section (1/cm) added to every background, one value or
        one per region (0 for regions outside the fuel). None (default) takes
        the stored ``shield_escape`` for every region.

    Returns
    -------
    callable
        ``fn(library, densities, region) -> NuclideLibrary``. Nuclides are
        matched by name, so it also works on a ``library.subset(...)``;
        tabled nuclides the library does not hold are skipped, and a library
        with none of them is returned as is. Untabled nuclides and channels
        other than ``(n,gamma)`` keep the library's values.

    Examples
    --------
    >>> library = load_library("fuel.npz")
    >>> history, keff = burnup(library, ..., xs_update=bondarenko_update(
    ...     "fuel.npz", escape=[0.48, 0.0]))
    """
    return _BondarenkoUpdate(source, escape)
