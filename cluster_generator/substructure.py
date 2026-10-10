"""
Tools for host+subhalo composite cluster systems.

Provides two capabilities:

* **Task 1 – pressure-matched truncation** (:func:`find_pressure_truncation_radius`,
  :func:`find_tidal_truncation_radius`): locate the radius at which a subhalo's
  pressure profile crosses the local host ICM pressure.

* **Task 3 – composite field evaluator** (:class:`CompositeCluster`): a
  callable ``f(x, y, z) → {density, pressure, velocity, subhalo_id}`` that
  replaces host gas with subhalo gas inside each truncation sphere and blends
  the boundary with a tanh profile.  Particle sampling and AMR grid export
  build on top of this evaluator.

The helper :func:`combined_dm_density` constructs a smooth total-DM-density
callable (host NFW + spherically averaged subhalo catalog) for use as the
``total_density`` argument of
:meth:`~cluster_generator.model.ClusterModel.from_dens_and_tden`, implementing
the option-2 mass-bookkeeping approach: the host HSE is solved in the combined
smooth potential *before* gas replacement is applied.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from unyt import unyt_array, unyt_quantity

from cluster_generator.model import ClusterModel
from cluster_generator.utils import mylog

__all__ = [
    "find_pressure_truncation_radius",
    "find_tidal_truncation_radius",
    "combined_dm_density",
    "CompositeCluster",
]

# ---------------------------------------------------------------------------
# Task 1: truncation-radius helpers
# ---------------------------------------------------------------------------


def find_pressure_truncation_radius(
    subhalo: ClusterModel,
    host: ClusterModel,
    position: np.ndarray,
) -> unyt_quantity:
    r"""
    Find the radius at which the subhalo pressure profile crosses the local
    host ICM pressure.

    The host pressure is evaluated at ``|position|`` (the subhalo center's
    host-centric radius).  The truncation radius is the innermost radius
    ``r`` where ``P_sub(r) = P_host(|position|)``.

    Parameters
    ----------
    subhalo : ClusterModel
        Subhalo model with a ``"pressure"`` field.
    host : ClusterModel
        Host cluster model with a ``"pressure"`` field.
    position : array-like, shape (3,)
        Subhalo center in kpc, relative to the host center.

    Returns
    -------
    r_trunc : unyt_quantity
        Pressure-matched truncation radius in kpc.

    Raises
    ------
    ValueError
        If the subhalo central pressure is at or below the local host
        pressure — the subhalo gas would be compressed everywhere and no
        truncation radius is defined.
    RuntimeError
        If spline root-finding fails despite the pressure difference
        bracketing zero (try increasing ``num_points`` in the subhalo
        model).

    Notes
    -----
    The host pressure varies across the subhalo face, so the match is
    exact at one point only (the subhalo center projected onto the host
    pressure profile at ``|position|``).  The tanh blend in
    :class:`CompositeCluster` removes the resulting pressure kink over
    a user-specified scale.
    """
    position = np.asarray(position, dtype=float)
    r_host = unyt_quantity(float(np.sqrt(np.dot(position, position))), "kpc")

    P_host_local = float(host.find_field_at_radius("pressure", r_host).in_units("Msun/kpc/Myr**2").v)

    r_sub = subhalo["radius"].in_units("kpc").v
    P_sub = subhalo["pressure"].in_units("Msun/kpc/Myr**2").v
    dP = P_sub - P_host_local

    if dP[0] <= 0.0:
        raise ValueError(
            f"Subhalo central pressure ({P_sub[0]:.3e} Msun/kpc/Myr²) is at or "
            f"below the local host ICM pressure ({P_host_local:.3e} Msun/kpc/Myr²) "
            f"at |position| = {float(r_host.v):.1f} kpc.  The subhalo gas is everywhere "
            "under-pressured; no truncation radius is defined."
        )

    if dP[-1] >= 0.0:
        mylog.warning(
            "Subhalo pressure stays above the host ICM throughout the model grid. "
            "Returning the outer grid edge (%.3g kpc) as the truncation radius.",
            r_sub[-1],
        )
        return unyt_quantity(r_sub[-1], "kpc")

    spl = InterpolatedUnivariateSpline(r_sub, dP, k=3, ext=3)
    roots = spl.roots()
    if len(roots) == 0:
        raise RuntimeError(
            "Spline root-finding found no crossing despite dP changing sign. "
            "Try increasing num_points in the subhalo ClusterModel."
        )

    return unyt_quantity(float(roots[0]), "kpc")


def find_tidal_truncation_radius(
    subhalo: ClusterModel,
    host: ClusterModel,
    position: np.ndarray,
) -> unyt_quantity:
    r"""
    Estimate the tidal (Jacobi) radius of the subhalo.

    Parameters
    ----------
    subhalo : ClusterModel
    host : ClusterModel
    position : array-like, shape (3,)
        Subhalo center in kpc (host-centric).

    Returns
    -------
    r_tidal : unyt_quantity
        Tidal radius in kpc.

    Notes
    -----
    Uses the Jacobi approximation for a circular orbit:

    .. math::

        r_J = d \left(\frac{m_{\rm sub}}{3\,M_{\rm host}(<d)}\right)^{1/3}

    where :math:`d = |\mathbf{position}|`.  Valid when the subhalo mass
    is sub-dominant and the orbit is nearly circular.
    """
    position = np.asarray(position, dtype=float)
    d = unyt_quantity(float(np.sqrt(np.dot(position, position))), "kpc")

    M_host_d = float(host.find_field_at_radius("total_mass", d).in_units("Msun").v)
    M_sub = float(subhalo["total_mass"].in_units("Msun").v[-1])

    r_tidal = float(d.v) * (M_sub / (3.0 * M_host_d)) ** (1.0 / 3.0)
    return unyt_quantity(r_tidal, "kpc")


# ---------------------------------------------------------------------------
# Option-2 mass bookkeeping: combined host+subhalo DM density
# ---------------------------------------------------------------------------


def combined_dm_density(
    host_dm_density,
    subhalo_masses: np.ndarray,
    subhalo_positions: np.ndarray,
    smoothing_scale: float = 0.05,
    min_smoothing: float = 10.0,
):
    r"""
    Build a callable total DM density for
    :meth:`~cluster_generator.model.ClusterModel.from_dens_and_tden`.

    The subhalo catalog is incorporated via spherical averaging: each
    subhalo at host-centric radius :math:`r_i` is smeared into a
    Gaussian shell of width
    :math:`\sigma_i = \max(\texttt{smoothing\_scale}\cdot r_i,
    \texttt{min\_smoothing})` kpc, then divided by :math:`4\pi r^2` to
    convert shell mass to volume density.  Integrating over all :math:`r`
    recovers :math:`m_i` to within the Gaussian tail correction
    (negligible for :math:`r_i \gg \sigma_i`).

    The returned callable should be passed as ``total_density`` to
    :meth:`~cluster_generator.model.ClusterModel.from_dens_and_tden` so
    that the host HSE is solved in the combined smooth potential.

    Parameters
    ----------
    host_dm_density : callable
        Host DM density in Msun kpc⁻³ evaluated at an array of radii in
        kpc (e.g. ``nfw_density_profile(rho_s, r_s)``).
    subhalo_masses : array-like, shape (N,)
        Subhalo masses in Msun.
    subhalo_positions : array-like, shape (N, 3)
        Subhalo centers in kpc (host-centric).
    smoothing_scale : float, optional
        Gaussian width as a fraction of each subhalo's host-centric
        radius.  Default 0.05.
    min_smoothing : float, optional
        Minimum smoothing width in kpc.  Default 10.0.

    Returns
    -------
    total_density : callable
        ``total_density(r)`` → ndarray in Msun kpc⁻³.
    """
    masses = np.asarray(subhalo_masses, dtype=float)
    positions = np.atleast_2d(np.asarray(subhalo_positions, dtype=float))
    r_catalog = np.sqrt(np.sum(positions**2, axis=1))  # (N,)
    sigma = np.maximum(smoothing_scale * r_catalog, min_smoothing)  # (N,)

    def total_density(r):
        r = np.atleast_1d(np.asarray(r, dtype=float))
        rho = np.asarray(host_dm_density(r), dtype=float).copy()
        # Gaussian shell contribution from each subhalo — vectorised (N_sub, N_r)
        dr = r[np.newaxis, :] - r_catalog[:, np.newaxis]
        kernel = np.exp(-0.5 * (dr / sigma[:, np.newaxis]) ** 2) / (
            sigma[:, np.newaxis] * np.sqrt(2.0 * np.pi)
        )
        rho_sub = (masses[:, np.newaxis] * kernel / (4.0 * np.pi * r[np.newaxis, :] ** 2)).sum(axis=0)
        return rho + rho_sub

    return total_density


# ---------------------------------------------------------------------------
# Task 3: composite field evaluator
# ---------------------------------------------------------------------------


class CompositeCluster:
    r"""
    Vectorized host+subhalo composite field evaluator.

    Evaluates ``(density, pressure, velocity, subhalo_id)`` at arbitrary
    3-D positions without any commitment to a particle layout or grid.
    Inside each subhalo's truncation sphere the subhalo gas *replaces*
    the host gas; the boundary is blended with a tanh profile of
    characteristic width ``blend_width``.  Where subhalos overlap, an
    arbitration rule selects the winner.

    Particle sampling and AMR grid export build on top by calling
    :meth:`__call__` on their own position arrays.

    Parameters
    ----------
    host : ClusterModel
        Host cluster model (must have ``"density"`` and ``"pressure"``).
    host_center : array-like, shape (3,)
        Host center in kpc.
    subhalos : list of ClusterModel
        Subhalo models, each with ``"density"`` and ``"pressure"``.
    subhalo_positions : list of array-like, each shape (3,)
        Subhalo centers in kpc (same frame as ``host_center``).
    truncation_radii : array-like, shape (N_sub,)
        Pressure-matched truncation radii in kpc; use
        :func:`find_pressure_truncation_radius` to compute them.
    host_velocity : array-like, shape (3,), optional
        Host bulk velocity in kpc Myr⁻¹.  Default zero.
    subhalo_velocities : list of array-like, each shape (3,), optional
        Subhalo bulk velocities in kpc Myr⁻¹.  Default zeros.
    blend_width : float, optional
        Characteristic width of the tanh boundary layer in kpc.
        Default 50 kpc.  Smaller values give a sharper edge.
    overlap_rule : {"deepest_potential", "nearest_normalized"}, optional
        Arbitration rule when multiple subhalos have non-trivial weight
        at the same point.

        ``"deepest_potential"`` — picks the subhalo whose gravitational
        potential is most negative at the query point (requires
        ``"gravitational_potential"`` in each subhalo model; falls back
        to ``"nearest_normalized"`` if absent).

        ``"nearest_normalized"`` — picks the subhalo with the smallest
        :math:`r_i / r_{{\rm trunc},i}`.

        Default ``"deepest_potential"``.

    Notes
    -----
    **Pressure continuity**: by construction, when a subhalo's truncation
    radius was found with :func:`find_pressure_truncation_radius`, the
    subhalo and host pressures are equal at :math:`r_{{\rm trunc}}`.  The
    tanh blend therefore keeps the pressure field continuous at the
    boundary regardless of ``blend_width``.

    **Density**: density is *not* continuous in general — the blend
    transitions between two different density profiles over the blend
    layer.

    **Weight convention**: the tanh weight for subhalo *i* at distance
    :math:`r_i` from its center is

    .. math::

        w_i = \tfrac{1}{2}\!\left(1 - \tanh\!\frac{r_i - r_{{\rm trunc},i}}{b}\right)

    so :math:`w_i \to 1` deep inside and :math:`w_i \to 0` far outside.
    The blended field is :math:`(1 - w_i)\,f_{\rm host} + w_i\,f_{\rm sub}`.
    """

    _OVERLAP_RULES = frozenset({"deepest_potential", "nearest_normalized"})

    def __init__(
        self,
        host: ClusterModel,
        host_center,
        subhalos: list,
        subhalo_positions: list,
        truncation_radii,
        *,
        host_velocity=None,
        subhalo_velocities=None,
        blend_width: float = 50.0,
        overlap_rule: str = "deepest_potential",
    ):
        if overlap_rule not in self._OVERLAP_RULES:
            raise ValueError(
                f"overlap_rule must be one of {sorted(self._OVERLAP_RULES)}, got {overlap_rule!r}"
            )

        self.host = host
        self.host_center = np.asarray(host_center, dtype=float)
        self.host_velocity = np.zeros(3) if host_velocity is None else np.asarray(host_velocity, dtype=float)
        self.subhalos = list(subhalos)
        self.subhalo_positions = [np.asarray(p, dtype=float) for p in subhalo_positions]
        self.truncation_radii = np.asarray(truncation_radii, dtype=float)
        self.blend_width = float(blend_width)
        self.overlap_rule = overlap_rule

        n_sub = len(self.subhalos)
        if subhalo_velocities is None:
            self.subhalo_velocities = [np.zeros(3) for _ in range(n_sub)]
        else:
            self.subhalo_velocities = [np.asarray(v, dtype=float) for v in subhalo_velocities]

        if len(self.subhalo_positions) != n_sub:
            raise ValueError("len(subhalo_positions) must equal len(subhalos)")
        if len(self.truncation_radii) != n_sub:
            raise ValueError("len(truncation_radii) must equal len(subhalos)")

        self._host_splines = self._build_splines(host)
        self._sub_splines = [self._build_splines(s) for s in self.subhalos]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_splines(model: ClusterModel) -> dict:
        r = model["radius"].in_units("kpc").v
        splines = {}
        for field in ("density", "pressure"):
            if field in model:
                splines[field] = InterpolatedUnivariateSpline(r, model[field].v, k=3, ext=3)
        if "gravitational_potential" in model:
            splines["gravitational_potential"] = InterpolatedUnivariateSpline(
                r, model["gravitational_potential"].v, k=3, ext=3
            )
        return splines

    def _tanh_weights(self, r_sub: np.ndarray) -> np.ndarray:
        """Shape (N_sub, N_pts); 1 = deep inside subhalo, 0 = outside."""
        return 0.5 * (1.0 - np.tanh((r_sub - self.truncation_radii[:, np.newaxis]) / self.blend_width))

    def _arbitrate(self, r_sub: np.ndarray, w: np.ndarray) -> np.ndarray:
        """
        Return winner index per point; -1 means host dominates.

        Parameters
        ----------
        r_sub : (N_sub, N_pts)
        w     : (N_sub, N_pts) tanh weights
        """
        N_pts = w.shape[1]
        winner = np.full(N_pts, -1, dtype=np.intp)
        n_sub = len(self.subhalos)

        if n_sub == 0:
            return winner

        # Points where at least one subhalo has a non-trivial weight
        active = np.max(w, axis=0) > 0.01
        if not np.any(active):
            return winner

        if self.overlap_rule == "nearest_normalized":
            normalized = r_sub / self.truncation_radii[:, np.newaxis]
            winner[active] = np.argmin(normalized[:, active], axis=0)

        else:  # deepest_potential
            phi = np.empty((n_sub, N_pts))
            use_potential = False
            for i, sp in enumerate(self._sub_splines):
                if "gravitational_potential" in sp:
                    phi[i, active] = sp["gravitational_potential"](r_sub[i, active])
                    use_potential = True
                else:
                    # per-subhalo fallback: treat missing Φ as normalized distance
                    phi[i] = r_sub[i] / self.truncation_radii[i]
            if not use_potential:
                mylog.warning(
                    "No subhalo model has a gravitational_potential field; "
                    "falling back to nearest_normalized arbitration."
                )
            # Most negative Φ wins (or smallest normalized distance in fallback)
            winner[active] = np.argmin(phi[:, active], axis=0)

        return winner

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __call__(self, x, y, z) -> dict:
        """
        Evaluate composite gas fields at positions (x, y, z).

        Parameters
        ----------
        x, y, z : array-like or unyt_array
            Positions in kpc.  Scalars are promoted to 1-element arrays.

        Returns
        -------
        dict
            ``"density"``   : unyt_array, Msun kpc⁻³

            ``"pressure"``  : unyt_array, Msun kpc⁻¹ Myr⁻²

            ``"velocity"``  : unyt_array, shape (..., 3), kpc Myr⁻¹

            ``"subhalo_id"``: int ndarray, shape (...,), -1 = host
        """

        def _strip(a):
            if hasattr(a, "in_units"):
                return np.atleast_1d(a.in_units("kpc").v).astype(float)
            return np.atleast_1d(np.asarray(a, dtype=float))

        x, y, z = _strip(x), _strip(y), _strip(z)
        pos = np.stack([x, y, z], axis=1)  # (N_pts, 3)
        N_pts = len(x)
        n_sub = len(self.subhalos)

        r_host = np.linalg.norm(pos - self.host_center, axis=1)

        r_sub = np.empty((n_sub, N_pts))
        for i, sp in enumerate(self.subhalo_positions):
            r_sub[i] = np.linalg.norm(pos - sp, axis=1)

        w = self._tanh_weights(r_sub)  # (N_sub, N_pts)
        winner = self._arbitrate(r_sub, w)  # (N_pts,)

        # Start from host values at every point
        dens = self._host_splines["density"](r_host).copy()
        pres = self._host_splines["pressure"](r_host).copy()
        vel = np.tile(self.host_velocity, (N_pts, 1)).astype(float)
        subhalo_id = np.full(N_pts, -1, dtype=int)

        for i in range(n_sub):
            mask = winner == i
            if not np.any(mask):
                continue
            wi = w[i, mask]
            r_si = r_sub[i, mask]
            d_sub = self._sub_splines[i]["density"](r_si)
            p_sub = self._sub_splines[i]["pressure"](r_si)
            dens[mask] = (1.0 - wi) * dens[mask] + wi * d_sub
            pres[mask] = (1.0 - wi) * pres[mask] + wi * p_sub
            vel[mask] = (1.0 - wi[:, np.newaxis]) * vel[mask] + wi[:, np.newaxis] * self.subhalo_velocities[i]
            subhalo_id[mask] = i

        return {
            "density": unyt_array(dens, self.host["density"].units),
            "pressure": unyt_array(pres, self.host["pressure"].units),
            "velocity": unyt_array(vel, "kpc/Myr"),
            "subhalo_id": subhalo_id,
        }
