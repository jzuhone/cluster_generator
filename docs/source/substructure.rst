.. _substructure:

Substructure
------------

Real galaxy clusters are not smooth: they host a population of smaller
dark-matter halos — **subhalos** — each carrying their own dark matter
and, at the few-percent level, their own gas. The
:py:mod:`substructure` module provides tools to build physically
consistent idealized cluster+subhalo systems and to evaluate the
composite gas fields at arbitrary 3-D positions. The resulting field
evaluator is code-independent; particle sampling and AMR grid export
build on top of it.

.. raw:: html

   <hr style="color:black">

.. contents::

.. raw:: html

   <hr style="height:10px;background-color:black">

.. _substructure_theory:

Mathematical Overview
=====================

.. _no_superposition:

Why Gas Pressure Does Not Superpose
'''''''''''''''''''''''''''''''''''''

A naive approach to including subhalo gas is to add the density profiles
of the host and the subhalo:

.. math::

    \rho_{\rm total}({\bf r}) = \rho_{\rm host}(|{\bf r}|)
                               + \rho_{\rm sub}(|{\bf r} - {\bf r}_s|),

where :math:`{\bf r}_s` is the subhalo center. This is correct for the
dark matter, which is collisionless. It is **wrong** for the gas.

Each HSE profile satisfies

.. math::

    \nabla P = \rho_g\, {\bf g}

in the gravitational field of its *own* halo. Adding two such density
fields therefore violates HSE in both halos simultaneously: the combined
pressure gradient no longer balances the combined gravitational field,
producing an overpressured region that would drive outward flows the
moment the simulation starts.

The physically correct prescription is that the subhalo gas *replaces*
the host gas inside the subhalo, with a pressure boundary condition
imposed at the interface.

.. _pressure_truncation_theory:

Pressure-Matched Truncation
''''''''''''''''''''''''''''

At position :math:`{\bf r}_s` relative to the host center, the host ICM
has a pressure :math:`P_{\rm host}(|{\bf r}_s|)`. Deep inside the
subhalo, the gas is in hydrostatic equilibrium with the subhalo's own
gravitational potential and has a centrally peaked pressure profile
:math:`P_{\rm sub}(r_{\rm sub})` (here :math:`r_{\rm sub} = |{\bf r} -
{\bf r}_s|`). The two pressures cross at the **truncation radius**
:math:`r_{\rm trunc}`, defined implicitly by

.. math::

    P_{\rm sub}(r_{\rm trunc}) = P_{\rm host}(|{\bf r}_s|).

Inside :math:`r_{\rm trunc}` the subhalo gas dominates; outside it the
host ICM dominates. This definition guarantees pressure continuity at
the interface, which is essential for numerical stability.

.. admonition:: Caveat: single-point match

    The host pressure varies across the face of the subhalo. The match
    above is exact only at :math:`|{\bf r}_s|` (the host-centric radius
    of the subhalo center). Closer to the host center the host pressure
    is higher; farther out it is lower. For subhalos whose diameter is
    small compared to the pressure scale length of the host, this is an
    excellent approximation. The tanh boundary layer (see below) removes
    the resulting thin pressure discontinuity smoothly.

In practice, :func:`~substructure.find_pressure_truncation_radius`
builds a cubic spline of :math:`P_{\rm sub}(r) - P_{\rm host}(|{\bf
r}_s|)` and returns the innermost root.

.. _tidal_truncation_theory:

Tidal Truncation
''''''''''''''''

An alternative (or supplementary) truncation criterion is the **Jacobi
radius**, the radius beyond which the host tidal field exceeds the
subhalo's self-gravity. For a circular orbit at host-centric distance
:math:`d`,

.. math::

    r_J = d\left(\frac{m_{\rm sub}}{3\,M_{\rm host}(<d)}\right)^{1/3},

where :math:`M_{\rm host}(<d)` is the host enclosed mass within
:math:`d`. This formula assumes the subhalo mass is sub-dominant and the
orbit is circular; it is provided by
:func:`~substructure.find_tidal_truncation_radius`.

.. _mass_bookkeeping_theory:

Mass Bookkeeping
'''''''''''''''''

The host HSE should be solved in the *combined* gravitational potential
of the host dark matter and all subhalos. Because subhalo positions are
distributed in 3-D, their contribution cannot be directly added to the
spherically symmetric host NFW profile. However, once the subhalo
catalog (masses :math:`m_i` at host-centric positions :math:`{\bf
r}_i`) is drawn, the **spherically averaged** subhalo DM density can be
computed.

Treating each subhalo as a point mass in a thin shell at its
host-centric radius :math:`r_i = |{\bf r}_i|`, the contribution to the
enclosed mass profile is a step function:

.. math::

    M_{\rm sub}(<r) = \sum_{r_i < r} m_i.

Rather than differentiating the step function, each subhalo's mass is
spread into a Gaussian shell of width
:math:`\sigma_i = \max(f\,r_i,\,\sigma_{\rm min})`,

.. math::

    \rho_{\rm sub}(r) =
        \sum_i \frac{m_i}{4\pi r^2}
        \frac{1}{\sigma_i\sqrt{2\pi}}
        \exp\!\left(-\frac{(r-r_i)^2}{2\sigma_i^2}\right),

where a default of :math:`f = 0.05` (5 % of the host-centric radius)
and :math:`\sigma_{\rm min} = 10\,\text{kpc}` is used.  The combined
density

.. math::

    \rho_{\rm DM,\,total}(r) = \rho_{\rm host\,NFW}(r) + \rho_{\rm sub}(r)

is then passed as the ``total_density`` argument to
:meth:`~model.ClusterModel.from_dens_and_tden`, so the host HSE is
solved self-consistently in the combined potential.

The subhalo catalog only requires the host dark-matter profile (NFW
parameters), not the host gas HSE, so the construction order is natural:

1. Define host NFW.
2. Draw subhalo catalog (masses, positions) from host NFW.
3. Build :func:`~substructure.combined_dm_density` from the catalog.
4. Solve host HSE with :meth:`~model.ClusterModel.from_dens_and_tden`.
5. Build each subhalo :py:class:`~model.ClusterModel`.
6. Find truncation radii with
   :func:`~substructure.find_pressure_truncation_radius`.
7. Assemble :py:class:`~substructure.CompositeCluster`.

.. _composite_evaluator_theory:

The Composite Field Evaluator
''''''''''''''''''''''''''''''

At any 3-D position :math:`{\bf r}`, the composite gas fields are
computed as follows.

**Blending weight.** For subhalo :math:`i` at center :math:`{\bf r}_i`,
with truncation radius :math:`r_{{\rm trunc},i}` and blend width
:math:`b`, the weight is

.. math::

    w_i({\bf r}) =
        \tfrac{1}{2}\!\left[
            1 - \tanh\!\left(
                \frac{|{\bf r} - {\bf r}_i| - r_{{\rm trunc},i}}{b}
            \right)
        \right].

This gives :math:`w_i \to 1` deep inside the subhalo and
:math:`w_i \to 0` far outside, with the transition occurring over a
scale :math:`\sim b` centred on :math:`r_{{\rm trunc},i}`.

**Pressure continuity.** At :math:`r_{{\rm sub}} = r_{{\rm trunc},i}`,
:math:`w_i = \tfrac{1}{2}` and :math:`P_{\rm sub}(r_{{\rm trunc}}) =
P_{\rm host}(|{\bf r}_s|)` by construction, so

.. math::

    P_{\rm blend}(r_{{\rm trunc}}) =
        \tfrac{1}{2} P_{\rm host} + \tfrac{1}{2} P_{\rm sub}(r_{{\rm trunc}})
        = P_{\rm host},

independent of the choice of :math:`b`. Pressure is therefore
continuous at the boundary regardless of blend width.

**Overlap arbitration.** When multiple subhalos have non-negligible
weight at the same point, only the *winning* subhalo contributes.
Two rules are provided:

* ``"deepest_potential"`` — the subhalo whose gravitational potential
  :math:`\Phi_i(|{\bf r} - {\bf r}_i|)` is most negative at the query
  point wins. This correctly handles the case where one subhalo has
  already passed through another.
* ``"nearest_normalized"`` — the subhalo with the smallest
  :math:`|{\bf r} - {\bf r}_i| / r_{{\rm trunc},i}` wins. Cheaper to
  evaluate and usually equivalent to the potential rule when subhalo
  masses are similar.

**Field evaluation.** Starting from host values at every point, the
winning subhalo :math:`i^*` blends its fields in:

.. math::

    f({\bf r}) =
        (1 - w_{i^*})\,f_{\rm host}(|{\bf r}|)
        + w_{i^*}\,f_{\rm sub}(|{\bf r} - {\bf r}_{i^*}|),

where :math:`f` is density, pressure, or bulk velocity.

.. _substructure_examples:

Examples
========

The following examples assume the NFW parameters have already been
chosen (e.g. from :func:`~radial_profiles.nfw_scale_density` and a
target :math:`M_{200}`). All radii are in kpc, masses in
:math:`M_\odot`.

.. _example_combined_dm:

Step 1: Build the Host Model with Combined DM Potential
''''''''''''''''''''''''''''''''''''''''''''''''''''''''

Draw a subhalo catalog first (here a minimal two-subhalo example for
clarity), then construct the combined DM density and solve the host HSE.

.. code-block:: python

    import numpy as np
    from cluster_generator.radial_profiles import (
        beta_model_profile,
        nfw_density_profile,
        nfw_mass_profile,
        nfw_scale_density,
    )
    from cluster_generator.model import ClusterModel
    from cluster_generator.substructure import combined_dm_density

    # Host NFW
    rho_s_host = nfw_scale_density(conc=4.5)   # Msun/kpc^3
    r_s_host   = 400.0                          # kpc
    host_nfw_density = nfw_density_profile(rho_s_host, r_s_host)

    # Subhalo catalog: two subhalos (masses in Msun, positions in kpc)
    sub_masses    = np.array([5e12, 2e12])
    sub_positions = np.array([[800.0, 300.0, 0.0],
                               [-600.0, 0.0, 400.0]])

    # Combined DM density (host + spherically averaged subhalo catalog)
    total_dm = combined_dm_density(host_nfw_density, sub_masses, sub_positions)

    # Host gas profile (beta model)
    rho_gas0 = 2e-2        # Msun/kpc^3, central density
    r_c      = 100.0       # kpc, core radius
    beta     = 2.0/3.0
    host_gas = beta_model_profile(rho_gas0, r_c, beta)

    # Solve HSE — the combined potential is folded in via total_dm
    host = ClusterModel.from_dens_and_tden(
        rmin=0.1, rmax=3000.0,
        density=host_gas,
        total_density=total_dm,
    )

.. _example_subhalo_model:

Step 2: Build Each Subhalo Model
''''''''''''''''''''''''''''''''''

Each subhalo is a self-contained :py:class:`~model.ClusterModel`.
Higher-concentration NFW halos are typical for cluster substructure.

.. code-block:: python

    from cluster_generator.radial_profiles import (
        beta_model_profile,
        nfw_density_profile,
        nfw_scale_density,
    )
    from cluster_generator.model import ClusterModel

    def make_subhalo(M200_sub, c_sub, f_gas=0.03, rmin=0.1, rmax=500.0):
        """Build a subhalo ClusterModel given M200, concentration, gas fraction."""
        rho_s = nfw_scale_density(conc=c_sub)
        # Approximate r_s from M200 and c (simplified; adjust as needed)
        r_200 = (M200_sub / (4 * np.pi / 3 * 200 * 2.775e11)) ** (1.0 / 3.0)
        r_s   = r_200 / c_sub
        sub_total_dm = nfw_density_profile(rho_s, r_s)
        # Gas: a fraction f_gas of the total, in a compact beta model
        rho_gas_sub = beta_model_profile(rho_s * f_gas * 0.1, r_s * 0.3, 2.0 / 3.0)
        return ClusterModel.from_dens_and_tden(
            rmin=rmin, rmax=rmax,
            density=rho_gas_sub,
            total_density=sub_total_dm,
        )

    sub1 = make_subhalo(M200_sub=5e12, c_sub=8.0)
    sub2 = make_subhalo(M200_sub=2e12, c_sub=9.0)

.. _example_truncation:

Step 3: Find Truncation Radii
''''''''''''''''''''''''''''''

.. code-block:: python

    from cluster_generator.substructure import find_pressure_truncation_radius

    r_trunc1 = find_pressure_truncation_radius(sub1, host, sub_positions[0])
    r_trunc2 = find_pressure_truncation_radius(sub2, host, sub_positions[1])

    print(f"Sub1 truncation radius: {r_trunc1:.1f}")
    print(f"Sub2 truncation radius: {r_trunc2:.1f}")

You can also compute the tidal truncation radius for comparison:

.. code-block:: python

    from cluster_generator.substructure import find_tidal_truncation_radius

    r_tidal1 = find_tidal_truncation_radius(sub1, host, sub_positions[0])
    print(f"Sub1 tidal radius: {r_tidal1:.1f}")

.. note::

    The pressure-matched and tidal radii need not agree. The pressure
    match sets the gas boundary; the tidal radius sets the DM boundary.
    For idealized ICM work, the pressure-matched radius is the relevant
    one for gas replacement.

.. _example_composite:

Step 4: Assemble the Composite Evaluator
'''''''''''''''''''''''''''''''''''''''''

.. code-block:: python

    from cluster_generator.substructure import CompositeCluster

    cc = CompositeCluster(
        host=host,
        host_center=[0.0, 0.0, 0.0],       # kpc
        subhalos=[sub1, sub2],
        subhalo_positions=list(sub_positions),
        truncation_radii=[float(r_trunc1.v), float(r_trunc2.v)],
        host_velocity=[0.0, 0.0, 0.0],      # kpc/Myr
        subhalo_velocities=[
            [50.0, -20.0, 0.0],
            [-30.0, 10.0, 5.0],
        ],
        blend_width=50.0,                   # kpc
        overlap_rule="deepest_potential",
    )

.. _example_evaluate:

Step 5: Evaluate Fields
''''''''''''''''''''''''

:meth:`~substructure.CompositeCluster.__call__` accepts scalar or
array-like positions in kpc and returns a dict of
:py:class:`~unyt.unyt_array` objects.

.. code-block:: python

    import numpy as np

    # Single point
    result = cc(800.0, 300.0, 0.0)       # subhalo 1 center
    print(result["density"])
    print(result["pressure"])
    print(result["subhalo_id"])          # 0 → subhalo 1 wins

    # Evaluate on a 1-D slice through the box
    x = np.linspace(-2000.0, 2000.0, 500)
    y = np.zeros(500)
    z = np.zeros(500)
    result_slice = cc(x, y, z)

    density_slice  = result_slice["density"]   # unyt_array, Msun/kpc^3
    pressure_slice = result_slice["pressure"]  # unyt_array, Msun/kpc/Myr^2
    who_wins       = result_slice["subhalo_id"]  # -1=host, 0=sub1, 1=sub2

The velocity field can be used to set bulk velocity offsets on gas
particles or cells:

.. code-block:: python

    velocity_slice = result_slice["velocity"]  # shape (500, 3), kpc/Myr

.. _substructure_api:

API Reference
=============

.. currentmodule:: cluster_generator.substructure

.. autofunction:: find_pressure_truncation_radius

.. autofunction:: find_tidal_truncation_radius

.. autofunction:: combined_dm_density

.. autoclass:: CompositeCluster
   :members: __call__
   :special-members:
