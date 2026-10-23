.. _cluster_models:

Cluster Models
--------------
The :py:class:`model.ClusterModel` class is one of the core structures available in ``cluster_generator``. These objects hold all of
the data related to a single cluster being modeled. The user can use a :py:class:`model.ClusterModel` object to generate particle distributions,
enforce hydrostatic equilibrium, virialize halos, and generate intial conditions. Furthermore, the user can generate
:py:class:`model.ClusterModel` objects using a variety of pre-built protocols using :py:class:`radial_profiles.RadialProfile` objects to provide necessary fields of data.

.. raw:: html

   <hr style="color:black">

.. contents::

.. raw:: html

   <hr style="height:10px;background-color:black">

.. _math_overview_models:

Cluster Models: Mathematical Overview
=====================================

The Intracluster Medium (ICM)
''''''''''''''''''''''''''''''

Assuming the intracluster medium of galaxy clusters can be modeled as an
ideal fluid, the momentum density :math:`\rho{\bf v}` of the
gas obeys the Euler momentum equation (here written in conservative form
and ignoring magnetic fields, viscosity, etc.):

.. math::

    \frac{\partial({\rho_g{\bf v}})}{\partial{t}} + \nabla \cdot (\rho_g{\bf v}{\bf v})
    = -\nabla{P} + \rho_g{\bf g}

where :math:`\rho_g` is the gas density, :math:`{\bf v}` is the gas velocity,
:math:`P` is the gas pressure, and :math:`{\bf g}` is the gravitational
acceleration. The assumption of hydrostatic equilibrium implies that
:math:`{\bf v} = 0` and that all time derivatives are zero, giving:

.. math::

    \nabla{P} = \rho_g{\bf g}

In the context of galaxy clusters, we may typically assume an ideal gas behavior for the ICM, and therefore,

.. math::

    \Gamma = -\frac{\nabla P}{\rho_g} = \frac{-k_b T}{m_p \eta} \left[\frac{d\ln(\rho_g)}{dr} + \frac{d\ln(T)}{dr} \right].

As such, we recognize that from :math:`\rho_g` and :math:`T`, we can generally obtain the acceleration and therefore the dynamical mass
of the system. Because the dynamical mass need not be equivalent to the gasseous mass, it follows that finding the dynamical mass will also
yield the halo mass of the system.

It is worth noting throughout the remainder of this document that for :math:`r\to \infty`, if :math:`\rho \sim r^\alpha` and :math:`T\sim r^\beta`,
then

.. math::

    \Gamma \sim r^{\beta-1}.

This point becomes important as we begin discussing the interaction between the gas dynamics above and the potential of the system.

Gravity
'''''''

In the preceding section, we described the basic mathematics of hydrostatic equilibrium; however, to find the dynamical mass and other related quantities,
an expression for :math:`\Gamma` must be obtained independently. In each gravitational theory implemented in CG, the dynamical field may be determined in terms of
:math:`\Gamma` and can then be used to construct the correct mass profiles. In the window below, we've included 3 archetypal examples:

.. tab-set::

    .. tab-item:: Newtonian

        In Newtonian gravity, the dynamical mass equation derived from the Poisson equation is

        .. math::

            M_{\mathrm{dyn}}(<r) = \frac{r^2}{G}\Gamma = \frac{-r^2 k_b T}{G m_p \eta} \left[\frac{d\ln(\rho_g)}{dr} + \frac{d\ln(T)}{dr} \right].

    .. tab-item:: AQUAL

        In the AQUAL flavor of MOND, the dynamical mass is different from the Newtonian case by a factor of :math:`\mu(a)`.

        .. math::

            M_{\mathrm{dyn}} = \frac{r^2}{G}\mu\left(\frac{| \Gamma |}{a_0}\right)\Gamma.

    .. tab-item:: QUMOND

        In the QUMOND flavor of MOND, the equations cannot be solved independently and therefore numerical methods are required to solve the
        implicit equation

        .. math::

            \nu\left(\frac{|\nabla \Psi |}{a_0}\right)\nabla \Psi = \Gamma,\;\;\nabla \Psi = \frac{GM_{\mathrm{dyn}}}{r^2}

Each of these theories has a different asymptotic behavior in terms of the temperature. Therefore, the required temperature behavior at large
radii is constrained by the following table. This should be considered when choosing a model.

+--------------+-----------------------------------------------------------------------------------+----------------------------+
| Gravity Type | :math:`\Gamma` and :math:`\Phi` relationship                                      | Mass Stability Condition   |
+==============+===================================================================================+============================+
| Newtonian    | :math:`\Gamma = \nabla \Phi = GM_{\mathrm{dyn}}(<r)/r^2`                          | :math:`T\sim r^{-1}`       |
|              |                                                                                   |                            |
+--------------+-----------------------------------------------------------------------------------+----------------------------+
| AQUAL        | :math:`\mu\left(\frac{|\Gamma|}{a_0}\right)\Gamma = GM_{\mathrm{dyn}}(<r)/r^2`    | :math:`T\sim r^{0}`        |
|              |                                                                                   |                            |
+--------------+-----------------------------------------------------------------------------------+----------------------------+
| QUMOND       | :math:`\Gamma = \nu\left(\frac{|\nabla \Psi|}{a_0}\right) \nabla \Psi` where      | :math:`T\sim r^{0}`        |
|              | :math:`\nabla \Psi = GM_{\mathrm{dyn}}(<r)/r^2`                                   |                            |
+--------------+-----------------------------------------------------------------------------------+----------------------------+

.. raw:: html

   <hr style="height:10px;background-color:black">

Generating a :py:class:`model.ClusterModel` Using Radial Profiles
=================================================================

Based on the mathematics above, there are a variety of ways to produce :py:class:`model.ClusterModel` objects. Most of the common approaches
that see use in practice are built into the ``cluster_generator`` package; however, the :py:meth:`~model.ClusterModel.from_arrays` class method can
be used to generate a :py:class:`model.ClusterModel` object manually. The available generation approaches are listed as follows:

+---------------------------------+--------------------------------------------------------------------------+------------------------------------------------------------------+
| Method                          |                                 Function                                 | Description                                                      |
+=================================+==========================================================================+==================================================================+
| From :math:`\rho_g`             | :py:meth:`~model.ClusterModel.from_dens_and_tden`                        | Generates the galaxy cluster from the gas and dynamical density  |
| and :math:`\rho_{\mathrm{dyn}}` |                                                                          | profiles. Computes temperature / grav. field.                    |
+---------------------------------+--------------------------------------------------------------------------+------------------------------------------------------------------+
| From :math:`\rho_g`             | :py:meth:`~model.ClusterModel.from_dens_and_temp`                        | Generates the galaxy cluster from the gas density and temperature|
| and :math:`T_g`                 |                                                                          | profiles. Computes total mass, dm, stellar etc.                  |
+---------------------------------+--------------------------------------------------------------------------+------------------------------------------------------------------+
| From :math:`\rho_g`             |  :py:meth:`~model.ClusterModel.from_dens_and_entr`                       | Generates the galaxy cluster from the gas density and entropy    |
| and :math:`\rho_{\mathrm{dyn}}` |                                                                          | profiles. Computes total mass, dm, stellar etc.                  |
+---------------------------------+--------------------------------------------------------------------------+------------------------------------------------------------------+

``ClusterModel`` from Gas Density and Gas Temperature Profiles
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
The user may generate a :py:class:`model.ClusterModel` from the :py:meth:`model.ClusterModel.from_dens_and_temp` method, which requires :py:class:`radial_profiles.RadialProfile` objects
for the temperature and the gas density. Using the condition of hydrostatic equilibrium, the potential, pressure, dynamical mass, and other
necessary fields are automatically computed based on the chosen gravity theory.

The pressure :math:`P(r)` is provided via the ideal gas law:

.. math::

    P(r) = \frac{\rho_g(r) k_b T(r)}{m_p \eta},

where :math:`\eta` is the mean-molecular mass (generally 0.6 for galaxy clusters).

Once the pressure is determined, Euler's Equations can be used for an incompressible fluid, yielding

.. math::

    \Gamma = \frac{-\nabla P(r)}{\rho_g} = \nabla \Phi

Once the potential has been determined, the corresponding gravity theory is applied to determine the dynamical mass of the system and
by extension determine the necessary halo component.

.. math::

    M_{dm} = M_{\mathrm{dyn}} - M_{\mathrm{bary}}.

+-----------------------+-----------------------------------+
|Provided               | Computed                          |
+=======================+===================================+
| :math:`T_g`           | :math:`\rho_{dm}, M_{dm}, M_{dyn}`|
| :math:`\rho_{g}`      | :math:`S, \Phi, \nabla \Phi, P`   |
| :math:`\rho_{\star}`  |                                   |
+-----------------------+-----------------------------------+



``ClusterModel`` from Gas Density and Total Density Profiles
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
In the case where the dynamical density is already known, the halo mass function is trivially obtained from

.. math::

    \rho_{\mathrm{halo}} = \rho_{\mathrm{dyn}} - \rho_{\mathrm{bary}},

where the baryonic component may contain a stellar distribution if the user supplies one. The difficult in this case
is to efficiently determine the temperature profile which yields HSE. Recalling that

.. math::

    \Gamma = -\frac{\nabla P}{\rho_g} = \frac{-k_b T}{m_p \eta} \left[\frac{d\ln(\rho_g)}{dr} + \frac{d\ln(T)}{dr} \right].

The differential equation may be inverted, yielding

.. math::

    T(r) = \frac{-m_p \eta}{k_b \rho_g} \int_{r_0}^r \rho_g(r') \nabla \Phi(r') dr' + \frac{\rho_g(r_0)}{\rho_g(r)}T(r_0).

The most efficient approach here is to take :math:`r_0 = \infty`, in which case,

.. math::

    T(r) = \frac{m_p \eta}{k_b \rho_g} \int_{r}^\infty \rho_g(r') \nabla \Phi(r') dr'.

From this, the temperature profile is obtained.

``ClusterModel`` from Gas Density and Gas Entropy Profiles
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Users may also generate a :py:class:`model.ClusterModel` object from an entropy profile instead of the corresponding temperature profile.
There is a 1-to-1 correspondence between entropy and temperature given that the gas density is fixed, therefore, the procedure
follows exactly from the previous section; however, the entropy is converted first to a temperature profile using the widely accepted formula

.. math::

    S(r) = k_b T_g(r)n_e(r)^{-2/3}.


``ClusterModel`` Without Gas
++++++++++++++++++++++++++++++++++++++++++
Not all of the :py:class:`model.ClusterModel` objects are generated with a gas density profile. If one wants to produce a gas-less model,
the :py:meth:`model.ClusterModel.no_gas` method is available. Users must provide a dynamical density profile, and are optionally allowed
to provide a stellar component. See the API reference for syntactical information.

Checking the Hydrostatic Equilibrium
====================================
In the vast majority of cases, the aim of the :py:class:`model.ClusterModel` object is to produce a model which is in hydrostatic equilibrium. To check how successful the algorithm was in
generating a hydrostatic galaxy cluster, one can use the :py:meth:`model.ClusterModel.check_hse` method, which will provide the maximal relative deviation of the system from hydrostatic equilibrium.

This is done by recalling that the hydrostatic equilibrium condition requires that

.. math::

    -\frac{\nabla P}{\rho_g} = \nabla \Phi.

This, we define the hydrostatic variable :math:`\xi` such that

.. math::

    \xi = \frac{\nabla \Phi \rho_g + \nabla \Phi}{\nabla \Phi \rho_g}.

clearly, :math:`\xi \approx 0` indicates a successfully equilibrated cluster.

Setting a Magnetic Field Strength Profile
=========================================
Magnetic fields play a major role in the dynamics of the ionized plasma modeled in hydro simulations of galaxy clusters. In ``cluster_generator``, there
are currently two approaches for generating a magnetic field for a galaxy cluster.

Setting a Magnetic Field from :math:`\beta`
+++++++++++++++++++++++++++++++++++++++++++
Users can specify the magnetic field profile using the :py:meth:`~model.ClusterModel.set_magnetic_field_from_beta`, which allows the user
to specify the magnetic pressure factor :math:`\beta`, which is defined such that

.. math::

    \beta = \frac{p_{\mathrm{thermal}}}{p_{\mathrm{magnetic}}}.

Setting a Magnetic Field from plasma density
++++++++++++++++++++++++++++++++++++++++++++
Another common approach for initializing magnetic fields is to let the magnetic field be proportional to a power of the gas density. Most commonly, this value is
:math:`\eta = \frac{2}{3}`; however, the user may specify whichever exponent they choose. To initialize a cluster with a magnetic field using this approach, use the :py:meth:`~model.ClusterModel.set_magnetic_field_from_density`.

Adding Other Fields
===================
Users may add additional fields to the :py:class`~model.ClusterModel` instance using the :py:meth:`~model.ClusterModel.set_field` method, which takes a
``name`` and ``value`` and loads the corresponding field into the model.

.. attention::

    User's should be aware that ``value``'s being attributed as fields must be ``unyt_array``'s, not generic ``list`` types or ``np.ndarray``. Furthermore,
    the length of the value being specified must match ``ClusterModel.num_elements``.

Reading and Writing :py:class:`model.ClusterModel` Objects to and from Disk
===========================================================================
.. admonition:: User Advice

    Generating :py:class:`model.ClusterModel` instances can be **slow**. As many cases as possible, it is worthwhile to
    write models to disk in one of the formats discussed below so that they can be read more easily.

Many different protocols for writing and reading :py:class:`model.ClusterModel` instances to / from disk have been implemented. The following table summarizes the
options available; however, we advice consulting the API reference for each of the methods to get more information about specifics regarding each of the available formats.

+---------------+--------------------------------------------------------+-----------------------------------------------------+-----------------------------------+
| Format        | Write Method                                           | Read Method                                         | Notes                             |
+===============+========================================================+=====================================================+===================================+
| ASCII         | :py:meth:`~model.ClusterModel.write_model_to_ascii`    | **Not Yet Implemented**                             | None                              |
+---------------+--------------------------------------------------------+-----------------------------------------------------+-----------------------------------+
| HDF5          | :py:meth:`~model.ClusterModel.write_model_to_h5`       | :py:meth:`~model.ClusterModel.from_h5_file`         | None                              |
+---------------+--------------------------------------------------------+-----------------------------------------------------+-----------------------------------+
| BINARY        | :py:meth:`~model.ClusterModel.write_model_to_binary`   | **Not Yet Implemented**                             | None                              |
+---------------+--------------------------------------------------------+-----------------------------------------------------+-----------------------------------+

Non-Physical Profiles
=====================

Sometimes, for a variety of reasons, generated :py:class:`model.ClusterModel` instances might have profiles which are non-physical. Sometimes, this is caused by a poor choice
of radial profile for the given gravity theory, or is caused because some other physical constraint is not respected. CG has a model (:py:mod:`correction`) entirely focused on
fixing these problems. For a comprehensive description of these issues and the algorithms used to solve them, see :ref:`correction` .
