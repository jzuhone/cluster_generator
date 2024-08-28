.. _particles:

Particles
---------

In practice, (magneto)hydrodynamics codes break down into two generic categories: particle based and grid based. Particle
based codes include smooth-particle hydrodynamics codes likes GADGET as well as more modern moving-mesh codes like AREPO. Grid codes
are generally AMR codes like RAMSES.

Even in codes that utilize an AMR structure, it is typical for gravitationally interacting particles (DM, black holes, stars, etc.) to
be treated as individual particles as well, subject to a Poisson solving scheme like the Barnes-Hut Algorithm. As such, there are a variety of
contexts in which ``cluster_generator`` must convert it's semi-analytic radial profiles into realized particles which can then be fed to
the simulation code of choice. This is the purpose of the :py:mod:`particles` module.


Theoretical Overview
====================

Generically, different types of particles may need different sets of parameters; however, all particles have a mass, a position, and a
velocity which must be constructed in such a way as to be analogous to the radial profile from which they originated. For fluid-type particles,
a minimum of the density and internal energy must be specified as well.

.. note::

    Why do we specify a density of gas particles as well as a mass? This is really just a shortcut; as will be described below, we
    position particles in such a way as to make the total mass and local density accurate on scales containing at least a couple of particles.
    Specifying a density just saves us the time of having to interpolate a density once the particles have already been allocated.

Particle Mass
'''''''''''''

Particles of a given type are constrained to all share a particle-mass. This means that regions of higher density are represented by
a greater abundance of particles, not by particles of a greater (per-particle) mass. Thus, if :math:`N` particles are generated to represent
a total mass of :math:`M`, then the per-particle mass is simply :math:`M/N`.

Particle Position
'''''''''''''''''

Because :py:class:`model.ClusterModel` are spherically symmetric, the particle placement is done in two stages: first, the radial
position is determined and then the angular position is determined.

The radial position is determined via `inverse transform sampling <https://en.wikipedia.org/wiki/Inverse_transform_sampling>`_ of the total
mass profile for the particular component of the system being represented by the particles. Once radial positions have been assigned, the angular position
of each particle is assigned randomly on the sphere.

Particle Velocities
'''''''''''''''''''

The mass density :math:`\rho({\bf r})` of such a system can be derived by
integrating the phase-space distribution function :math:`f({\bf r}, {\bf v})`
over velocity space:

.. math::

    \rho({\bf r}) = \int{f({\bf r}, {\bf v})d^3{\bf v}}

where :math:`{\bf r}` and :math:`{\bf v}` are the position and velocity
vectors. Assuming spherical symmetry and isotropy, all quantities are simply
functions of the scalars :math:`r` and :math:`v`, and
:math:`d^3{\bf v} = 4\pi{v^2}dv`:

.. math::

    \rho(r) = 4\pi\int{f(r, v)v^2dv}

Assuming zero net angular momentum for the cluster, there is a unique
distribution function :math:`f(E)` which corresponds to the density
:math:`\rho(r)`. Since the total energy of a particle is
:math:`E = v^2/2 + \Phi` (where :math:`\Phi(r)` is the gravitational
potential) and further defining :math:`\Psi = -\Phi` and
:math:`{\cal E} = -E = \Psi - \frac{1}{2}v^2`, we find:

.. math::

    \rho(r) = 4\pi\int_0^{\Psi}f({\cal E})\sqrt{2(\Psi-{\cal E})}d{\cal E}

After differentiating this equation once with respect to :math:`\Psi` and
inverting the resulting Abel integrel equation, we finally have:

.. math::

    f({\cal E}) = \frac{1}{\sqrt{8}\pi^2}\left[\int^{\cal E}_0{d^2\rho \over d\Psi^2}{d\Psi
    \over \sqrt{{\cal E} - \Psi}} + \frac{1}{\sqrt{{\cal E}}}\left({d\rho \over d\Psi}\right)_{\Psi=0} \right]

which given a density-potential pair for an equilibrium halo, can be used to
determine particle speeds. For our cluster models, this equation must (in
general) be solved numerically, even if the underlying dark matter, stellar,
and gas densities can be expressed analytically.

To generate the particle speeds, the distribution function :math:`f({\cal E})`
is used with uniform random numbers :math:`u \in [0, 1]` via an
acceptance-rejection method. The particle velocity components are isotropically
distributed in the tangential directions :math:`\theta` and :math:`\phi`.

Checking the Virial Equilibrium
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It is probably a good idea to check that the resulting distribution functions
for the dark matter and/or stars are consistent with the input mass density
profiles. The :py:meth:`~cluster_model.ClusterModel.check_dm_virial`
or :py:meth:`~cluster_model.ClusterModel.check_star_virial`
methods can be used to perform a quick check on the accuracy of the virial
equilibrium model for each of these types. These methods return two NumPy
arrays, the first being the density profile computed from integrating the
distribution function, and the second being the relative difference between
the input density profile and the one calculated using this method.

.. code-block:: python

    import matplotlib.pyplot as plt
    rho, diff = p.check_dm_virial()
    # Plot this up
    fig, ax = plt.subplots(figsize=(10,10))
    ax.loglog(vir["radius"], vir["dark_matter_density"], 'x',
              label="Input mass density", markersize=10)
    ax.loglog(vir["radius"], rho, label="Derived mass density", lw=3)
    ax.legend()
    ax.set_xlabel("r (kpc)")
    ax.set_ylabel("$\mathrm{\\rho\ (M_\odot\ kpc^{-3})}$")

.. image:: _images/check_density.png

One can see that the derived density diverges from the input density at large
radii, due to difficulties with numerically integrating to infinite radius. So long
as the maximum radius of the profile is very large, this should not matter very
much.

Generating Particles from ``ClusterModel`` Objects
==================================================

Once a :py:class:`~model.ClusterModel` object is created,
it can be used to generate particle positions and velocities, of gas, dark matter,
and/or star types. For each particle species, there is a corresponding method (i.e. :py:meth:`~model.ClusterModel.generate_gas_particles`) to
generate the particular type of particle.

The ``ClusterParticles`` Class
==============================

The :py:class:`~particles.ClusterParticles` class is a
container for particle properties. It is the format that is returned
from the various ``generate_*_particles`` methods described above. This
class can be used to perform further operations on particles or write
them to disk.

``ClusterParticles`` Operations
'''''''''''''''''''''''''''''''''
Several kinds of operations can be performed on
:py:class:`~particles.ClusterParticles` objects.

Adding ``ClusterParticles`` Objects
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:py:class:`~particles.ClusterParticles` objects can be added
together. In this case, we add particles of different types so that they
are combined into a single object:

.. code-block:: python

    all_particles = gas_particles+dm_particles+star_particles

If you have multiple :py:class:`~particles.ClusterParticles`
objects with the same particle types, the particle field arrays will simply
be concatenated together:

.. code-block:: python

    gas_parts = gas_parts1+gas_parts2

Dropping Particle Types
^^^^^^^^^^^^^^^^^^^^^^^

To drop all fields of a specific particle type from a
:py:class:`~particles.ClusterParticles` instance, use the
:py:meth:`~particles.ClusterParticles.drop_ptypes` method:

.. code-block:: python

    # Drop gas particles
    parts.drop_ptypes("gas")

    # Drop DM and star particles
    parts.drop_ptypes(["dm", "star"])

Add Position and Velocity Offsets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, a :py:class:`~particles.ClusterParticles` object is
centered at (0, 0, 0) kpc and has a bulk velocity of (0, 0, 0) kpc/Myr.
To translate the particle positions of a
:py:class:`~particles.ClusterParticles` instance to a new center,
or to boost the particle velocities to a new frame, or both, we can use the
:py:meth:`~particles.ClusterParticles.add_offsets` method:

.. code-block:: python

    # shift the particle positions by this amount in each direction
    r_ctr = [1000.0, -1000.0, 10.0] # kpc
    # shift the particle velocities by this amount in each direction
    v_ctr = [-500.0, 200.0, 0.0] # kpc/Myr
    parts.add_offsets(r_ctr, v_ctr)

.. note::

    The :py:meth:`~particles.ClusterParticles.add_offsets` does
    exactly as it is named, it adds *offsets* to the positions and velocities,
    so these are relative translations by the given amounts and not movements
    to the values of the ``r_ctr`` and ``v_ctr`` parameters.

Make a Cut on Radius
^^^^^^^^^^^^^^^^^^^^

To cut out particles beyond a certain radius, use the
:py:meth:`~particles.ClusterParticles.make_radial_cut` method:

.. code-block:: python

    # make a radial cut at r_max, assuming the center is [0, 0, 0] kpc
    r_max = 5000.0 # in kpc
    parts.make_radial_cut(r_max)

    # make a radial cut at r_max, assuming the center is
    # [500, 500, 500] kpc
    r_max = 5000.0 # in kpc
    center = [500, 500, 500] # in kpc
    parts.make_radial_cut(r_max, center=center)

You can also cut out only certain particle types:

.. code-block:: python

    # make radial cut on stars only
    r_max = 5000.0 # in kpc
    parts.make_radial_cut(r_max, ptypes="star")

    # make radial cut on stars and dm
    r_max = 5000.0 # in kpc
    parts.make_radial_cut(r_max, ptypes=["star","dm"])

Add Black Hole Particles
^^^^^^^^^^^^^^^^^^^^^^^^

To add a single black hole particle, use the
:py:meth:`~particles.ClusterParticles.add_black_hole`
method. The simplest way to do this is to simply provide it with a
mass, which will place a black hole particle at [0.0, 0.0, 0.0] kpc
with zero velocity:

.. code-block:: python

    Mbh = 3.0e9 # assumed units of Msun
    parts.add_black_hole(Mbh)

to supply an alternate position and velocity, use ``pos`` and ``vel``:

.. code-block:: python

    Mbh = 3.0e9 # assumed units of Msun
    pos = [300.0, 100.0, -100.0] # assumed units of kpc
    vel = [-200.0, -100.0, 50.0] # assumed units of kpc/Myr
    parts.add_black_hole(Mbh, pos=pos, vel=vel)

to choose the position and velocity of the DM particle with the minimum
gravitational potential, set ``use_pot_min=True``:

.. code-block:: python

    Mbh = 3.0e9 # assumed units of Msun
    parts.add_black_hole(Mbh, use_pot_min=True)

Add a New Field or Change a Field
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A new field can be added to the particles rather easily. For example, if
you wanted to add a field which added a tag to the DM particles to keep
track of which halo they originated from:

.. code-block:: python

    num_particles1 = 1_000_000
    num_particles2 = 200_000
    halo1 = np.ones(num_particles1)
    halo2 = 2.0*np.ones(num_particles2)
    cluster1.set_field("dm", "tag", halo1)
    cluster2.set_field("dm", "tag", halo2)

If you pick a field that already exists, it will be overwritten by default
with the new values, but you will get a warning. If you want to add the numerical
values to those of the existing field, set ``add=True``:

.. code-block:: python

    import unyt as u

    B0 = 1.5*np.ones(num_particles)*u.uG # A constant field of 1.5 microgauss
    cluster1.set_field("gas", "magnetic_field_x", B0, add=True)

If the field you are adding is a passive scalar, set ``passive_scalar=True``:

.. code-block:: python

    import unyt as u

    metals = 0.3*np.ones(num_particles)*u.Zsun
    cluster1.set_field("gas", "metals", metals, passive_scalar=True)

.. warning::

    It is obviously not recommended to alter a particle field created
    from a radial profile or an equilibrium condition!

``ClusterParticles`` I/O
++++++++++++++++++++++++

:py:class:`~particles.ClusterParticles` objects can be written
to disk or read from a file on disk. The normal way of writing the particles
to disk is to use the
:py:meth:`~particles.ClusterParticles.write_particles` method:

.. code-block:: python

    # overwrite is a boolean which allows you to overwrite an existing file
    parts.write_particles("my_particles.h5", overwrite=True)

A :py:class:`~particles.ClusterParticles` object can be read
in from disk using the
:py:meth:`~particles.ClusterParticles.from_file` method:

.. code-block:: python

    import cluster_generator as cg
    new_parts = cg.ClusterParticles.from_file("my_particles.h5")

To only read in certain particle types from the file, specify them in
``ptypes``:

.. code-block:: python

    import cluster_generator as cg

    # only gas particles
    gas_only = cg.ClusterParticles.from_file("my_particles.h5", ptypes="gas")

    # only dm, star particles
    dm_star = cg.ClusterParticles.from_file("my_particles.h5",
                                            ptypes=["dm", "star"])

Gadget-Like I/O
^^^^^^^^^^^^^^^

``cluster_generator`` also provides for the creation of Gadget-like snapshot/IC
files for use with codes such as Gadget, Arepo, GIZMO, etc. The
:py:meth:`~particles.ClusterParticles.write_to_gadget_file`
writes an HDF5 file with the different particle types in the
:py:class:`~particles.ClusterParticles` object in a format
that can be used as initial conditions for these codes. It requires a
``box_size`` parameter, which determines the width of the cubical box that
the initial conditions will be set within.

.. code-block:: python

    box_size = 20000.0 # in kpc
    parts.write_to_gadget_file("cluster_ics.hdf5", box_size, overwrite=True)

To create a new :py:class:`~particles.ClusterParticles` object
from one of these files, use the
:py:meth:`~particles.ClusterParticles.from_gadget_file` method:

.. code-block:: python

    import cluster_generator as cg

    # all particle types
    parts = cg.ClusterParticles.from_gadget_file("cluster_ics.hdf5")

    # only gas particles
    gas_only = cg.ClusterParticles.from_gadget_file("cluster_ics.hdf5",
                                                    ptypes="gas")

    # only dm, star particles
    dm_star = cg.ClusterParticles.from_gadget_file("cluster_ics.hdf5",
                                                   ptypes=["dm", "star"])

For more information on how these files are used in Gadget-like codes, see
:ref:`codes`.
