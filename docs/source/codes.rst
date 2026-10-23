.. _codes:

Setting Up Idealized Cluster Problems in Various Codes
------------------------------------------------------

``FLASH``
=========

The ``FLASH`` simulation setup ``GalaxyClusterMerger`` can be used with the
inputs to ``cluster_generator`` to perform a single-cluster simulation or a
binary merger. ``GalaxyClusterMerger`` has two modes in which it can be run in:
``RigidGravity``, which uses rigid potentials to represent the dark matter and
stars, or ``SelfGravity``, which represents the dark matter and stars as
particles. ``GalaxyClusterMerger`` also supports the inclusion of a tangled
magnetic field in the simulations.

The ``GalaxyClusterMerger`` setup is currently available from John ZuHone by
request for users who have a licensed copy of the ``FLASH`` code.

``GAMER``
=========

The ``GAMER`` simulation setup ``ClusterMerger`` can be used with the inputs to
``cluster_generator`` to perform a single-cluster simulation, binary merger, or
a triple merger. The required, optional, and recommended ``Makefile`` options
for the ``ClusterMerger`` setup are as follows:

* ``SIMU_OPTION += -DMODEL=HYDRO`` (required)
* ``SIMU_OPTION += -DGRAVITY`` (required)
* ``SIMU_OPTION += -DPARTICLE`` (required)
* ``SIMU_OPTION += -DNCOMP_PASSIVE_USER=n`` (required, n = ``Merger_Coll_NumHalos+Merger_Coll_UseMetals``, see below )
* ``SIMU_OPTION += -DMHD`` (optional, for simulations with magnetic fields)
* ``SIMU_OPTION += -DEOS=EOS_GAMMA`` (required)
* ``SIMU_OPTION += -DPAR_NATT_USER=2`` (required, supports the type and halo attributes)
* ``SIMU_OPTION += -DGPU`` (recommended)
* ``SIMU_OPTION += -DLOAD_BALANCE=HILBERT`` (recommended)
* ``SIMU_OPTION += -DOPENMP`` (recommended)
* ``SIMU_OPTION += -DSUPPORT_HDF5`` (required)

Arepo
=====

``cluster_generator'' can be used to create HDF5-based initial conditions for
one, two, or three clusters for an idealized merger simulation for the Arepo code.
The way to do this is to create a Gadget-like HDF5 initial conditions file:



After generating the initial conditions file, the ICs will need to be processed
through two steps before the simulation proper can be run. The first step is to
add a background grid of cells to the particle distribution. This is done with
the ``ADD_BACKGROUND_GRID`` Config option in Arepo. The Arepo configuration
options to be enabled in ``Config.sh`` for this step include:



GIZMO
=====

Coming soon!

Enzo
====

Coming soon!

Ramses
======

Coming soon!

AthenaK
=======

Coming soon!
