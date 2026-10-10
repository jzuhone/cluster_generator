cluster_generator
=================

|yt-project| |docs| |testing| |Github Page| |Pylint| |coverage| |ncodes|

.. raw:: html

   <hr style="height:10px;background-color:black">

`Cluster Generator <https://jzuhone.github.io/cluster_generator>`_ (CG) is a cross-platform Python library for generating initial conditions of galaxy clusters for N-body / hydrodynamics codes.
CG provides a variety of construction approaches, different physical assumption, profiles, and gravitational theories. Furthermore, CG is intended to interface with
a number of N-body / hydrodynamics codes used in studies of galaxy clusters, reducing the headache of converting initial conditions between formats for different simulation softwares. GCP's goal is to provide
comprehensive tools for modeling and implementation of galaxy clusters in astrophysical simulations to promote the study of galaxy cluster dynamics.

This repository contains the core package, which is constructed modularly to facilitate easy development by users to meet particular scientific use cases. All of the
necessary tools to get started building initial conditions are provided.

You can access the documentation `here <http://jzuhone.github.io/cluster_generator>`_, or build it from scratch using the ``./docs`` directory in this source distribution.

Development occurs here on Github, if you encounter any bugs, issues, documentation failures, or want to suggest features, we recommend that you submit an issue on
the issues page of the repository.

For installation directions, visit the `getting started page <https://jzuhone.github.io/cluster_generator/build/html/Getting_Started.html>`_.


.. raw:: html

   <hr style="color:black">
Features
========

.. grid:: 2

    .. grid-item::

        .. dropdown:: Radial Profiles

            Each of the following radial profiles is included for use in cluster construction:

            .. dropdown:: NFW Profiles

              - :py:func:`~radial_profiles.nfw_density_profile`
              - :py:func:`~radial_profiles.nfw_mass_profile`
              - :py:func:`~radial_profiles.snfw_density_profile`
              - :py:func:`~radial_profiles.snfw_mass_profile`
              - :py:func:`~radial_profiles.cored_snfw_density_profile`
              - :py:func:`~radial_profiles.cored_snfw_mass_profile`
              - :py:func:`~radial_profiles.tnfw_density_profile`
              - :py:func:`~radial_profiles.tnfw_mass_profile`

            .. dropdown:: Hernquist Profiles

              - :py:func:`~radial_profiles.hernquist_density_profile`
              - :py:func:`~radial_profiles.hernquist_mass_profile`
              - :py:func:`~radial_profiles.cored_hernquist_density_profile`
              - :py:func:`~radial_profiles.cored_hernquist_mass_profile`

            .. dropdown:: Einasto Profiles

              - :py:func:`~radial_profiles.einasto_density_profile`
              - :py:func:`~radial_profiles.einasto_mass_profile`

            .. dropdown:: General Profiles

              - :py:func:`~radial_profiles.power_law_profile`
              - :py:func:`~radial_profiles.constant_profile`
              - :py:func:`~radial_profiles.beta_model_profile`

            .. dropdown:: Paper Specific Profiles

              - **Vikhlinin et. al. 2006**
                - :py:func:`~radial_profiles.vikhlinin_density_profile`
                - :py:func:`~radial_profiles.vikhlinin_temperature_profile`
              - **Ascasibar & Markevitch 2006**
                - :py:func:`~radial_profiles.am06_density_profile`
                - :py:func:`~radial_profiles.am06_temperature_profile`

            .. dropdown:: Entropy Profiles

              - :py:func:`~radial_profiles.walker_entropy_profile`
              - :py:func:`~radial_profiles.baseline_entropy_profile`
              - :py:func:`~radial_profiles.broken_entropy_profile`

    .. grid-item::

        .. dropdown:: Gravitational Theories

            ``cluster_generator`` not only provides initial condition generation capacity, but also provides a
            comprehensive catalog of alternative gravity theories to explore. The following are built-in, but adding more
            is a relatively simple task:

            - :ref:`Newtonian Gravity <gravity>`

            .. dropdown:: MONDian Gravities

                - :ref:`AQUAL <aqual>`
                - :ref:`QUMOND <qumond>`


    .. grid-item::

        .. dropdown:: Implemented Codes

            ``cluster_generator`` provides end-to-end initial condition generation tools for **all** of the following
            codes:

            - :ref:`RAMSES <ramses>`
            - :ref:`ATHENA++ <athena>`
            - :ref:`AREPO <arepo>`
            - :ref:`GAMER <gamer>`
            - :ref:`FLASH <flash>`
            - :ref:`GIZMO <gizmo>`
            - :ref:`ENZO <enzo>`

    .. grid-item::

        .. dropdown:: Available Datasets

            The :ref:`Collections <collections>` system provides users access to pre-built galaxy clusters from the available literature. Cluster fits
            are available for all of the following papers:

            - `Vikhlinin et. al. 2006 <https://ui.adsabs.harvard.edu/abs/2006ApJ...640..691V/abstract>`_


Resources
=========

.. grid:: 2
    :padding: 3
    :gutter: 5

    .. grid-item-card::
        :img-top: _images/index/stopwatch_icon.png

        Quickstart Guide
        ^^^^^^^^^^^^^^^^
        New to ``cluster_generator``? The quickstart guide is the best place to start learning to use all of the
        tools that we have to offer!

        +++

        .. button-ref:: getting_started
            :expand:
            :color: secondary
            :click-parent:

            To The Quickstart Page

    .. grid-item-card::
        :img-top: _images/index/lightbulb.png

        Examples
        ^^^^^^^^
        Have some basic experience with ``cluster_generator``, but want to see a guide on how to execute a particular task? Need
        to find some code to copy and paste? The examples page contains a wide variety of use case examples and explainations
        for all of the various parts of the ``cluster_generator`` library.

        +++

        .. button-ref:: examples
            :expand:
            :color: secondary
            :click-parent:

            To the Examples Page

    .. grid-item-card::
        :img-top: _images/index/book.svg

        User References
        ^^^^^^^^^^^^^^^^
        The user guide contains comprehensive, text based explainations of the backbone components of the ``cluster_generator`` library.
        If you're looking for information on the underlying code or for more details on particular aspects of the API, this is your best resource.

        +++

        .. button-ref:: codes
            :expand:
            :color: secondary
            :click-parent:

            To the User Guide

    .. grid-item-card::
        :img-top: _images/index/api_icon.png

        API Reference
        ^^^^^^^^^^^^^

        Doing a deep dive into our code? Looking to contribute to development? The API reference is a comprehensive resource
        complete with source code and type hinting so that you can find every detail you might need.

        +++

        .. button-ref:: api
            :expand:
            :color: secondary
            :click-parent:

            API Reference


Contents
========
.. raw:: html

   <hr style="height:10px;background-color:black">

.. toctree::
   :maxdepth: 1

   Getting_Started
   gravity
   models
   collections
   substructure
   examples
   api

Related Projects
================

.. grid:: 2
    :padding: 3
    :gutter: 5

    .. grid-item-card::
        :img-top: _images/index/PyXSIM.png

        PyXSIM
        ^^^^^^

        Convert your ``cluster_generator`` systems into synthetic photon event lists for use in simulating observations using X-ray observatories using PyXSIM.
        ``cluster_generator`` is designed to easily interface with this library to provide as much ease as possible when building simulated observations of clusters.

        +++

        .. button-link:: http://hea-www.cfa.harvard.edu/~jzuhone/pyxsim/
            :expand:
            :color: secondary
            :click-parent:

            Documentation


    .. grid-item-card::
        :img-top: _images/index/SOXS.png

        SOXS
        ^^^^

        Coupled with PyXSIM, SOXS is a instrument simulation tool for turning mock photon counts into realistic X-ray observations specific to the
        behavior of specific instruments like *CHANDRA*, *XMM-Newton*, and *NuSTAR*.

        +++

        .. button-link:: https://www.lynxobservatory.com/soxs
            :expand:
            :color: secondary
            :click-parent:

            Documentation





Indices and tables
==================

.. raw:: html

   <hr style="height:10px;background-color:black">


* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. |yt-project| image:: https://img.shields.io/static/v1?label="works%20with"&message="yt"&color="blueviolet"
   :target: https://yt-project.org

.. |docs| image:: https://img.shields.io/badge/docs-latest-brightgreen.svg
   :target: https://jzuhone.github.io/cluster_generator/build/html/index.html

.. |testing| image:: https://github.com/Eliza-Diggins/cluster_generator/actions/workflows/test.yml/badge.svg
.. |Pylint| image:: https://github.com/Eliza-Diggins/cluster_generator/actions/workflows/pylint.yml/badge.svg
.. |Github Page| image:: https://github.com/Eliza-Diggins/cluster_generator/actions/workflows/docs.yml/badge.svg
.. |coverage| image:: https://coveralls.io/repos/github/Eliza-Diggins/cluster_generator/badge.svg
   :target: https://coveralls.io/github/Eliza-Diggins/cluster_generator
.. |ncodes| image:: https://img.shields.io/static/v1?label="Implemented%20Sim.%20Codes"&message="7"&color="red"
    :target: https://jzuhone.github.io/cluster_generator/build/html/codes.html
