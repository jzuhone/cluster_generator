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


Contents
========
.. raw:: html

   <hr style="height:10px;background-color:black">

.. toctree::
   :maxdepth: 1

   getting_started
   models
   collections
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
