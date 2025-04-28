Cluster Generator
=================

+-------------------+----------------------------------------------------------+
| **Code**          | |black| |isort| |yt-project| |Pre-Commit|                |
+-------------------+----------------------------------------------------------+
| **Documentation** | |docformatter| |NUMPSTYLE| |docs|                        |
+-------------------+----------------------------------------------------------+
| **GitHub**        | |CONTRIBUTORS| |COMMIT|                                  |
+-------------------+----------------------------------------------------------+
| **PyPi**          |                                                          |
+-------------------+----------------------------------------------------------+

`Cluster Generator <https://jzuhone.github.io/cluster_generator>`_ (CG) is a cross-platform Python library for generating
models of galaxy clusters suitable for use in idealized (magneto)hydrodynamic simulations as well as a variety of other contexts.
With the intention of providing an easy-to-use library without compromising on versatility, Cluster Generator implements a variety
of cluster construction methods, physical assumptions, and gravitational theories. Most importantly, Cluster Generator provides a variety of
frontends to export initial conditions in formats which are ready-to-run on many of the most popular astrophysical simulation codes.

Installation
------------
.. note::

    Cluster Generator is still in a pre-release stage of development. Be sure to communicate with the developers about
    the use of this software and any issues you face with installation or usage.

Cluster Generator is compatible with python versions above at least 3.8.
To install ``cluster_generator`` from the source code, simply clone the repository and install with pip as follows:

.. code-block:: shell

    $: pip install git+https://www.github.com/jzuhone/cluster_generator

Cluster Generator is intended to be a light-weight library; however, some dependencies are necessary (pip will install them for you):

- `unyt <http://unyt.readthedocs.org>`_ [Units and quantity manipulations]
- `numpy <http://www.numpy.org>`_ [Numerical operations]
- `scipy <http://www.scipy.org>`_ [Interpolation and curve fitting]
- `h5py <http://www.h5py.org>`_ [h5 file interaction]
- `tqdm <https://tqdm.github.io>`_ [Progress bars]
- `ruamel.yaml <https://yaml.readthedocs.io>`_ [yaml support]

Development
-----------

Cluster Generator is an open source project and community development is encouraged. If you are interested in assisting with development,
feel free to fork this repository and submit a pull request or simply reach out to discuss features or issues. If you're planning
on participating in the development process, we suggest familiarizing yourself with the API documentation before proceeding.

Acknowledging Cluster Generator
-------------------------------

Because Cluster Generator is still in active development, a methods paper has not yet been released. Users of this software
should therefore acknowledge its use in the prose of their work with a statement along the lines of

    The initial conditions were generated using the Cluster Generator library (version) developed by John ZuHone, which can
    be accessed at http://www.github.com/jzuhone/cluster_generator.


.. |yt-project| image:: https://img.shields.io/badge/works%20with-yt-purple
   :target: https://yt-project.org

.. |docs| image:: https://img.shields.io/badge/docs-latest-brightgreen
   :target: https://github.com/jzuhone/cluster_generator

.. |testing| image:: https://github.com/jzuhone/cluster_generator/actions/workflows/test.yml/badge

.. |black| image:: https://img.shields.io/badge/code%20style-black-000000
   :target: https://github.com/psf/black

.. |isort| image:: https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336
   :target: https://pycqa.github.io/isort/

.. |Pre-Commit| image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
   :target: https://github.com/pre-commit/pre-commit
   :alt: pre-commit

.. |CONTRIBUTORS| image:: https://img.shields.io/github/contributors/jzuhone/cluster_generator
    :target: https://github.com/jzuhone/cluster_generator/graphs/contributors

.. |COMMIT| image:: https://img.shields.io/github/last-commit/jzuhone/cluster_generator

.. |NUMPSTYLE| image:: https://img.shields.io/badge/%20style-numpy-459db9
    :target: https://numpydoc.readthedocs.io/en/latest/format.html

.. |docformatter| image:: https://img.shields.io/badge/%20formatter-docformatter-fedcba
    :target: https://github.com/PyCQA/docformatter
