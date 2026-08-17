#!/usr/bin/env python
from distutils.extension import Extension

import numpy as np
from Cython.Build import cythonize
from setuptools import setup

cython_utils = Extension(
    "cluster_generator.opt.cython_utils",
    sources=["cluster_generator/opt/cython_utils.pyx"],
    language="c",
    libraries=["m"],
    include_dirs=[np.get_include()],
)

setup(
    name="cluster_generator",
    packages=["cluster_generator"],
    version="0.1.0",
    description="Generating equilbrium models of galaxy clusters.",
    author="John ZuHone",
    author_email="jzuhone@gmail.com",
    url="https://github.com/jzuhone/cluster_generator",
    download_url="https://github.com/jzuhone/cluster_generator/tarball/0.1.0",
    install_requires=[
        "numpy>2.0,<3.0",
        "scipy",
        "yt",
        "unyt",
        "cython",
        "ruamel.yaml",
        "h5py",
        "kspace",
    ],
    extras_require={
        # Exact bounded Voronoi cell volumes for AREPO IC generation
        # (codes.setup_arepo_ics / relax_arepo_ics).  pyvoro2 wraps voro++
        # and ships wheels for modern Python.
        "arepo": ["pyvoro2"],
        # AMR grids/plotfiles (datasets.AMRClusterDataset,
        # amr_hierarchy.AMRHierarchy).  pyamrex is not distributed on PyPI --
        # this entry documents the requirement, but install it with
        # `conda install -c conda-forge pyamrex` (a 'nompi' build is enough).
        "amr": [],
    },
    classifiers=[
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    include_package_data=True,
    ext_modules=cythonize([cython_utils]),
)
