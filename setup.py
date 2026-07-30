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
opt_utils = Extension(
    "cluster_generator.opt.structures",
    sources=["cluster_generator/opt/structures.pyx"],
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
    ],
    classifiers=[
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    include_package_data=True,
    ext_modules=cythonize([cython_utils, opt_utils]),
)
