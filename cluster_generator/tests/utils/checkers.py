"""
Utilities for Testing Cluster Models and Particles.

This module provides utility functions for testing the consistency and correctness of
`ClusterModel` and `ClusterParticles` objects in the cluster generator framework. It includes
functions to compare generated models and particles against stored reference data, or to store
new reference data for future tests. This is crucial in ensuring the reliability and accuracy
of simulations involving dark matter, stars, and gas in a cluster environment.

The module supports:
- Storing and validating `ClusterModel` objects, which represent the state and properties of a
  simulated cluster, including its fields and virial distribution functions.
- Storing and validating `ClusterParticles` objects, which contain the data for different types
  of particles (e.g., dark matter, gas, stars) in a simulation, ensuring their consistency with
  stored reference data.

Functions
---------
model_answer_testing(model, filename, answer_store, answer_dir)
    Test a model against stored answers or store new answers for future comparisons.

particle_answer_testing(parts, filename, answer_store, answer_dir)
    Test particles data against stored answers or store new answers for future comparisons.

Notes
-----
These utilities use `numpy.testing.assert_allclose` for comparing floating-point arrays to
within a relative tolerance of `1e-7`, ensuring high precision consistency between
generated data and stored answers.

Examples
--------
To test a `ClusterModel` object against stored answers:

    >>> from cluster_generator.tests.utils import model_answer_testing
    >>> model_answer_testing(my_model, "model.h5", False, "/path/to/answers")

To store new reference data for a `ClusterParticles` object:

    >>> from cluster_generator.tests.utils import particle_answer_testing
    >>> particle_answer_testing(my_particles, "particles.h5", True, "/path/to/answers")
"""

from pathlib import Path
from typing import Union

from numpy.testing import assert_allclose

from cluster_generator.model import ClusterModel
from cluster_generator.particles import ClusterParticles


def model_answer_testing(
    model: ClusterModel, filename: str, answer_store: bool, answer_dir: Union[str, Path]
) -> None:
    """
    Test a model against stored answers or store new answers for future comparisons.

    This function either writes the current state of a :py:class:`ClusterModel` to an HDF5 file
    or compares the current model against an existing stored model to ensure consistency.

    Parameters
    ----------
    model : :py:class:`ClusterModel`
        The cluster model to test or store.
    filename : str
        The name of the file where the model data is stored or will be stored.
    answer_store : bool
        If ``True``, store the model as a new answer file. If ``False``, compare the current model against the stored model.
    answer_dir : Union[str, Path]
        The directory where the answer file is located or will be stored.

    Raises
    ------
    AssertionError
        If the current model does not match the stored model when ``answer_store`` is ``False``.

    Notes
    -----
    This will check the ``fields`` of the model against the answer array by array to within
    single precision (``rtol=1e-7``). It then also checks the dm_virial and star_virial
    distribution functions for consistency to within the same margin of error.
    """
    # Ensure the answer directory is a Path object for consistency
    answer_path = Path(answer_dir) / filename

    if answer_store:
        # Write the current model to the specified HDF5 file
        model.write_model_to_h5(str(answer_path), overwrite=True)
    else:
        # Load the stored model from the specified HDF5 file
        old_model = ClusterModel.from_h5_file(str(answer_path))

        # Compare fields between the current and stored models
        for field in old_model.fields:
            assert_allclose(
                old_model[field],
                model[field],
                rtol=1e-7,
                err_msg=f"Mismatch in field '{field}'",
            )

        # Compare the dark matter virial distribution functions
        assert_allclose(
            old_model.dm_virial.df,
            model.dm_virial.df,
            rtol=1e-7,
            err_msg="Mismatch in dark matter virial distribution function",
        )

        # Compare the star virial distribution functions
        assert_allclose(
            old_model.star_virial.df,
            model.star_virial.df,
            rtol=1e-7,
            err_msg="Mismatch in star virial distribution function",
        )


def particle_answer_testing(
    parts: ClusterParticles,
    filename: str,
    answer_store: bool,
    answer_dir: Union[str, Path],
) -> None:
    """
    Test particles data against stored answers or store new answers for future comparisons.

    This function either writes the current state of `ClusterParticles` to a file or compares
    the current particle data against an existing stored file to ensure consistency.

    Parameters
    ----------
    parts : :py:class:`ClusterParticles`
        The cluster particles data to test or store.
    filename : str
        The name of the file where the particle data is stored or will be stored.
    answer_store : bool
        If True, store the particles as a new answer file. If False, compare the current particles against the stored particles.
    answer_dir : Union[str, Path]
        The directory where the answer file is located or will be stored.

    Raises
    ------
    AssertionError
        If the current particles data does not match the stored particles when `answer_store` is False.
    """
    # Ensure the answer directory is a Path object for consistency
    answer_path = Path(answer_dir) / filename

    if answer_store:
        # Write the current particle data to the specified file
        parts.write_particles(answer_path, overwrite=True)
    else:
        # Load the stored particle data from the specified file
        old_parts = ClusterParticles.from_file(answer_path)

        # Compare fields between the current and stored particle data
        for field in old_parts.fields:
            assert_allclose(
                old_parts[field],
                parts[field],
                rtol=1e-7,
                err_msg=f"Mismatch in particle field {field}",
            )
