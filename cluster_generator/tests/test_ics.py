"""
Test suite for ``ClusterICs`` class.

This module contains tests for the ``ClusterICs`` class in the ``cluster_generator.ics`` module. The tests are structured to verify the functionalities of ``ClusterICs``, including:

- Generation of particles for single and multiple clusters.
- Resampling of particles after initial condition setups.
- Calculation of binary cluster centers.

The tests use fixtures to manage test directories and answer files, ensuring a clean testing environment and reproducibility.
"""

from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
from numpy.random import RandomState

from cluster_generator.ics import ClusterICs, compute_centers_for_binary
from cluster_generator.particles import ClusterParticles
from cluster_generator.tests.test_models import get_model
from cluster_generator.tests.utils.checkers import particle_answer_testing

# Set up a random number generator with a fixed seed for reproducibility
prng = RandomState(25)


@pytest.mark.usefixtures("answer_dir", "answer_store", "temp_dir")
class TestClusterICs:
    """
    Test suite for the ``ClusterICs`` class.

    This test class verifies the functionalities of the ``ClusterICs`` class, including
    the generation of particles for single and multiple clusters, the resampling of particles, and
    binary cluster center calculations.

    Attributes
    ----------
    MODEL_NAME : str
        The filename for the model used in testing.
    SINGLE_ICS : :py:class:`ClusterICs`
        The initial conditions object for a single cluster.
    DOUBLE_ICS : :py:class:`ClusterICs`
        The initial conditions object for two interacting clusters.
    SINGLE_PARTS : :py:class:`ClusterParticles`
        The generated particles for a single cluster.
    DOUBLE_PARTS : :py:class:`ClusterParticles`
        The generated particles for two interacting clusters.
    """

    MODEL_NAME: ClassVar[str] = "base_model.h5"
    SINGLE_ICS: ClassVar[ClusterICs] = None
    DOUBLE_ICS: ClassVar[ClusterICs] = None
    SINGLE_PARTS: ClassVar["ClusterParticles"] = None
    DOUBLE_PARTS: ClassVar["ClusterParticles"] = None

    @pytest.fixture(autouse=True)
    def setup_class(self, answer_dir, temp_dir):
        """
        Setup class method to initialize ``ClusterICs`` objects and generate particles.

        This method is automatically called once for the entire test class to initialize
        the necessary state, including generating initial conditions for both single and
        double clusters.

        Parameters
        ----------
        answer_dir : str
            Path to the answer directory provided by the fixture.
        temp_dir : str
            Path to the temporary directory provided by the fixture.
        """
        # Setup paths and cluster parameters
        model_path = Path(temp_dir) / self.__class__.MODEL_NAME

        if not model_path.exists():
            _ = get_model(temp_dir)

        num_particles_single = {k: 1000 for k in ["dm", "star", "gas"]}
        num_particles_double = {k: 2000 for k in ["dm", "star", "gas"]}

        # Initialize the ClusterICs for a single cluster
        self.__class__.SINGLE_ICS = ClusterICs(
            basename="single",
            num_halos=1,
            profiles=model_path,
            center=[0.0, 0.0, 0.0],
            velocity=[0.0, 0.0, 0.0],
            num_particles=num_particles_single,
        )

        # Compute centers and velocities for a binary cluster system
        center1, center2 = compute_centers_for_binary([0.0, 0.0, 0.0], 3000.0, 500.0)
        velocity1 = [500.0, 0.0, 0.0]
        velocity2 = [-500.0, 0.0, 0.0]

        # Initialize the ClusterICs for two interacting clusters
        self.__class__.DOUBLE_ICS = ClusterICs(
            basename="double",
            num_halos=2,
            profiles=[model_path, model_path],
            center=[center1, center2],
            velocity=[velocity1, velocity2],
            num_particles=num_particles_double,
        )

        # Generate particles for a single cluster and save to disk
        self.__class__.SINGLE_PARTS = self.__class__.SINGLE_ICS.setup_particle_ics(
            output_directory=temp_dir, prng=prng
        )
        self.__class__.SINGLE_PARTS.write_particles(
            str(Path(temp_dir) / "single_particles.h5"), overwrite=True
        )

        # Generate particles for two interacting clusters and save to disk
        self.__class__.DOUBLE_PARTS = self.__class__.DOUBLE_ICS.setup_particle_ics(
            output_directory=temp_dir, prng=prng
        )
        self.__class__.DOUBLE_PARTS.write_particles(
            str(Path(temp_dir) / "double_particles.h5"), overwrite=True
        )

    def test_single_cluster_particles(self, answer_store, answer_dir):
        """
        Test particle generation for a single cluster.

        This test verifies the correctness of the particle generation for a single cluster
        by comparing the generated particles against stored answers.

        Parameters
        ----------
        answer_store : bool
            Whether to store new answers or compare against existing ones.
        answer_dir : str
            Path to the answer directory.
        """
        particle_answer_testing(
            self.__class__.SINGLE_PARTS, "single_particles.h5", answer_store, answer_dir
        )

    def test_double_cluster_particles(self, answer_store, answer_dir):
        """
        Test particle generation for two interacting clusters.

        This test verifies the correctness of the particle generation for two interacting clusters
        by comparing the generated particles against stored answers.

        Parameters
        ----------
        answer_store : bool
            Whether to store new answers or compare against existing ones.
        answer_dir : str
            Path to the answer directory.
        """
        particle_answer_testing(
            self.__class__.DOUBLE_PARTS, "double_particles.h5", answer_store, answer_dir
        )

    def test_compute_centers_for_binary(self):
        """
        Test computation of binary cluster centers.

        This test checks the calculation of the central positions for two clusters
        with given parameters to ensure they are computed correctly.
        """
        center1, center2 = compute_centers_for_binary([0.0, 0.0, 0.0], 3000.0, 500.0)
        assert np.allclose(
            center1, [-np.sqrt(1500**2 - 250**2), -250.0, 0.0]
        ), "Center1 calculation is incorrect."
        assert np.allclose(
            center2, [np.sqrt(1500**2 - 250**2), 250.0, 0.0]
        ), "Center2 calculation is incorrect."

    def test_resample_particle_ics(self, temp_dir):
        """
        Test resampling of particle initial conditions.

        This test checks the functionality of resampling particle initial conditions
        after relaxation or reconfiguration. Ensures particle data is consistent post-resampling.

        Parameters
        ----------
        temp_dir : str
            Path to the temporary directory provided by the fixture.
        """
        # Example resampling test for single cluster particles
        resampled_parts = self.__class__.SINGLE_ICS.resample_particle_ics(
            self.__class__.SINGLE_PARTS
        )
        resampled_path = Path(temp_dir) / "resampled_single_particles.h5"
        resampled_parts.write_particles(str(resampled_path), overwrite=True)
        assert resampled_path.exists(), "Resampled particle file does not exist."
        assert np.all(
            [n > 0 for n in resampled_parts.num_particles.values()]
        ), "No particles found after resampling."
