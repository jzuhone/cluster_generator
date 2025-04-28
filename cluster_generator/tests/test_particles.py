"""
Test suite for particle generation and properties in cluster generator.

This module provides a comprehensive set of tests for verifying the particle generation
process in the cluster generator module. It checks the generation of different types of
particles (dark matter, star, gas), ensures correct initialization of particle properties,
and verifies the integrity of combined particle datasets.

Tests include:
- Generation of particles with specific properties.
- Validation of particle numbers and properties.
- Consistency checks across combined particle types.
"""

from typing import ClassVar

import numpy as np
import pytest
from numpy.random import RandomState

from cluster_generator.model import ClusterModel
from cluster_generator.particles import ClusterParticles
from cluster_generator.tests.test_models import get_model
from cluster_generator.tests.utils.checkers import particle_answer_testing

# Set up a random number generator with a fixed seed for reproducibility
prng = RandomState(25)


@pytest.mark.usefixtures("answer_dir", "answer_store", "temp_dir")
class TestParticleGeneration:
    """
    Test suite for particle generation in cluster generator.

    This test class is structured to verify various functionalities of particle generation,
    including the creation of dark matter, star, and gas particles, and their properties.

    Attributes
    ----------
    MODEL : :py:class:`ClusterModel`
        The generated cluster model for testing.
    DM_PARTICLES : :py:class:`ClusterParticles`
        The generated dark matter particles.
    STAR_PARTICLES : :py:class:`ClusterParticles`
        The generated star particles.
    GAS_PARTICLES : :py:class:`ClusterParticles`
        The generated gas particles.
    ALL_PARTICLES : :py:class:`ClusterParticles`
        The combined set of all generated particles.
    """

    MODEL: ClassVar["ClusterModel"] = None
    DM_PARTICLES: ClassVar[ClusterParticles] = None
    STAR_PARTICLES: ClassVar[ClusterParticles] = None
    GAS_PARTICLES: ClassVar[ClusterParticles] = None
    ALL_PARTICLES: ClassVar[ClusterParticles] = None

    @pytest.fixture(autouse=True)
    def setup_class(self, temp_dir):
        """
        Setup class method to initialize the model and generate particle sets.

        This method is automatically called once for the entire test class to initialize
        the necessary state, including generating dark matter, star, and gas particles.

        It ensures the model and particles are only generated once for all tests.

        Parameters
        ----------
        answer_dir : str
            Path to the answer directory provided by the fixture.
        """
        # Generate the model if not already created
        if self.__class__.MODEL is None:
            self.__class__.MODEL = get_model(temp_dir)

        # Generate dark matter, star, and gas particles
        self.__class__.DM_PARTICLES = self.__class__.MODEL.generate_dm_particles(
            1000, prng=prng
        )
        self.__class__.STAR_PARTICLES = self.__class__.MODEL.generate_star_particles(
            1000, prng=prng
        )
        self.__class__.GAS_PARTICLES = self.__class__.MODEL.generate_gas_particles(
            1000, prng=prng
        )

        # Combine all particle types into one dataset
        self.__class__.ALL_PARTICLES = (
            self.__class__.GAS_PARTICLES
            + self.__class__.DM_PARTICLES
            + self.__class__.STAR_PARTICLES
        )

    def test_particle_counts(self):
        """
        Test the number of particles generated for each particle type.

        Ensures that the correct number of dark matter, star, and gas particles are generated.
        """
        assert (
            self.__class__.DM_PARTICLES.num_particles["dm"] == 1000
        ), "Incorrect number of dark matter particles."
        assert (
            self.__class__.STAR_PARTICLES.num_particles["star"] == 1000
        ), "Incorrect number of star particles."
        assert (
            self.__class__.GAS_PARTICLES.num_particles["gas"] == 1000
        ), "Incorrect number of gas particles."

    def test_particle_properties(self):
        """
        Test the properties of generated particles.

        This test verifies that particle positions, velocities, and masses are initialized correctly.
        It checks for realistic physical values and ensures no NaNs or infinities.
        """
        for particle_type in ["dm", "star", "gas"]:
            positions = self.__class__.ALL_PARTICLES[particle_type, "particle_position"]
            velocities = self.__class__.ALL_PARTICLES[
                particle_type, "particle_velocity"
            ]
            masses = self.__class__.ALL_PARTICLES[particle_type, "particle_mass"]

            assert np.all(
                np.isfinite(positions)
            ), f"Non-finite positions found in {particle_type} particles."
            assert np.all(
                np.isfinite(velocities)
            ), f"Non-finite velocities found in {particle_type} particles."
            assert np.all(
                np.isfinite(masses)
            ), f"Non-finite masses found in {particle_type} particles."
            assert np.all(
                masses > 0
            ), f"Non-positive masses found in {particle_type} particles."

    def test_particle_boundaries(self):
        """
        Test that all particles are within the expected spatial boundaries.

        Ensures that generated particles are within a certain distance from the origin,
        indicating proper initialization within a physical domain.
        """
        max_distance = 50000  # Example spatial boundary in kpc

        for particle_type in ["dm", "star", "gas"]:
            positions = self.__class__.ALL_PARTICLES[particle_type, "particle_position"]
            distances = np.sqrt(np.sum(positions**2, axis=1))
            assert np.all(
                distances < max_distance
            ), f"{particle_type} particles found outside the spatial boundary."

    def test_particle_combination(self):
        """
        Test combining different types of particles into a single dataset.

        This test checks the integrity of particle data when dark matter, star, and gas particles are combined.
        Ensures no data corruption or loss during combination.
        """
        combined_particles = (
            self.__class__.DM_PARTICLES
            + self.__class__.STAR_PARTICLES
            + self.__class__.GAS_PARTICLES
        )
        total_num_particles = (
            self.__class__.DM_PARTICLES.num_particles["dm"]
            + self.__class__.STAR_PARTICLES.num_particles["star"]
            + self.__class__.GAS_PARTICLES.num_particles["gas"]
        )

        assert (
            combined_particles.num_particles["dm"]
            == self.__class__.DM_PARTICLES.num_particles["dm"]
        ), "Mismatch in dark matter particle count after combination."
        assert (
            combined_particles.num_particles["star"]
            == self.__class__.STAR_PARTICLES.num_particles["star"]
        ), "Mismatch in star particle count after combination."
        assert (
            combined_particles.num_particles["gas"]
            == self.__class__.GAS_PARTICLES.num_particles["gas"]
        ), "Mismatch in gas particle count after combination."
        assert (
            sum(combined_particles.num_particles.values()) == total_num_particles
        ), "Total particle count mismatch after combination."

    def test_particle_data_consistency(self, answer_store, answer_dir):
        """
        Test data consistency for all generated particles.

        This test compares generated particles against reference data to ensure consistency
        in data generation across different runs.

        Parameters
        ----------
        answer_store : bool
            Whether to store new answers or compare against existing ones.
        answer_dir : str
            Path to the answer directory.
        """
        particle_answer_testing(
            self.__class__.ALL_PARTICLES, "particles.h5", answer_store, answer_dir
        )
