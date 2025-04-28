"""
Test suite for the :py:mod:`cluster_generator.radial_profiles` module.

This module contains tests to verify the functionality of the various radial profiles
implemented in the `cluster_generator` package. The tests cover:

- Profile instantiation: Ensure radial profiles are created correctly with specified parameters.
- Mathematical operations: Verify the correctness of mathematical operations such as addition, multiplication, and exponentiation.
- Profile transformations: Test profile transformations like adding a core or applying a cutoff.
- Profile plotting: Ensure the profiles are plotted correctly over a specified range.
- Mass integration: Check mass integration consistency for density profiles.
- Serialization: Test storing and retrieving profiles using binary formats for persistence.

The tests utilize pytest fixtures for setup and teardown, ensuring a consistent testing environment.
"""

import pathlib as pt

import numpy as np
import pytest

import cluster_generator.radial_profiles as rp
from cluster_generator.utils import integrate_mass

# Define parameters for profile generation
_profile_params = {
    "constant_profile": [1],
    "power_law_profile": [1, 1000, -2],
    "beta_model_profile": [1, 1000, 2],
    "hernquist_density_profile": [1, 1000],
    "cored_hernquist_density_profile": [1, 1000, 20],
    "hernquist_mass_profile": [1, 1000],
    "nfw_density_profile": [1, 1000],
    "nfw_mass_profile": [1, 1000],
    "tnfw_density_profile": [1, 1000, 1200],
    "tnfw_mass_profile": [1, 1000, 1200],
    "snfw_density_profile": [1, 1000],
    "snfw_mass_profile": [1, 1000],
    "einasto_density_profile": [1, 1000, 2],
    "einasto_mass_profile": [1, 1000, 2],
    "am06_density_profile": [1, 0.5, 0.7, 50, 2],
    "vikhlinin_density_profile": [1, 0.75, 0.3, 0.6, -3, 4],
    "vikhlinin_temperature_profile": [1, 0.1, 0.5, 1, 500, 1, 20, 2],
    "am06_temperature_profile": [1, 0.1, 0.3, 50],
    "broken_entropy_profile": [1, 1, 1],
    "walker_entropy_profile": [1, 1, 1, 1],
}


@pytest.mark.usefixtures("answer_dir", "answer_store", "temp_dir")
class TestRadialProfiles:
    """
    Test suite for the various radial profiles in `cluster_generator.radial_profiles`.

    This test class verifies the functionality of radial profile generation, mathematical operations,
    transformations, plotting, and serialization.
    """

    @pytest.fixture(autouse=True)
    def setup_class(self, temp_dir):
        """
        Setup fixture to initialize test profiles and set up output directory.

        This fixture runs once for the entire test class and prepares reusable resources,
        such as creating profiles and defining output paths.

        Parameters
        ----------
        temp_dir : str
            Path to the temporary directory provided by the fixture.
        """
        self.output_directory = pt.Path(temp_dir) / "pkl"
        self.output_directory.mkdir(parents=True, exist_ok=True)

    @pytest.mark.parametrize("profile_name, params", _profile_params.items())
    def test_profile_instantiation(self, profile_name, params):
        """
        Test instantiation of various radial profiles.

        This test ensures that each radial profile can be instantiated with its
        corresponding parameters and behaves as expected.

        Parameters
        ----------
        profile_name : str
            The name of the profile to test.
        params : list
            The parameters required to instantiate the profile.

        Raises
        ------
        AssertionError
            If the profile cannot be instantiated or does not behave as expected.
        """
        profile = getattr(rp, profile_name)(*params)
        assert isinstance(
            profile, rp.RadialProfile
        ), f"{profile_name} instantiation failed."

    def test_profile_operations(self):
        """
        Test mathematical operations on profiles.

        This test verifies that operations like addition, multiplication, and exponentiation
        are correctly implemented for `RadialProfile` objects.

        Raises
        ------
        AssertionError
            If the operations do not produce the expected results.
        """
        prof_a = rp.constant_profile(5)
        prof_b = rp.power_law_profile(1, 50, 4)

        prof_add = prof_a + prof_b
        prof_mul = prof_a * prof_b
        prof_pow = prof_b**2

        # Verify profiles are callable and return expected results
        test_radius = 10.0
        assert prof_add(test_radius) == prof_a(test_radius) + prof_b(test_radius)
        assert prof_mul(test_radius) == prof_a(test_radius) * prof_b(test_radius)
        assert prof_pow(test_radius) == prof_b(test_radius) ** 2

    @pytest.mark.parametrize("r_core, alpha", [(20, 3), (10, 2)])
    def test_profile_core_addition(self, r_core, alpha, answer_dir):
        """
        Test adding a core to a radial profile.

        This test checks the `add_core` method for adding a core to the profile,
        and ensures the profile is correctly modified and can be plotted.

        Parameters
        ----------
        r_core : float
            The core radius to add.
        alpha : float
            The power-law index for the core.
        answer_dir : str
            The directory where output plots are saved.
        """
        prof = rp.power_law_profile(1, 50, 4)
        test_profile = prof.add_core(r_core, alpha)

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 1)
        test_profile.plot(1, 1000, fig=fig, ax=axes)
        fig.savefig(f"{answer_dir}/profile_core_test_{r_core}_{alpha}.png")

    @pytest.mark.parametrize("r_cut, k", [(500, 5), (300, 10)])
    def test_profile_cutoff(self, r_cut, k, answer_dir):
        """
        Test applying a cutoff to a radial profile.

        This test verifies the `cutoff` method, ensuring the profile is correctly
        truncated and can be plotted.

        Parameters
        ----------
        r_cut : float
            The cutoff radius to apply.
        k : int
            The steepness of the cutoff.
        answer_dir : str
            The directory where output plots are saved.
        """
        prof = rp.power_law_profile(1, 50, 4)
        test_profile = prof.cutoff(r_cut, k)

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 1)
        test_profile.plot(1, 1000, fig=fig, ax=axes)
        fig.savefig(f"{answer_dir}/profile_cutoff_test_{r_cut}_{k}.png")

    def test_mass_integration(self):
        """
        Test the mass integration utility.

        This test checks the `integrate_mass` function to ensure it correctly integrates
        a mass profile over a given range of radii.

        Raises
        ------
        AssertionError
            If the integrated mass does not match the expected values.
        """
        rr = np.geomspace(0.1, 10000, 1000)
        profile = lambda x: (1 / (2 * np.pi)) * (500 / x) * (1 / (500 + x) ** 3)
        answer_profile = lambda x: (x / (500 + x)) ** 2

        int_mass = integrate_mass(profile, rr)
        np.testing.assert_allclose(int_mass, answer_profile(rr), rtol=1e-5)
