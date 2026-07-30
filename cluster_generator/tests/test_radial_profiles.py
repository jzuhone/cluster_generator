"""
Tests for the ``RadialProfile`` objects
"""

import logging
import os

import numpy as np
import pathlib as pt
import pytest
from numpy.testing import assert_array_equal

import cluster_generator.radial_profiles as rp
from cluster_generator.utils import integrate_mass

_params = {  # Stores all of the parameters for the generation of each of the test cases.
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
    "ad07_density_profile": [1, 1, 1, 1, 1],
    "ad07_temperature_profile": [1, 1, 1, 1],
    "broken_entropy_profile": [1, 1, 1],
    "walker_entropy_profile": [1, 1, 1, 1],
}


@pytest.mark.usefixtures("answer_dir")
class TestProfiles:
    """Base tests for core functionality of profiles"""

    prof_a = rp.constant_profile(5)
    prof_b = rp.power_law_profile(1, 50, 4)

    def test_dunder(self):
        _ = [self.prof_b.__str__(), self.prof_b.__repr__(), self.prof_b.__pow__(3)]

    def test_core(self, answer_dir):
        import matplotlib.pyplot as plt

        test_profile = self.prof_b.add_core(20, 3)

        fig, axes = plt.subplots(1, 1)
        test_profile.plot(1, 1000, fig=fig, ax=axes)
        fig.savefig(f"{answer_dir}/profile_core_test.png")

    def test_trunc(self, answer_dir):
        import matplotlib.pyplot as plt

        test_profile = self.prof_b.cutoff(500)

        fig, axes = plt.subplots(1, 1)
        test_profile.plot(1, 1000, fig=fig, ax=axes)
        fig.savefig(f"{answer_dir}/profile_trunc_test.png")


class TestUtilities:
    """This class is used to stress test the numerical nethods used here."""

    def test_integrate_mass(self):
        rr = np.geomspace(0.1, 10000, 1000)  # the integration domain

        def profile(x):
            return (1 / (2 * np.pi)) * (500 / x) * (1 / (500 + x) ** 3)

        def answer_profile(x):
            (x / (500 + x)) ** 2

        int = integrate_mass(profile, rr)

        np.testing.assert_allclose(int, answer_profile(rr))


@pytest.mark.filterwarnings("ignore:Casting")
@pytest.mark.skip(reason="Implementation not-complete.")
def test_profiles(answer_dir, answer_store):
    """Tests for consistency"""

    # -- checking if we are writing -- #
    output_directory = f"{answer_dir}/pkl"
    pt.Path(output_directory).mkdir(parents=True, exist_ok=True)

    # -- building and checking -- #
    x = np.geomspace(0.1, 1e6, 10000)

    for name, args in _params.items():
        try:
            _f = getattr(rp, name)(*args)
        except KeyError as e:
            raise KeyError(
                f"Failed to find profile {name} in globals. Did you write the name correctly?"
            ) from e

        if answer_store:
            _f.to_binary(os.path.join(output_directory, f"{name}.rp"))
        else:
            # -- actually checking -- #
            try:
                old = rp.RadialProfile.from_binary(os.path.join(output_directory, f"{name}.rp"))
            except FileNotFoundError:
                logging.info("The profile %s did not have a prior instance in the pkl file.", name)
                _f.to_binary(os.path.join(output_directory, f"{name}.rp"))
                continue

            assert_array_equal(old(x), _f(x), err_msg=f"Failed to match prior values for {name}.")
