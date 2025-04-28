"""
Test suite for the :py:class:`ClusterModel`.

This module contains a suite of tests to verify the functionality of the `ClusterModel` class
from the `cluster_generator` package. The tests cover:

- Model generation and storage: Ensure the model can be generated and stored correctly.
- Hydrostatic equilibrium: Verify that the model maintains hydrostatic equilibrium.
- Virial equilibrium: Test both dark matter and stellar components for virial equilibrium.
- Magnetic field settings: Validate the setting of magnetic field profiles.
- Interpolation and mass calculations: Test interpolation methods and mass calculations within specified radii.

The tests utilize pytest fixtures for setup and teardown, ensuring a consistent testing environment.
"""
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from cluster_generator import ClusterModel
from cluster_generator.tests.utils.checkers import model_answer_testing
from cluster_generator.tests.utils.generators import generate_model


def get_model(temp_dir):
    _model_path = Path(temp_dir) / TestClusterModel.MODEL_NAME

    if not _model_path.exists():
        model = generate_model()
        model.write_model_to_h5(_model_path)

    return ClusterModel.from_h5_file(_model_path)


@pytest.mark.usefixtures("answer_dir", "answer_store", "temp_dir")
class TestClusterModel:
    """
    Test suite for the :py:class:`ClusterModel`.

    This test class is structured to verify the various functionalities of the `ClusterModel` class,
    including storage and retrieval of model data, hydrostatic and virial equilibrium checks,
    magnetic field settings, and particle generation.

    Attributes
    ----------
    MODEL : ClassVar[:py:class:`ClusterModel`]
        The generated cluster model for testing.
    MODEL_NAME : ClassVar[str]
        The name of the model file to be saved in the temporary directory.
    """

    MODEL: ClassVar[ClusterModel] = None
    MODEL_NAME: ClassVar[str] = "base_model.h5"

    @pytest.fixture(autouse=True)
    def setup_class(self, temp_dir):
        """
        Setup fixture to generate a ClusterModel instance and save it to disk.

        This fixture is automatically called before each test method to initialize
        the necessary state, including generating a model and saving it to disk.

        Parameters
        ----------
        temp_dir : str
            Path to the temporary directory provided by the fixture.
        """
        # If the base model hasn't been generated yet, we do so.
        # Ensure that the model ends up in the temp directory so we have
        # if for future tests.
        if self.__class__.MODEL is None:
            self.__class__.MODEL = generate_model()
            _model_path = Path(temp_dir) / self.__class__.MODEL_NAME

            # Save the generated model to the temporary directory
            self.__class__.MODEL.write_model_to_h5(str(_model_path), overwrite=True)

        return self.__class__.MODEL

    def test_model(self, answer_store, answer_dir):
        """
        Test storing and verifying a ClusterModel against a stored answer.

        This test checks that the generated ClusterModel can be stored to an HDF5 file
        and then read back in to verify its consistency with the original model.

        Parameters
        ----------
        answer_store : bool
            If True, stores the model as a new answer file.
        answer_dir : str
            The directory where the answer file is located or will be stored.
        """
        model_answer_testing(
            self.__class__.MODEL, self.__class__.MODEL_NAME, answer_store, answer_dir
        )

    def test_hydrostatic_equilibrium(self):
        """
        Test that the ClusterModel maintains hydrostatic equilibrium.

        This test ensures that the generated model is in hydrostatic equilibrium by
        checking that the relative deviation is below a specified threshold.

        Raises
        ------
        AssertionError
            If the model's deviation from hydrostatic equilibrium exceeds the threshold.
        """
        deviation = self.__class__.MODEL.check_hse()
        assert np.all(
            deviation < 1.0e-4
        ), f"Model deviation from hydrostatic equilibrium is too high: {deviation.max()}"

    def test_dark_matter_virial_equilibrium(self):
        """
        Test that the dark matter component of the ClusterModel is in virial equilibrium.

        This test verifies that the dark matter distribution satisfies the virial theorem.

        Raises
        ------
        AssertionError
            If the dark matter does not satisfy virial equilibrium.
        """
        deviation = self.__class__.MODEL.check_dm_virial()[1]
        assert np.all(
            deviation < 1.0e-4
        ), f"Dark matter deviation from virial equilibrium is too high: {deviation.max()}"

    def test_star_virial_equilibrium(self):
        """
        Test that the stellar component of the ClusterModel is in virial equilibrium.

        This test checks if the stellar distribution satisfies the virial theorem,
        ensuring stability within the model.

        Raises
        ------
        AssertionError
            If the stellar component does not satisfy virial equilibrium.
        """
        if self.__class__.MODEL.star_virial:
            deviation = self.__class__.MODEL.check_star_virial()[1]
            assert np.all(
                deviation < 1.0e-4
            ), f"Stellar component deviation from virial equilibrium is too high: {deviation.max()}"

    @pytest.mark.parametrize("beta", [1.0, 10.0, 100.0])
    def test_magnetic_field_from_beta(self, beta):
        """
        Test setting a magnetic field profile from the plasma beta parameter.

        This test checks the creation of a magnetic field profile based on the specified
        plasma beta value, ensuring that the field is consistent with the pressure and beta.

        Parameters
        ----------
        beta : float
            The plasma beta parameter used to set the magnetic field.

        Raises
        ------
        AssertionError
            If the magnetic field strength does not match the expected value derived from the pressure.
        """
        self.__class__.MODEL.set_magnetic_field_from_beta(beta)
        B_expected = np.sqrt(8.0 * np.pi * self.__class__.MODEL["pressure"] / beta)
        B_actual = self.__class__.MODEL["magnetic_field_strength"].in_units("gauss")
        np.testing.assert_allclose(
            B_actual,
            B_expected,
            rtol=1e-5,
            err_msg=f"Magnetic field does not match expected values for beta={beta}",
        )

    def test_radius_density_interpolation(self):
        """
        Test the interpolation of field values at specific radii.

        This test checks that the interpolation method correctly returns the expected field values
        at specified radii.

        Raises
        ------
        AssertionError
            If the interpolated field value does not match the expected value.
        """
        test_radius = 100.0  # kpc
        expected_density = self.__class__.MODEL.find_field_at_radius(
            "density", test_radius
        )
        interpolated_density = np.interp(
            test_radius,
            self.__class__.MODEL["radius"].to_value("kpc"),
            self.__class__.MODEL["density"].to_value("Msun/kpc**3"),
        )
        assert np.isclose(
            interpolated_density, expected_density.value, rtol=1e-5
        ), f"Interpolated density does not match expected value at radius {test_radius} kpc."

    @pytest.mark.parametrize("r_max", [500, 1000, 1500])
    def test_mass_within_radius(self, r_max):
        """
        Test the calculation of mass within a given radius.

        This test verifies that the mass calculated within a specified radius is consistent
        with the expected total mass derived from the density profiles.

        Parameters
        ----------
        r_max : float
            The maximum radius in kpc within which to calculate the mass.

        Raises
        ------
        AssertionError
            If the calculated mass does not match the expected mass.
        """
        mass_within_radius = self.__class__.MODEL.mass_in_radius(r_max)
        total_mass = self.__class__.MODEL["total_mass"][
            self.__class__.MODEL["radius"].to_value("kpc") < r_max
        ][-1]
        assert np.isclose(
            mass_within_radius["total"].value, total_mass.value, rtol=1e-5
        ), f"Calculated mass within {r_max} kpc does not match expected total mass."
