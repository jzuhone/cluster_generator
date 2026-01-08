"""
Testing module for the :py:mod:``data_structures`` module.
"""

import os

import pytest

import cluster_generator.frontend
from cluster_generator import ClusterModel
from cluster_generator.tests.utils import generate_model, h5_answer_testing


@pytest.mark.usefixtures("answer_dir", "answer_store", "temp_dir")
class Test_YTHDF5:
    """
    Tests for the :py:class:`data_structures.YTHDF5`` class.
    """

    @pytest.mark.slow
    def test_construction(self, answer_dir: str, answer_store: bool, temp_dir: str):
        """
        Test YTHDF5 construction process for the base model.
        """
        # Loading the model from tmp directory.
        model_path = os.path.join(temp_dir, "base_model.h5")

        if not os.path.exists(model_path):
            model = generate_model()
        else:
            model = ClusterModel.from_h5_file(model_path)

        # construct the hdf5 yt structure from this model
        # This is done in the tmp directory with a low resolution (to assure rapid speed).
        model.create_dataset(os.path.join(temp_dir, "yt_dataset.h5"), overwrite=True)

        # Test the new model against the one stored in the answer directory.
        h5_answer_testing(
            os.path.join(temp_dir, "yt_dataset.h5"),
            "yt_dataset.h5",
            answer_store,
            answer_dir,
        )

    def _load_yt(self, answer_dir: str, temp_dir: str) -> cluster_generator.frontend.ClusterGeneratorDataset:
        # Locating the model.
        # If we've run test_construction, then there should be one in the temp file; but otherwise, we should
        # look in the answer directory.
        ds_path = os.path.join(temp_dir, "yt_dataset.h5")

        if not os.path.exists(ds_path):
            ds_path = os.path.join(answer_dir, "yt_dataset.h5")
            assert os.path.exists(ds_path), f"{ds_path} does not exist"

        import yt

        return yt.load(ds_path)

    def test_yt_load(self, answer_dir: str, answer_store: bool, temp_dir: str):
        """
        Try to load the model in yt.
        """
        _ = self._load_yt(answer_dir, temp_dir)

    def test_yt_fields(self, answer_dir, answer_store: bool, temp_dir: str):
        """
        Check that all of the anticipated fields are correctly loaded.
        """
        _expected_fields = [
            ("gas", "density"),
            ("gas", "momentum_density_x"),
            ("gas", "momentum_density_y"),
            ("gas", "momentum_density_z"),
            ("gas", "velocity_x"),
            ("gas", "velocity_y"),
            ("gas", "velocity_z"),
            ("stellar", "velocity_x"),
            ("stellar", "velocity_y"),
            ("stellar", "velocity_z"),
            ("dark_matter", "velocity_x"),
            ("dark_matter", "velocity_y"),
            ("dark_matter", "velocity_z"),
            ("gas", "temperature"),
            ("gas", "pressure"),
        ]

        ds = self._load_yt(answer_dir, temp_dir)
        for field in _expected_fields:
            assert field in ds.derived_field_list, f"{field} is not in the dataset."
