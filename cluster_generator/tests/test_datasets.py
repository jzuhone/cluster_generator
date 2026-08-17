"""
Testing module for the :py:mod:``datasets`` module.
"""

import os

import pytest

import cluster_generator.frontend
from cluster_generator.tests.utils import generate_model


@pytest.fixture(scope="class")
def dataset_path(tmp_path_factory):
    """
    Build a low-resolution :py:class:`datasets.AMRClusterDataset`
    plotfile, with one locally-refined region around the cluster core,
    shared by every test in :class:`Test_AMRClusterDataset`.
    """
    model = generate_model()
    path = tmp_path_factory.mktemp("cg_dataset") / "plt_dataset"
    model.create_dataset(
        str(path),
        domain_dimensions=(32, 32, 32),
        overwrite=True,
        regions=[{"center": [0.0, 0.0, 0.0], "width": 2000.0, "num_levels": 1}],
        max_grid_size=16,
    )
    return str(path)


class Test_AMRClusterDataset:
    """
    Tests for the :py:class:`datasets.AMRClusterDataset` class.
    """

    @pytest.mark.slow
    def test_construction(self, dataset_path: str):
        """
        Test AMRClusterDataset construction process for the base model.
        """
        assert os.path.exists(dataset_path)
        assert os.path.exists(os.path.join(dataset_path, "Header"))

    def test_yt_load(self, dataset_path: str):
        """
        Try to load the model in yt, and confirm it's recognized as a
        :py:class:`cluster_generator.frontend.ClusterGeneratorDataset`
        (rather than falling back to a generic ``BoxlibDataset``) with the
        expected two AMR levels.
        """
        import yt

        ds = yt.load(dataset_path)
        assert isinstance(ds, cluster_generator.frontend.ClusterGeneratorDataset)
        assert ds.max_level == 1

    def test_yt_fields(self, dataset_path: str):
        """
        Check that all of the anticipated fields are correctly loaded.
        """
        import yt

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

        ds = yt.load(dataset_path)
        for field in _expected_fields:
            assert field in ds.derived_field_list, f"{field} is not in the dataset."
