import os
import weakref

import numpy as np
import pathlib as pt
from yt.data_objects.index_subobjects.grid_patch import AMRGridPatch
from yt.data_objects.static_output import Dataset
from yt.geometry.grid_geometry_handler import GridIndex
from yt.utilities.file_handler import HDF5FileHandler

from .fields import ClusterGeneratorFieldInfo


class ClusterGeneratorGrid(AMRGridPatch):
    _id_offset = 0

    def __init__(self, id, index, level):
        super().__init__(id, filename=index.index_filename, index=index)
        self.Parent = None
        self.Children = []
        self.Level = level


class ClusterGeneratorHierarchy(GridIndex):
    grid = ClusterGeneratorGrid

    def __init__(self, ds, dataset_type="cluster_generator"):
        self.dataset_type = dataset_type
        self.dataset = weakref.proxy(ds)
        # for now, the index file is the dataset!
        self._handle = ds._handle
        self.index_filename = self.dataset.parameter_filename
        self.directory = os.path.dirname(self.index_filename)
        # float type for the simulation edges and must be float64 now
        self.float_type = np.float64
        super().__init__(ds, dataset_type)

    def _detect_output_fields(self):
        # This needs to set a self.field_list that contains all the available,
        # on-disk fields. No derived fields should be defined here.
        # NOTE: Each should be a tuple, where the first element is the on-disk
        # fluid type or particle type.  Convention suggests that the on-disk
        # fluid type is usually the dataset_type and the on-disk particle type
        # (for a single population of particles) is "io".
        _available_field_list = [
            ("cluster_generator", "density"),
            ("cluster_generator", "dark_matter_density"),
            ("cluster_generator", "stellar_density"),
            ("cluster_generator", "pressure"),
            ("cluster_generator", "momentum_density_x"),
            ("cluster_generator", "momentum_density_y"),
            ("cluster_generator", "momentum_density_z"),
            ("cluster_generator", "magnetic_pressure"),
        ]

        self.field_list = []

        # check for each of these fields in the file.
        for field in _available_field_list:
            if field[1] not in self._handle["grid"].keys():
                # the field isn't in the HDF5 file.
                continue

            self.field_list.append(field)

    def _count_grids(self):
        # Determine the number of grids in the dataset.
        # For cluster_generator, this is just obtained from the size of the
        # chunkmap dataset in the hdf5 file.
        self.num_grids = self.dataset.parameters["NGRID"]

    def _parse_index(self):
        # This needs to fill the following arrays, where N is self.num_grids:
        #   self.grid_left_edge         (N, 3) <= float64
        #   self.grid_right_edge        (N, 3) <= float64
        #   self.grid_dimensions        (N, 3) <= int
        #   self.grid_particle_count    (N, 1) <= int
        #   self.grid_levels            (N, 1) <= int
        #   self.grids                  (N, 1) <= grid objects
        #   self.max_level = self.grid_levels.max()
        # ------------------------------------------------------------------
        # Notes:
        # We require a grid divisible by the chunksize, so all grids are 1
        # cube-chunksize.
        self.grid_dimensions = (
            np.ones((self.num_grids, 3), dtype="int32") * (self.dataset.parameters["chunksize"])
        )

        # -- Determining the grid edges -- #
        # The chunkmap provides the grid_id for the edges, but that still
        # needs to be converted via the bounding box.

        # Determine the grid size from parameters
        _dd = self.dataset.parameters["domain_dimensions"]
        _dx = (
            self.dataset.domain_right_edge - self.dataset.domain_left_edge
        ) / _dd  # The size of a single cell along each axis.

        self.grid_left_edge = (
            self._handle["chunks"]["chunkmap"][:, 0, :].T * _dx
        ) + self.dataset.domain_left_edge
        self.grid_right_edge = (
            self._handle["chunks"]["chunkmap"][:, 1, :].T * _dx
        ) + self.dataset.domain_left_edge

        self.grid_particle_count = np.zeros((self.num_grids, 1), dtype="int32")
        self.grid_levels = np.zeros((self.num_grids, 1), dtype="int32")
        self.max_level = 0
        self.grids = np.empty(self.num_grids, dtype="object")

        for i in range(self.num_grids):
            self.grids[i] = self.grid(i, self, self.grid_levels[i, 0])

    def _populate_grid_objects(self):
        # the minimal form of this method is
        #
        # for g in self.grids:
        #     g._prepare_grid()
        #     g._setup_dx()
        #
        # This must also set:
        #   g.Children <= list of child grids
        #   g.Parent   <= parent grid
        # This is handled by the frontend because often the
        # children must be identified.
        for g in self.grids:
            g._prepare_grid()
            g._setup_dx()


class ClusterGeneratorDataset(Dataset):
    _index_class = ClusterGeneratorHierarchy
    _field_info_class = ClusterGeneratorFieldInfo

    # names of additional modules that may be required to load data from disk
    _load_requirements: list[str] = ["h5py"]
    _handle = None

    def __init__(
        self,
        filename,
        dataset_type="cluster_generator",
        storage_filename=None,
        units_override=None,
        unit_system="cgs",
        default_species_fields=None,
    ):
        self._handle = HDF5FileHandler(filename)
        self.fluid_types += ("cluster_generator",)
        super().__init__(
            filename,
            dataset_type,
            units_override=units_override,
            unit_system=unit_system,
            default_species_fields=default_species_fields,
        )
        self.storage_filename = storage_filename

        # refinement factor between a grid and its subgrid
        # self.refine_by = 2

    def _set_code_unit_attributes(self):
        # Set the frontend's convention for various data units.
        # cluster_generator uniformly applies extragalactic coordinates (kpc,Msun,Myr,etc.)
        # Thus, these can al be set manually.
        self.length_unit = self.quan(1.0, "kpc")
        self.mass_unit = self.quan(1.0, "Msun")
        self.time_unit = self.quan(1.0, "Myr")
        self.velocity_unit = self.quan(1.0, "kpc/Myr")
        self.magnetic_unit = self.quan(1.0, "gauss")

    def _parse_parameter_file(self):
        # Loading the hdf5 file attributes.
        self.parameters = {}
        attribute_keys = self._handle.attrs.keys()
        for attribute_key in attribute_keys:
            self.parameters[attribute_key] = self._handle.attrs[attribute_key]

        self.domain_left_edge = self.parameters["bbox"][:, 0]
        self.domain_right_edge = self.parameters["bbox"][:, 1]
        self.dimensionality = 3
        self.domain_dimensions = self.parameters["domain_dimensions"]
        self.current_time = 0
        self.cosmological_simulation = 0
        self.current_redshift = 0
        self.omega_lambda = 0
        self.omega_matter = 0
        self.hubble_constant = 0
        self._periodicity = (True,) * 3
        self.mu = self.parameters.get("mu", 1.2)  # mean molecular weight.

        # -- pulling the chunkmap information -- #
        self.parameters["NGRID"] = self._handle["chunks"]["chunkmap"].shape[-1]

    @classmethod
    def _is_valid(cls, filename: str, *args, **kwargs) -> bool:
        # This accepts a filename or a set of arguments and returns True or
        # False depending on if the file is of the type requested.
        #
        # The functionality in this method should be unique enough that it can
        # differentiate the frontend from others. Sometimes this means looking
        # for specific fields or attributes in the dataset in addition to
        # looking at the file name or extension.
        import h5py

        _required_attributes = ["domain_dimensions", "bbox", "chunksize", "model_count"]
        if pt.Path(filename).suffix not in [".h5", ".hdf5"]:
            return False  # Not the correct file type.

        with h5py.File(filename, "r") as f:
            attr_keys = list(f.attrs.keys())

            if any(attrib not in attr_keys for attrib in _required_attributes):
                return False

        return True

    def close(self):
        self._handle.close()
