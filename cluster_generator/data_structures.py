"""
IO Backend for integrating :py:class:`cluster_generator.model.ClusterModel` and :py:class:`cluster_generator.ics.ClusterICs` instances
with external packages like :py:mod:`yt`.
"""
import os
import pathlib as pt
from contextlib import contextmanager

import h5py
import numpy as np
from scipy.interpolate import dfitpack
from tqdm import tqdm

from cluster_generator import ClusterModel
from cluster_generator.opt.structures import construct_chunks, dump_field_to_hdf5
from cluster_generator.utils import cgparams, ensure_ytarray, mylog

fields = {
    "density": "Msun/kpc**3",
    "dm_density": "Msun/kpc**3",
    "stellar_density": "Msun/kpc**3",
    "pressure": "Msun/(kpc*Myr**2)",
    "momentum_density_x": "Msun/(Myr*kpc**2)",
    "momentum_density_y": "Msun/(Myr*kpc**2)",
    "momentum_density_z": "Msun/(Myr*kpc**2)",
    "magnetic_pressure": "Msun/(kpc*Myr**2)",
}


class YTHDF5:
    """
    Wrapper class for YT style HDF5 files. Used to manage the writing of :py:class:`model.ClusterModel` instances
    to YT datasets.
    """

    def __init__(self, filename):
        """
        Initializes the :py:class:`data_structures.YTHDF5` data structure.

        Parameters
        ----------
        filename: str
            The file path from which to open the YT-HDF file.
        """
        assert os.path.exists(filename), f"The file {filename} doesn't appear to exist."
        fo = h5py.File(filename, "a")

        #: The path to the base hdf5 file.
        self.filename = filename
        #: The dimensions of the grid for each of the fields.
        self.domain_dimensions = None
        #: the bounding box for the model.
        self.bbox = None
        #: the number of models added to the HDF5 file.
        self.model_count = None
        #: the size of each chunk (maximum).
        self.chunksize = None

        for k, v in fo.attrs.items():
            setattr(self, k, v)

        #: The chunkmap, provides information to the backend for IO operations.
        self.chunkmap = fo["chunks"]["chunkmap"][:]

        fo.close()

    def __str__(self):
        return f"<YTHDF5 File @ {self.filename}>"

    def __repr__(self):
        return self.__str__()

    def __getitem__(self, item):
        with self.open() as file:
            return file.attrs[item]

    def __setitem__(self, key, value):
        with self.open() as file:
            file.attrs[key] = value

    @classmethod
    @contextmanager
    def temporary(cls, domain_dimensions=(512, 512, 512), bbox=None, chunksize=64):
        """
        Create a temporary :py:class:`data_structures.YTHDF5` instance.

        .. note::

            This method can be used as a context manager.

        Parameters
        ----------
        domain_dimensions: tuple
            The dimensions of the grid for each of the fields.
        bbox: array-like
            The bounding box of the simulation region. Size should be ``3,2``.
        chunksize: int
            The maximum size of a given chunk in the HDF5 file.

        Returns
        -------
        :py:class:`data_structures.YTHDF5`
        """
        import tempfile

        path = tempfile.NamedTemporaryFile(delete=False)
        built_cls = cls.build(
            path.name,
            domain_dimensions=domain_dimensions,
            bbox=bbox,
            overwrite=True,
            chunksize=chunksize,
        )
        yield built_cls
        os.remove(path.name)

    @classmethod
    def load(cls, filename):
        """
        Load an existing :py:class:`data_structures.YTHDF5` instance from file.

        Parameters
        ----------
        filename: str
            The path to the file.

        Returns
        -------
        :py:class:`data_structures.YTHDF5`

        """
        return cls(filename)

    @classmethod
    def build(
        cls,
        filename,
        domain_dimensions=(512, 512, 512),
        bbox=None,
        overwrite=False,
        chunksize=64,
    ):
        """
        Create a new :py:class:`data_structures.YTHDF5` instance.

        Parameters
        ----------
        filename: str
            The filename for the underlying file.
        domain_dimensions: array-like, optional
            The dimensions of the grid along each axis (3-tuple)
        bbox: array-like
            The bounding box of the grid. Should be of size ``(3,2)``.
        overwrite: bool, optional
            If ``True``, then the file will be overwritten if it already exists. Default ``False``.
        chunksize: int, optional
            The maximum size of a chunk. Chunking is used to conserve memory during computations. A higher ``chunksize`` will
            increase the memory usage but decrease computation time, a lower value will do the opposite. Default is ``64``.

        Returns
        -------
        :py:class:`data_structures.YTHDF5`
            The generated YT dataset.
        """
        filename = pt.Path(filename)

        # Sanity checking
        domain_dimensions = np.array(domain_dimensions, dtype="uint32")
        if bbox is None:
            bbox = np.array([[0, 1], [0, 1], [0, 1]], dtype="float64")

        assert np.array_equal(
            domain_dimensions % chunksize, np.array([0, 0, 0])
        ), "The chunksize does not evenly divide the domain."

        # Managing file existence logic.
        if (filename.exists()) and (not overwrite):
            raise IOError(
                f"Could not create YTHDF5 object at {filename} because it already exists."
            )
        elif filename.exists():
            mylog.info(f"{filename} exists. Overwriting it...")
            os.remove(filename)
        else:
            pass

        # Creating the HDF5 file structure
        cls._construct_hdf5_schema(filename, domain_dimensions, bbox, chunksize)
        return cls.load(filename)

    @contextmanager
    def open(self):
        """
        Context manager for opening the HDF5 buffer for direct read write options.
        """
        fo = h5py.File(self.filename, "a")
        yield fo
        fo.close()

    @staticmethod
    def _construct_hdf5_schema(filename, domain_dimensions, bbox, chunksize):
        # loading the hdf5 file and constructing groups.
        try:
            _buffer = h5py.File(filename, "a")
        except FileNotFoundError:
            raise IOError(
                f"The parent directories of {filename} don't exist. YTHDF5 file was not generated."
            )

        # Managing global attributes
        _buffer.attrs["domain_dimensions"] = domain_dimensions
        _buffer.attrs["bbox"] = bbox
        _buffer.attrs["chunksize"] = chunksize
        _buffer.attrs["model_count"] = 0

        _grid = _buffer.create_group("grid")  # stores the grid information
        _ = _buffer.create_group("chunks")  # group for storing chunk information.

        # constructing the chunkmap
        chunkmap = construct_chunks(domain_dimensions, chunksize)
        _buffer["chunks"].create_dataset("chunkmap", chunkmap.shape, dtype="uint32")
        _buffer["chunks"]["chunkmap"][:] = chunkmap

        # constructing fields
        for field in fields:
            _grid.create_dataset(
                field,
                (chunkmap.shape[-1], chunksize, chunksize, chunksize),
                dtype="float64",
            )
            _grid[field].attrs["unit"] = fields[field]

        _buffer.close()

    @property
    def _estimated_size(self):
        return (np.prod(np.array(self.domain_dimensions))) * 8 * len(fields) / (1e9)

    @property
    def _estimated_chunk_memory(self):
        return (self.chunksize**3) * 8 / (1e9)

    def survey_memory(self):
        """
        Prints a survey of the expected memory and disk usage of the :py:class:`data_structures.YTHDF5` instance.

        If ``psutil`` is installed, additional information is provided regarding the systems capacity to execute the
        chunked operations.
        """
        mylog.info(f"MEMORY SURVEY: {self.filename}")
        mylog.info(f"Total size: {np.round(self._estimated_size,decimals=4)} GB.")
        mylog.info(
            f"Chunk size: {np.round(self._estimated_chunk_memory,decimals=4)} GB."
        )

        try:
            import psutil

            mylog.info(
                f"Free memory: {np.round(psutil.virtual_memory().available/1e9,decimals=3)} GB"
            )

            if psutil.virtual_memory().available / 1e9 < self._estimated_chunk_memory:
                mylog.warning(
                    "Free memory may be insufficient for chunked operations. Processes at this chunksize may fail."
                )
            else:
                pass
        except ImportError:
            pass  # The user doesn't have psutils installed, we don't want to force them to do so.

    def add_model(self, model, center, velocity):
        """
        Add the :py:class:`model.ClusterModel` instance to the :py:class:`data_structures.YTHDF5` instance.

        Parameters
        ----------
        model: :py:class:`model.ClusterModel`
            The model to add to the :py:class:`data_structures.YTHDF5` instance.
        center: array-like
            The center of the cluster in the coordinates of :py:attr:`data_structures.YTHDF5.bbox`.
        velocity: array-like
            The COM velocity of the cluster in the coordinates of :py:attr:`data_structures.YTHDF5.bbox`.
        """
        center, velocity = ensure_ytarray(center, "kpc"), ensure_ytarray(
            velocity, "kpc/Myr"
        )
        _relative_bbox = self.bbox - center.d.reshape((3, 1))

        mylog.info(f"Adding {model} to {self.__str__()}")
        mylog.info(
            f"\tPos: {[np.round(j,decimals=2) for j in center.d]} kpc, Vel: {[np.round(j,decimals=2) for j in velocity.to_value('km/s')]} km/s"
        )

        # setup the coordinates to maximize ease of interpolation.
        _rr = model["radius"].to_value("kpc")
        _dens = model["density"].to_value(fields["density"])

        with self.open() as ythdf5_io:
            for field, unit in tqdm(
                fields.items(),
                desc=f"Writing {model} to YTHDF5",
                leave=False,
                disable=(not cgparams["system"]["display"]["progress_bars"]),
            ):
                if "momentum_density" in field:
                    # manage the momentum density.
                    _momentum_index = {"x": 0, "y": 1, "z": 2}[field[-1]]
                    _yy = _dens * velocity[_momentum_index].to_value(unit)
                elif field in model.fields:
                    _yy = model[field].to_value(unit)
                else:
                    mylog.warning(
                        f"Failed to write model data for {field}; the field doesn't exist in {model}."
                    )
                    continue

                self._add_field(
                    ythdf5_io, _rr, _yy, field, _relative_bbox, self.chunkmap
                )

            mylog.info(f"Core fields of {model} where written to {self}.")

            ythdf5_io.attrs["model_count"] += 1

    def add_ICs(self, ics):
        """
        Add an entire :py:class:`ics.ClusterICs` instance to the :py:class:`data_structures.YTHDF5` buffer.

        Parameters
        ----------
        ics: :py:class:`ics.ClusterICs`
            The initial conditions to add to the HDF5 buffer.
        """
        mylog.info(f"Adding {ics.basename} to {self}.")

        for ic_id, ic_model in enumerate(
            tqdm(
                ics.profiles,
                desc=f"Writing {ics} to YTHDF5",
                position=0,
                leave=False,
                disable=(not cgparams["system"]["display"]["progress_bars"]),
            )
        ):
            model = ClusterModel.from_h5_file(ic_model)
            center, velocity = ics.center[ic_id], ics.velocity[ic_id]

            self.add_model(model, center, velocity)

    def _add_field(self, fileio, r, y, fieldname, bbox, chunkmap):
        # Interpolate and pass to the Cython level.
        _, _, _, _, _, k, _, n, t, c, _, _, _, _ = dfitpack.fpcurf0(
            r, y, 3, w=None, xb=r[0], xe=r[-1], s=0.0
        )
        _buffer_obj = fileio["grid"][fieldname]
        dump_field_to_hdf5(
            _buffer_obj, bbox, self.domain_dimensions, chunkmap, t, c, k, fieldname
        )


if __name__ == "__main__":
    with YTHDF5.temporary() as temp:
        print(temp)
