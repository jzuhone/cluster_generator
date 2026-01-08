"""
IO Backend for integrating :py:class:`cluster_generator.model.ClusterModel`
and :py:class:`cluster_generator.ics.ClusterICs` instances with external
packages like :py:mod:`yt`.
"""

from collections.abc import Collection
from contextlib import contextmanager
from numbers import Number
from typing import Any, Generic, TypeVar

import h5py
import numpy as np
import pathlib as pt
import unyt
from scipy.interpolate import dfitpack
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from cluster_generator.ics import ClusterICs
from cluster_generator.model import ClusterModel
from cluster_generator.opt.structures import construct_chunks, dump_field_to_hdf5
from cluster_generator.utils import Self, cgparams, ensure_ytarray, mylog

Instance = TypeVar("Instance")
Value = TypeVar("Value")
Attribute = TypeVar("Attribute")


class _YTHDF5_Attribute(Generic[Instance, Attribute, Value]):
    """Private attribute descriptor class for YTHDF5 structures."""

    def __set_name__(self, owner, name):
        self.public_name = name

    def __get__(self, instance: Instance | None, owner: type[Instance]) -> Value | Self:
        if (self.public_name in instance._attribute_dictionary) and (
            instance._attribute_dictionary[self.public_name]
        ) is not None:
            # the attribute is already loaded, we simple return it.
            return instance._attribute_dictionary[self.public_name]
        else:
            # The attribute is not already loaded, we should try to get it from disk.
            return self._get_from_file(instance)

    def _get_from_file(self, instance: Instance | None) -> Value | Self:
        with h5py.File(instance.filename, "r") as fo:
            if self.public_name in fo.attrs.keys():
                return fo.attrs[self.public_name]
            else:
                raise ValueError(
                    f"Attribute {self.public_name} is not present in attributes of {instance.filename}."
                )

    def __set__(self, instance: Instance | None, value: Any):
        with h5py.File(instance.filename, "a") as fo:
            fo.attrs[self.public_name] = value

        instance._attribute_dictionary[self.public_name] = value


class YTHDF5:
    """Wrapper class for YT style HDF5 files.

    Used to manage the writing of :py:class:`model.ClusterModel` instances
    to YT datasets.
    """

    _yt_fields: dict = {
        "density": "Msun/kpc**3",
        "dark_matter_density": "Msun/kpc**3",
        "stellar_density": "Msun/kpc**3",
        "pressure": "Msun/(kpc*Myr**2)",
        "momentum_density_x": "Msun/(Myr*kpc**2)",
        "momentum_density_y": "Msun/(Myr*kpc**2)",
        "momentum_density_z": "Msun/(Myr*kpc**2)",
        "magnetic_pressure": "Msun/(kpc*Myr**2)",
    }

    domain_dimensions: Collection[int] = _YTHDF5_Attribute()
    """array-like of int: The grid domain sizes along each of the axes.

    Should be a ``(3,)`` array of integer types specifying the number of cells
    contained along each of the axes.
    """
    bbox: Collection[float] = _YTHDF5_Attribute()
    """array-like of float: The bounding box of the represented domain.

    The ``bbox`` is a ``(3,2)`` array where the first index corresponds to each
    of the axes and the second to the min and max
    coordinate values respectively. The units are assumed to be kpc.
    """
    model_count: int = _YTHDF5_Attribute()
    """int: The number of models which are currently incorporated in this hdf5 dataset.
    """
    chunksize: int = _YTHDF5_Attribute()
    """int: The maximum size (along a single axis) of the computation chunks."""

    def __init__(self, filename: str | pt.Path) -> None:
        """Initialize a :py:class:`data_structures.YTHDF5` instance from an underlying
        HDF5 file.

        Parameters
        ----------
        filename : str
            The HDF5 file corresponding to the intended data structure.
        """
        # Manage the filename and assure that the file does exist.
        self.filename: pt.Path = pt.Path(filename)
        """:py:class:`pathlib.Path`: The path to the underlying HDF5 data."""
        assert self.filename.exists(), f"The file {self.filename} doesn't appear to exist."

        self._attribute_dictionary: dict = {}
        # The attribute dictionary is the __get__ / __set__ location for attribute descriptors.

        with h5py.File(filename, "a") as fo:
            self.chunkmap: np.ndarray = fo["chunks"]["chunkmap"][:]

            """:py:class:`np.ndarray`: The map of chunks for the underlying data
            structure.

            The chunkmap provides a mapping between a given chunk id and the
            corresponding grid coordinates of its edges.
            """

    def __str__(self) -> str:
        return f"<YTHDF5 File @ {self.filename}>"

    def __repr__(self) -> str:
        return self.__str__()

    @classmethod
    def load(cls, filename: str | pt.Path) -> Self:
        """Load an existing :py:class:`data_structures.YTHDF5` instance from file.

        Parameters
        ----------
        filename : str or :py:class:`pathlib.Path`
            The path to the file.

        Returns
        -------
        :py:class:`data_structures.YTHDF5`
        """
        return cls(filename)

    @classmethod
    def build(
        cls,
        filename: str | pt.Path,
        domain_dimensions: Collection[int] = (512, 512, 512),
        bbox: Collection[float] = None,
        overwrite: bool = False,
        chunksize: int = 64,
    ) -> Self:
        """Create a new :py:class:`data_structures.YTHDF5` instance.

        Parameters
        ----------
        filename : str
            The path at which to generate the HDF5 file and the
            corresponding data structure.
        domain_dimensions : array-like, optional
            The dimensions of the grid along each axis (3-tuple).
            By default, this is ``(512,512,512)``.
        bbox : array-like, optional
            The bounding box of the grid. Should be of size ``(3,2)``.
            By default, the bounding box will be ``[0,1]`` along
            each of the axes.
        overwrite : bool, optional
            If ``True``, then the file will be overwritten if it
            already exists. Default ``False``.
        chunksize : int, optional
            The maximum size of a chunk. Chunking is used to conserve
            memory during computations. A higher ``chunksize`` will
            increase the memory usage but decrease computation time,
            a lower value will do the opposite. Default is ``64``.

        Returns
        -------
        :py:class:`data_structures.YTHDF5`
            The generated YT dataset.
        """
        filename = pt.Path(filename)
        domain_dimensions = np.array(domain_dimensions, dtype="uint32")

        if bbox is None:
            bbox = np.array([[0, 1], [0, 1], [0, 1]], dtype="float64")

        # -- Sanity checks -- #
        assert np.array_equal(domain_dimensions % chunksize, np.array([0, 0, 0])), (
            "The chunksize does not evenly divide the domain. Please alter you chunksize so that it fits."
        )

        # Managing file existence logic.
        if filename.exists() and not overwrite:
            raise OSError(f"Could not create YTHDF5 object at {filename} because it already exists.")
        elif filename.exists():
            mylog.info("%s exists. Overwriting it...", filename)
            filename.unlink()
        else:
            pass

        # Creating the HDF5 file structure
        cls._construct_hdf5_schema(filename, domain_dimensions, bbox, chunksize)
        return cls.load(filename)

    @contextmanager
    def open(self, **kwargs):
        """Context manager for opening the HDF5 buffer for direct read write options."""
        fo = h5py.File(self.filename, kwargs.pop("mode", "a"), **kwargs)
        yield fo
        fo.close()

    @classmethod
    def _construct_hdf5_schema(
        cls,
        filename: str | pt.Path,
        domain_dimensions: Collection[int],
        bbox: Collection[float],
        chunksize: int,
    ):
        from cluster_generator.utils import mue

        # loading the hdf5 file and constructing groups.
        try:
            _buffer = h5py.File(filename, "a")
        except FileNotFoundError as e:
            raise OSError(
                f"The parent directories of {filename} don't exist. YTHDF5 file was not generated."
            ) from e

        # Managing global attributes
        _buffer.attrs["domain_dimensions"] = domain_dimensions
        _buffer.attrs["bbox"] = bbox
        _buffer.attrs["chunksize"] = chunksize
        _buffer.attrs["model_count"] = 0
        _buffer.attrs["mu"] = mue

        _grid = _buffer.create_group("grid")  # stores the grid information
        _ = _buffer.create_group("chunks")  # group for storing chunk information.

        # constructing the chunkmap
        chunkmap = construct_chunks(domain_dimensions, chunksize)
        _buffer["chunks"].create_dataset("chunkmap", chunkmap.shape, dtype="uint32")
        _buffer["chunks"]["chunkmap"][:] = chunkmap

        # constructing fields
        for field, unit in cls._yt_fields.items():
            _grid.create_dataset(
                field,
                (chunkmap.shape[-1], chunksize, chunksize, chunksize),
                dtype="float64",
            )
            _grid[field].attrs["unit"] = unit

        _buffer.close()

    @property
    def _estimated_size(self) -> float:
        return np.prod(np.array(self.domain_dimensions)) * 8 * len(self.__class__._yt_fields) / 1e9

    @property
    def _estimated_chunk_memory(self) -> Number:
        return (self.chunksize**3) * 8 / (1e9)

    def survey_memory(self):
        """Prints a survey of the expected memory and disk usage of the
        :py:class:`data_structures.YTHDF5` instance.

        If ``psutil`` is installed, additional information is provided
        regarding the systems capacity to execute the
        chunked operations.
        """
        mylog.info("MEMORY SURVEY: %s.", self.filename)
        mylog.info("Total size: %s GB.", str(np.round(self._estimated_size, decimals=4)))
        mylog.info(
            "Chunk size: %s GB.",
            str(np.round(self._estimated_chunk_memory, decimals=4)),
        )

        try:
            import psutil

            mylog.info(f"Free memory: {np.round(psutil.virtual_memory().available / 1e9, decimals=3)} GB")

            if psutil.virtual_memory().available / 1e9 < self._estimated_chunk_memory:
                mylog.warning(
                    "Free memory may be insufficient for chunked operations. "
                    "Processes at this chunksize may fail."
                )
            else:
                pass
        except ImportError:
            pass  # The user doesn't have psutils installed, we don't want to force them to do so.

    def add_model(
        self,
        model: ClusterModel,
        center: unyt.unyt_array | np.ndarray,
        velocity: unyt.unyt_array | np.ndarray,
    ):
        """Add a new :py:class:`model.ClusterModel`

        Parameters
        ----------
        model : :py:class:`model.ClusterModel`
            The model to add to the :py:class:`data_structures.YTHDF5` instance.
        center : array-like
            The center of the cluster in the coordinates of
            :py:attr:`data_structures.YTHDF5.bbox`.
        velocity : array-like
            The COM velocity of the cluster in the coordinates
            of :py:attr:`data_structures.YTHDF5.bbox`.
        """
        # Enforce unit conventions on method arguments.
        center, velocity = ensure_ytarray(center, "kpc"), ensure_ytarray(velocity, "kpc/Myr")
        _relative_bbox = self.bbox - center.d.reshape((3, 1))

        mylog.info("Adding %s to %s", model, self)
        mylog.info(
            "\tPos: %s kpc, Vel: %s km/s",
            [np.round(j, decimals=2) for j in center.d],
            [np.round(j, decimals=2) for j in velocity.to_value("km/s")],
        )

        # Pull out critical profiles as arrays to ease interpolation logic later on.
        _rr = model["radius"].to_value("kpc")

        with self.open(mode="a") as ythdf5_io, logging_redirect_tqdm(loggers=[mylog]):
            for field, unit in tqdm(
                self._yt_fields.items(),
                desc=f"Writing {model} to YTHDF5",
                leave=False,
                disable=(~cgparams["system"]["display"]["progress_bars"]),
            ):
                if "momentum_density" in field:
                    # The field is a momentum density field. The axis needs
                    # to be determined before we can proceed with the computations.
                    _momentum_index = {"x": 0, "y": 1, "z": 2}[field[-1]]
                    _yy = (model["density"] * velocity[_momentum_index]).to_value(unit)
                    # !
                    # NOTE: because we are working in a grid-context, particle
                    # velocities are not pertinent (unlike SPH).
                    #        Thus, the momentum density should be zero for stationary cells
                    #        (because the system is equilibrated).
                elif field in model.fields:
                    _yy = model[field].to_value(unit)
                else:
                    mylog.debug(
                        f"Failed to write model data for {field}; the field doesn't exist in {model}."
                    )
                    continue

                self._add_field(ythdf5_io, _rr, _yy, field, _relative_bbox, self.chunkmap)

            mylog.info("Core fields of %s where written to %s.", model, self)

            ythdf5_io.attrs["model_count"] += 1

    def add_ICs(self, ics: ClusterICs):
        """Add an entire :py:class:`ics.ClusterICs` instance to the
        :py:class:`data_structures.YTHDF5` buffer.

        Parameters
        ----------
        ics : :py:class:`ics.ClusterICs`
            The initial conditions to add to the HDF5 buffer.
        """
        mylog.info("Adding %s to %s.", ics.basename, self)

        for ic_id, ic_model in enumerate(
            tqdm(
                ics.profiles,
                desc=f"Writing {ics} to YTHDF5",
                position=0,
                leave=False,
                disable=(~cgparams["system"]["display"]["progress_bars"]),
            )
        ):
            model = ClusterModel.from_h5_file(ic_model)
            center, velocity = ics.center[ic_id], ics.velocity[ic_id]

            self.add_model(model, center, velocity)

    def _add_field(self, fileio, r, y, fieldname, bbox, chunkmap):
        # Construct the interpolation parameters from FITPACK.
        _, _, _, _, _, k, _, n, t, c, _, _, _, _ = dfitpack.fpcurf0(r, y, 3, w=None, xb=r[0], xe=r[-1], s=0.0)
        # As per scipy release 1.14, this should be deprecated as of scipy 2.0.
        # It is still present in core scipy code but will need to be replaced
        # to allow compatibility with scipy 2.0.
        _buffer_obj = fileio["grid"][fieldname]

        # Dump the field to HDF5 -> cython.
        dump_field_to_hdf5(_buffer_obj, bbox, self.domain_dimensions, chunkmap, t, c, k, fieldname)
