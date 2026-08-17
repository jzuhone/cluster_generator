"""
Builds a real AMReX plotfile directly from
:py:class:`cluster_generator.model.ClusterModel` /
:py:class:`cluster_generator.ics.ClusterICs` profiles, for use with
:py:mod:`yt` (via :py:mod:`cluster_generator.frontend`) or any other
AMReX-plotfile reader. Optionally locally-refined around each cluster's
core (see ``regions`` on :func:`AMRClusterDataset.build`).
"""

import numpy as np
import pathlib as pt
from scipy.interpolate import InterpolatedUnivariateSpline

from cluster_generator.amr_hierarchy import AMRHierarchy
from cluster_generator.ics import ClusterICs
from cluster_generator.model import ClusterModel
from cluster_generator.utils import ensure_ytarray, mu, mylog


class AMRClusterDataset:
    """
    Builds a (optionally locally-refined) AMReX plotfile directly from one
    or more :py:class:`~cluster_generator.model.ClusterModel` profiles.

    Construct with :meth:`build`, add each cluster with :meth:`add_model`
    or :meth:`add_ICs`, then call :meth:`write` to produce the plotfile.
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

    def __init__(self, hierarchy: AMRHierarchy):
        self.hierarchy = hierarchy
        self.model_count = 0

    def __str__(self) -> str:
        return f"<AMRClusterDataset ({self.model_count} model(s), {len(self.hierarchy.levels)} level(s))>"

    def __repr__(self) -> str:
        return self.__str__()

    @classmethod
    def build(
        cls,
        domain_dimensions=(512, 512, 512),
        bbox=None,
        regions=None,
        max_grid_size=64,
    ) -> "AMRClusterDataset":
        """
        Create a new :py:class:`AMRClusterDataset`.

        Parameters
        ----------
        domain_dimensions : array-like, optional
            The dimensions of the base (level 0) grid along each axis.
            By default, this is ``(512, 512, 512)``.
        bbox : array-like, optional
            The bounding box of the base grid, shape ``(3, 2)`` [kpc].
            By default, ``[0, 1]`` along each axis.
        regions : list of dict, optional
            Locally-refined regions around, e.g., a cluster core -- the
            same schema as
            :class:`cluster_generator.fields.RandomClusterField`'s
            ``refinement_regions`` (see :class:`AMRHierarchy`). Default:
            None (a plain uniform grid).
        max_grid_size : int, optional
            The maximum size of a single AMR box along any axis. A higher
            value increases memory usage per box but reduces the number of
            boxes. Default: 64.

        Returns
        -------
        :py:class:`AMRClusterDataset`
        """
        if bbox is None:
            bbox = np.array([[0, 1], [0, 1], [0, 1]], dtype="float64")
        bbox = np.asarray(bbox, dtype="float64")

        hierarchy = AMRHierarchy(
            left_edge=bbox[:, 0],
            right_edge=bbox[:, 1],
            base_dims=domain_dimensions,
            field_names=list(cls._yt_fields.keys()),
            regions=regions,
            max_grid_size=max_grid_size,
        )
        return cls(hierarchy)

    def add_model(
        self,
        model: ClusterModel,
        center,
        velocity,
    ):
        """Add a new :py:class:`~cluster_generator.model.ClusterModel`.

        Parameters
        ----------
        model : :py:class:`~cluster_generator.model.ClusterModel`
            The model to add.
        center : array-like
            The center of the cluster [kpc], in the same coordinates as
            the bounding box passed to :meth:`build`.
        velocity : array-like
            The COM velocity of the cluster [kpc/Myr].
        """
        center = ensure_ytarray(center, "kpc").d
        velocity = ensure_ytarray(velocity, "kpc/Myr")

        mylog.info("Adding %s to %s", model, self)
        mylog.info(
            "\tPos: %s kpc, Vel: %s km/s",
            [np.round(c, decimals=2) for c in center],
            [np.round(v, decimals=2) for v in velocity.to_value("km/s")],
        )

        r = model["radius"].to_value("kpc")

        for field, unit in self._yt_fields.items():
            if "momentum_density" in field:
                axis = {"x": 0, "y": 1, "z": 2}[field[-1]]
                y = (model["density"] * velocity[axis]).to_value(unit)
                # Because we are working in a grid-context, particle
                # velocities are not pertinent (unlike SPH). Thus, the
                # momentum density should be zero for stationary cells
                # (because the system is equilibrated).
            elif field in model.fields:
                y = model[field].to_value(unit)
            else:
                mylog.debug(f"Failed to write model data for {field}; the field doesn't exist in {model}.")
                continue

            spline = InterpolatedUnivariateSpline(r, y, k=3, ext="const")

            def func(x, y_, z, _center=center, _spline=spline):
                dx = x[:, np.newaxis, np.newaxis] - _center[0]
                dy = y_[np.newaxis, :, np.newaxis] - _center[1]
                dz = z[np.newaxis, np.newaxis, :] - _center[2]
                rr = np.sqrt(dx * dx + dy * dy + dz * dz)
                return _spline(rr.ravel()).reshape(rr.shape)

            self.hierarchy.fill_field(field, func, accumulate=True)

        self.model_count += 1

    def add_ICs(self, ics: ClusterICs):
        """Add an entire :py:class:`~cluster_generator.ics.ClusterICs` instance.

        Parameters
        ----------
        ics : :py:class:`~cluster_generator.ics.ClusterICs`
            The initial conditions to add.
        """
        mylog.info("Adding %s to %s.", ics.basename, self)
        for ic_id, ic_model in enumerate(ics.profiles):
            model = ClusterModel.from_h5_file(ic_model)
            self.add_model(model, ics.center[ic_id], ics.velocity[ic_id])

    def write(self, dirname, overwrite: bool = False) -> str:
        """
        Write the plotfile to disk.

        Parameters
        ----------
        dirname : str or :py:class:`pathlib.Path`
            The plotfile directory to create.
        overwrite : bool, optional
            If ``True``, delete ``dirname`` first if it already exists.
            Default: False.

        Returns
        -------
        str
            ``dirname``, for chaining.
        """
        import shutil

        dirname = pt.Path(dirname)
        if dirname.exists():
            if not overwrite:
                raise OSError(f"Cannot create {dirname}. It exists and overwrite=False.")
            shutil.rmtree(dirname)

        self.hierarchy.write_plotfile(
            dirname,
            job_info={
                "generator": "cluster_generator",
                "mu": mu,
                "model_count": self.model_count,
            },
        )
        return str(dirname)
