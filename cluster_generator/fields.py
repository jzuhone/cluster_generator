"""
3D fields for magnetic field initiation and other field based tasks.
"""

import os

import h5py
import numpy as np
from field_kit import FourierAnalysis, GaussianRandomField
from unyt import unyt_array

from cluster_generator.model import ClusterModel
from cluster_generator.utils import mylog


def parse_value(value, default_units):
    """
    Parses an array of values into the correct units.
    Parameters
    ----------
    value : array-like or tuple
        The array from which to convert values to correct units.
        If ``value`` is a ``unyt_array``, the unit is simply converted,
        if ``value`` is a tuple in the form ``(v_array,v_unit)``,
        the conversion will be made and will return an ``unyt_array``.
        Finally, if ``value`` is an array, it is assumed that the
        ``default_units`` are correct.
    default_units : str
        The default unit for the quantity.
    Returns
    -------
    unyt_array:
        The converted array.
    """
    if isinstance(value, unyt_array):
        val = unyt_array(value.v, value.units).in_units(default_units)
    elif isinstance(value, tuple):
        val = unyt_array(value[0], value[1]).in_units(default_units)
    else:
        val = unyt_array(value, default_units)
    return val


class ClusterField:
    """
    Parameters
    ----------
    left_edge : array-like
        The lower edge of the box [kpc] for each of the dimensions.
    right_edge : array-like
        The upper edge of the box [kpc] for each of the dimensions.
    ddims : array-like
        The number of grids in each of the axes.
    padding :
        The amount of additional padding to add to the boundary.
    """

    _units = "dimensionless"
    _name = "vector"
    _profile_field = "field"
    _vector_potential = False
    _divergence_clean = False

    def __init__(
        self,
        left_edge,
        right_edge,
        ddims,
        padding=0.1,
    ):
        ddims = np.array(ddims).astype("int")
        left_edge = parse_value(left_edge, "kpc").v
        right_edge = parse_value(right_edge, "kpc").v
        width = right_edge - left_edge
        deltas = width / ddims
        pad_dims = (2 * np.ceil(0.5 * padding * ddims)).astype("int")
        self.left_edge = left_edge - 0.5 * pad_dims * deltas
        self.right_edge = right_edge + 0.5 * pad_dims * deltas
        self.width = self.right_edge - self.left_edge
        self.ddims = ddims + pad_dims
        self.deltas = deltas
        self.comps = [f"{self._name}_{ax}" for ax in "xyz"]
        self.dx, self.dy, self.dz = self.deltas
        le = self.left_edge + self.deltas * 0.5
        re = self.right_edge - self.deltas * 0.5
        nx, ny, nz = self.ddims
        self.x = np.linspace(le[0], re[0], nx)
        self.y = np.linspace(le[1], re[1], ny)
        self.z = np.linspace(le[2], re[2], nz)
        kx = np.arange(nx, dtype="float64")
        ky = np.arange(ny, dtype="float64")
        kz = np.arange(nz, dtype="float64")
        kx[kx > nx // 2] = kx[kx > nx // 2] - nx
        ky[ky > ny // 2] = ky[ky > ny // 2] - ny
        kz[kz > nz // 2] = kz[kz > nz // 2] - nz
        kx /= nx * self.dx
        ky /= ny * self.dy
        kz /= nz * self.dz
        self.kx = kx
        self.ky = ky
        self.kz = kz
        self.g = None

    def _generate_field(self, *args, **kwargs):
        raise NotImplementedError("This method must be implemented in a subclass.")

    def generate_field(self):
        mylog.info("Starting field generation.")

        self._generate_field()

        mylog.info("Field generation complete.")

    def __getitem__(self, item):
        if item in "xyz":
            return unyt_array(getattr(self, item), "kpc")
        elif item in self.comps:
            comp = "xyz".index(item[-1])
            return unyt_array(self.g[comp, ...], self.units)
        else:
            raise KeyError

    @property
    def units(self):
        if self._vector_potential:
            return f"{self._units}*kpc"
        else:
            return self._units

    def write_file(
        self,
        filename,
        overwrite=False,
        length_unit=None,
        field_unit=None,
        format="hdf5",
    ):
        r"""
        Write the 3D field to a file. The coordinates of
        the cells along the different axes are also written.

        Parameters
        ----------
        filename : string
            The name of the file to write the fields to.
        overwrite : boolean, optional
            Overwrite an existing file with the same name. Default False.
        length_unit : string, optional
            The length unit (affects coordinates and potential fields).
            Default: "kpc"
        field_unit : string, optional
            The units for the field
        """
        from scipy.io import FortranFile

        if length_unit is None:
            length_unit = "kpc"
        if os.path.exists(filename) and not overwrite:
            raise OSError(f"Cannot create {filename}. It exists and overwrite=False.")
        all_comps = ["x", "y", "z"] + self.comps
        if format == "hdf5":
            write_class = h5py.File
        elif format == "fortran":
            write_class = FortranFile
        with write_class(filename, "w") as f:
            if format == "fortran":
                f.write_record(self["x"].size)
            for field in all_comps:
                if field in "xyz":
                    fd = self[field].to(length_unit)
                elif field_unit is not None:
                    if self._vector_potential:
                        units = f"{length_unit}*{field_unit}"
                    else:
                        units = field_unit
                    fd = self[field].to(units)
                else:
                    fd = self[field]
                if format == "hdf5":
                    d = f.create_dataset(field, data=fd.d)
                    d.attrs["units"] = str(fd.units)
                elif format == "fortran":
                    f.write_record(fd.d)
            if format == "hdf5":
                f.attrs["name"] = self._name
                f.attrs["units"] = self.units
                f.attrs["vector_potential"] = int(self._vector_potential)
                f.attrs["divergence_clean"] = int(self._divergence_clean)

    def map_field_to_particles(self, cluster_particles, ptype="gas", units=None):
        r"""
        Map the 3D field to a set of particles, creating new
        particle fields. This uses tri-linear interpolation.

        Parameters
        ----------
        cluster_particles : :class:`~cluster_generator.particles.ClusterParticles`
            The ClusterParticles object which will have new
            fields added.
        ptype : string, optional
            The particle type to add the new fields to. Default:
            "gas", which will almost always be the case.
        units : string, optional
            Change the units of the field. Default: None, which
            implies they will remain in "galactic" units.
        """
        from scipy.interpolate import RegularGridInterpolator

        v = np.zeros((cluster_particles.num_particles[ptype], 3))
        for i, ax in enumerate("xyz"):
            func = RegularGridInterpolator(
                (self["x"], self["y"], self["z"]),
                self[self._name + "_" + ax],
                bounds_error=False,
                fill_value=0.0,
            )
            v[:, i] = func(cluster_particles[ptype, "particle_position"].d)
        cluster_particles.set_field(ptype, self._name, unyt_array(v, self.units), units=units)


class RandomClusterField(ClusterField):
    def __init__(
        self,
        left_edge,
        right_edge,
        ddims,
        power_spec,
        ctr1,
        profile1,
        padding=0.1,
        ctr2=None,
        profile2=None,
        ctr3=None,
        profile3=None,
        r_max=None,
        prng=None,
    ):
        super().__init__(
            left_edge=left_edge,
            right_edge=right_edge,
            ddims=ddims,
            padding=padding,
        )

        self.grf = GaussianRandomField(self.left_edge, self.right_edge, self.ddims, power_spec, seed=prng)
        self.x = self.grf.x
        self.y = self.grf.y
        self.z = self.grf.z

        num_halos = 1
        self.ctr1 = parse_value(ctr1, "kpc").v
        if isinstance(profile1, ClusterModel):
            r1 = profile1["radius"].to_value("kpc")
            g1 = profile1[self._profile_field]
        elif isinstance(profile1, str):
            r1 = unyt_array.from_hdf5(profile1, dataset_name="radius", group_name="fields").to("kpc").d
            g1 = unyt_array.from_hdf5(profile1, dataset_name=self._profile_field, group_name="fields")
        else:
            r1, g1 = profile1
        self.r1 = parse_value(r1, "kpc").v
        self.g1 = parse_value(g1, self._units)
        if profile2 is not None:
            if isinstance(profile2, ClusterModel):
                r2 = profile2["radius"].to_value("kpc")
                g2 = profile2[self._profile_field]
            elif isinstance(profile2, str):
                r2 = unyt_array.from_hdf5(profile2, dataset_name="radius", group_name="fields").to("kpc").d
                g2 = unyt_array.from_hdf5(
                    profile2,
                    dataset_name=self._profile_field,
                    group_name="fields",
                )
            else:
                r2, g2 = profile2
            num_halos += 1
            if ctr2 is None:
                raise RuntimeError("Need to specify 'ctr2' for the second halo!")
            self.ctr2 = parse_value(ctr2, "kpc").v
            self.r2 = parse_value(r2, "kpc").v
            self.g2 = parse_value(g2, self._units)
        if profile3 is not None:
            if isinstance(profile3, ClusterModel):
                r3 = profile3["radius"].to_value("kpc")
                g3 = profile3[self._profile_field]
            elif isinstance(profile3, str):
                r3 = unyt_array.from_hdf5(profile3, dataset_name="radius", group_name="fields").to("kpc").d
                g3 = unyt_array.from_hdf5(
                    profile3,
                    dataset_name=self._profile_field,
                    group_name="fields",
                )
            else:
                r3, g3 = profile3
            num_halos += 1
            if ctr3 is None:
                raise RuntimeError("Need to specify 'ctr3' for the second halo!")
            self.ctr3 = parse_value(ctr3, "kpc").v
            self.r3 = parse_value(r3, "kpc").v
            self.g3 = parse_value(g3, self._units)
        self.r_max = r_max
        self.num_halos = num_halos

    def _generate_field(self):
        g = self.grf.generate_vector_field_realization()
        g_avg = np.sqrt(np.mean(g * g))

        mylog.info("Scaling the fields by cluster 1.")
        rr = np.sqrt(
            (self.x[:, np.newaxis, np.newaxis] - self.ctr1[0]) ** 2
            + (self.y[np.newaxis, :, np.newaxis] - self.ctr1[1]) ** 2
            + (self.z[np.newaxis, np.newaxis, :] - self.ctr1[2]) ** 2
        )
        if self.r_max is not None:
            rr[rr > self.r_max] = self.r_max
        idxs = np.searchsorted(self.r1, rr) - 1
        dr = (rr - self.r1[idxs]) / (self.r1[idxs + 1] - self.r1[idxs])
        g_rms = ((1.0 - dr) * self.g1[idxs] + dr * self.g1[idxs + 1]) ** 2
        if self.num_halos >= 2:
            mylog.info("Scaling the fields by cluster 2.")
            rr = np.sqrt(
                (self.x[:, np.newaxis, np.newaxis] - self.ctr2[0]) ** 2
                + (self.y[np.newaxis, :, np.newaxis] - self.ctr2[1]) ** 2
                + (self.z[np.newaxis, np.newaxis, :] - self.ctr2[2]) ** 2
            )
            if self.r_max is not None:
                rr[rr > self.r_max] = self.r_max
            idxs = np.searchsorted(self.r2, rr) - 1
            dr = (rr - self.r2[idxs]) / (self.r2[idxs + 1] - self.r2[idxs])
            g_rms += ((1.0 - dr) * self.g2[idxs] + dr * self.g2[idxs + 1]) ** 2
        if self.num_halos == 3:
            mylog.info("Scaling the fields by cluster 3.")
            rr = np.sqrt(
                (self.x[:, np.newaxis, np.newaxis] - self.ctr3[0]) ** 2
                + (self.y[np.newaxis, :, np.newaxis] - self.ctr3[1]) ** 2
                + (self.z[np.newaxis, np.newaxis, :] - self.ctr3[2]) ** 2
            )
            if self.r_max is not None:
                rr[rr > self.r_max] = self.r_max
            idxs = np.searchsorted(self.r3, rr) - 1
            dr = (rr - self.r3[idxs]) / (self.r3[idxs + 1] - self.r3[idxs])
            g_rms += ((1.0 - dr) * self.g3[idxs] + dr * self.g3[idxs + 1]) ** 2
            g_rms = np.sqrt(g_rms).in_units(self._units).d / g_avg

        g *= g_rms

        if self._divergence_clean:
            fa = FourierAnalysis(self.width, self.ddims)
            g = fa.divergence_component(g)
            # for an isotropic field, the divergence component is 1/3 of the
            # total
            g *= 3.0
            if self._vector_potential:
                g = fa.potential_of_field(
                    g,
                )
        self.g = g


class MagneticRandomClusterField(RandomClusterField):
    _units = "gauss"
    _profile_field = "magnetic_field_strength"
    _name = "magnetic_field"
    _vector_potential = False
    _divergence_clean = True

    def __init__(
        self,
        left_edge,
        right_edge,
        ddims,
        power_spec,
        ctr1,
        profile1,
        padding=0.1,
        ctr2=None,
        profile2=None,
        ctr3=None,
        profile3=None,
        r_max=None,
        prng=None,
    ):
        super().__init__(
            left_edge,
            right_edge,
            ddims,
            power_spec,
            ctr1,
            profile1,
            padding=padding,
            ctr2=ctr2,
            profile2=profile2,
            ctr3=ctr3,
            profile3=profile3,
            r_max=r_max,
            prng=prng,
        )


class MagneticPotentialRandomClusterField(MagneticRandomClusterField):
    _name = "magnetic_vector_potential"
    _vector_potential = True


class VelocityRandomClusterField(RandomClusterField):
    _units = "kpc/Myr"
    _name = "velocity"

    def __init__(
        self,
        left_edge,
        right_edge,
        ddims,
        power_spec,
        ctr1,
        profile1,
        padding=0.1,
        ctr2=None,
        profile2=None,
        ctr3=None,
        profile3=None,
        r_max=None,
        prng=None,
        divergence_clean=False,
    ):
        self._divergence_clean = divergence_clean
        super().__init__(
            left_edge,
            right_edge,
            ddims,
            power_spec,
            ctr1,
            profile1,
            padding=padding,
            ctr2=ctr2,
            profile2=profile2,
            ctr3=ctr3,
            profile3=profile3,
            r_max=r_max,
            prng=prng,
        )
