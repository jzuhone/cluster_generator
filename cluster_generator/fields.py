"""
3D fields for magnetic field initiation and other field based tasks.
"""

import os
from dataclasses import dataclass

import h5py
import numpy as np
from field_kit import FourierAnalysis, GaussianRandomField
from unyt import unyt_array

from cluster_generator.model import ClusterModel
from cluster_generator.utils import mylog


@dataclass
class FieldPatch:
    """
    A single locally-refined sub-box of a :class:`RandomClusterField`,
    carrying only the small-scale ("high-k") power beyond what the
    coarse grid resolves. See ``refinement_regions`` on
    :class:`RandomClusterField`.
    """

    left_edge: np.ndarray
    right_edge: np.ndarray
    ddims: np.ndarray
    deltas: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    g: np.ndarray
    window: np.ndarray
    frac_low: float


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


def _load_radial_profile(profile, profile_field):
    r"""
    Load a radial profile from a :class:`~cluster_generator.model.ClusterModel`,
    an HDF5 filename, or an ``(r, g)`` array-like pair.

    Parameters
    ----------
    profile : ClusterModel, string, or (r, g) array-like
        The source of the profile.
    profile_field : str
        The name of the field to read (e.g. "magnetic_field_strength").
    units : str
        The units to convert the field values to.

    Returns
    -------
    r : ndarray
        Radii [kpc].
    g : unyt_array
        The field values, in ``units``.
    """
    if isinstance(profile, ClusterModel):
        r = profile["radius"].to_value("kpc")
        g = profile[profile_field]
    elif isinstance(profile, str):
        r = unyt_array.from_hdf5(profile, dataset_name="radius", group_name="fields").to("kpc").d
        g = unyt_array.from_hdf5(profile, dataset_name=profile_field, group_name="fields")
    else:
        r, g = profile
    return r, g


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
        patches = getattr(self, "patches", [])
        if patches and format != "hdf5":
            raise NotImplementedError("Writing refinement patches is only supported for the 'hdf5' format.")
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
                f.attrs["num_patches"] = np.int32(len(patches))
                for i, patch in enumerate(patches):
                    grp = f.create_group(f"patch_{i:02d}")
                    grp.attrs["frac_low"] = patch.frac_low
                    grp.create_dataset("window", data=patch.window)
                    for ax, arr in zip("xyz", (patch.x, patch.y, patch.z), strict=True):
                        pd = unyt_array(arr, "kpc").to(length_unit)
                        d = grp.create_dataset(ax, data=pd.d)
                        d.attrs["units"] = str(pd.units)
                    for j, ax in enumerate("xyz"):
                        pfd = unyt_array(patch.g[j, ...], self.units)
                        if field_unit is not None:
                            pfd = pfd.to(
                                f"{length_unit}*{field_unit}" if self._vector_potential else field_unit
                            )
                        d = grp.create_dataset(f"{self._name}_{ax}", data=pfd.d)
                        d.attrs["units"] = str(pfd.units)

    def _interpolate_at_points(self, coords):
        r"""
        Evaluate the field at arbitrary points via tri-linear
        interpolation on the coarse grid, plus a correction from any
        locally-refined ``patches`` (see ``RandomClusterField``'s
        ``refinement_regions``) whose box contains the point. Patches
        are assumed non-overlapping.
        """
        from scipy.interpolate import RegularGridInterpolator

        coords = np.asarray(coords)
        v = np.zeros((coords.shape[0], 3))
        for i, ax in enumerate("xyz"):
            func = RegularGridInterpolator(
                (self["x"], self["y"], self["z"]),
                self[self._name + "_" + ax],
                bounds_error=False,
                fill_value=0.0,
            )
            v[:, i] = func(coords)

        for patch in getattr(self, "patches", []):
            w_func = RegularGridInterpolator(
                (patch.x, patch.y, patch.z),
                patch.window,
                bounds_error=False,
                fill_value=0.0,
            )
            w = w_func(coords)
            # outside the patch box, w == 0 (fill_value), so this
            # correction is a no-op and the coarse-only value is kept
            correction = 1.0 - (1.0 - np.sqrt(patch.frac_low)) * w
            for i in range(3):
                g_func = RegularGridInterpolator(
                    (patch.x, patch.y, patch.z),
                    patch.g[i, ...],
                    bounds_error=False,
                    fill_value=0.0,
                )
                v[:, i] = v[:, i] * correction + g_func(coords)
        return v

    def interpolate_to_points(self, coords, units=None):
        r"""
        Evaluate the field at arbitrary points, e.g. the cell centers
        of an external (AMR) simulation mesh, via tri-linear
        interpolation. Automatically uses the finer ``patches`` data
        (see ``RandomClusterField``'s ``refinement_regions``) where
        available.

        Parameters
        ----------
        coords : array-like, shape (N, 3)
            The (x, y, z) coordinates [kpc] at which to evaluate the
            field.
        units : string, optional
            Change the units of the returned field. Default: None,
            which implies they will remain in the field's native
            units.

        Returns
        -------
        unyt_array, shape (N, 3)
            The interpolated field values at ``coords``.
        """
        v = self._interpolate_at_points(coords)
        result = unyt_array(v, self.units)
        if units is not None:
            result = result.in_units(units)
        return result

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
        v = self._interpolate_at_points(cluster_particles[ptype, "particle_position"].d)
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
        refinement_regions=None,
    ):
        super().__init__(
            left_edge=left_edge,
            right_edge=right_edge,
            ddims=ddims,
            padding=padding,
        )

        if refinement_regions and not np.allclose(self.deltas, self.deltas[0]):
            raise ValueError(
                "refinement_regions requires an isotropic coarse grid "
                f"(dx == dy == dz); got deltas={self.deltas}."
            )
        self.refinement_regions = refinement_regions or []
        self.patches = []
        self.power_spec = power_spec

        if self.refinement_regions:
            if isinstance(prng, np.random.Generator):
                raise ValueError(
                    "When using refinement_regions, 'prng' must be an int seed or "
                    "None (not an existing Generator instance), so independent "
                    "reproducible per-patch seeds can be derived from it."
                )
            seed_seq = np.random.SeedSequence(prng)
            coarse_seed, *self._patch_seeds = seed_seq.spawn(1 + len(self.refinement_regions))
        else:
            coarse_seed = prng
            self._patch_seeds = []

        self.grf = GaussianRandomField(
            self.left_edge, self.right_edge, self.ddims, power_spec, seed=coarse_seed
        )
        self.x = self.grf.x
        self.y = self.grf.y
        self.z = self.grf.z

        num_halos = 1
        self.ctr1 = parse_value(ctr1, "kpc").v
        r1, g1 = _load_radial_profile(profile1, self._profile_field)
        self.r1 = parse_value(r1, "kpc").v
        self.g1 = parse_value(g1, self._units)

        if profile2 is not None:
            num_halos += 1
            if ctr2 is None:
                raise RuntimeError("Need to specify 'ctr2' for the second halo!")
            self.ctr2 = parse_value(ctr2, "kpc").v
            r2, g2 = _load_radial_profile(profile2, self._profile_field)
            self.r2 = parse_value(r2, "kpc").v
            self.g2 = parse_value(g2, self._units)
        if profile3 is not None:
            num_halos += 1
            if ctr3 is None:
                raise RuntimeError("Need to specify 'ctr3' for the second halo!")
            self.ctr3 = parse_value(ctr3, "kpc").v
            r3, g3 = _load_radial_profile(profile3, self._profile_field)
            self.r3 = parse_value(r3, "kpc").v
            self.g3 = parse_value(g3, self._units)
        self.r_max = r_max
        self.num_halos = num_halos

    def _compute_g_rms(self, x, y, z):
        r"""
        Evaluate the target field-strength profile [self._units] at the
        given grid coordinates [kpc], summing in quadrature over all
        configured halos. This is the physical target rms amplitude;
        callers are responsible for normalizing a raw GRF realization to
        match it (dividing by that realization's own rms).
        """
        mylog.info("Scaling the fields by cluster 1.")
        rr = np.sqrt(
            (x[:, np.newaxis, np.newaxis] - self.ctr1[0]) ** 2
            + (y[np.newaxis, :, np.newaxis] - self.ctr1[1]) ** 2
            + (z[np.newaxis, np.newaxis, :] - self.ctr1[2]) ** 2
        )
        if self.r_max is not None:
            rr[rr > self.r_max] = self.r_max
        idxs = np.searchsorted(self.r1, rr) - 1
        dr = (rr - self.r1[idxs]) / (self.r1[idxs + 1] - self.r1[idxs])
        g_rms = ((1.0 - dr) * self.g1[idxs] + dr * self.g1[idxs + 1]) ** 2
        if self.num_halos >= 2:
            mylog.info("Scaling the fields by cluster 2.")
            rr = np.sqrt(
                (x[:, np.newaxis, np.newaxis] - self.ctr2[0]) ** 2
                + (y[np.newaxis, :, np.newaxis] - self.ctr2[1]) ** 2
                + (z[np.newaxis, np.newaxis, :] - self.ctr2[2]) ** 2
            )
            if self.r_max is not None:
                rr[rr > self.r_max] = self.r_max
            idxs = np.searchsorted(self.r2, rr) - 1
            dr = (rr - self.r2[idxs]) / (self.r2[idxs + 1] - self.r2[idxs])
            g_rms += ((1.0 - dr) * self.g2[idxs] + dr * self.g2[idxs + 1]) ** 2
        if self.num_halos == 3:
            mylog.info("Scaling the fields by cluster 3.")
            rr = np.sqrt(
                (x[:, np.newaxis, np.newaxis] - self.ctr3[0]) ** 2
                + (y[np.newaxis, :, np.newaxis] - self.ctr3[1]) ** 2
                + (z[np.newaxis, np.newaxis, :] - self.ctr3[2]) ** 2
            )
            if self.r_max is not None:
                rr[rr > self.r_max] = self.r_max
            idxs = np.searchsorted(self.r3, rr) - 1
            dr = (rr - self.r3[idxs]) / (self.r3[idxs + 1] - self.r3[idxs])
            g_rms += ((1.0 - dr) * self.g3[idxs] + dr * self.g3[idxs + 1]) ** 2
        return np.sqrt(g_rms).in_units(self._units).d

    def _divergence_clean_field(self, g, width, ddims):
        fa = FourierAnalysis(width, ddims)
        # solenoidal_component() projects out the divergence-free
        # (transverse) part of the field, as required for a physical
        # magnetic field. It doesn't renormalize the amplitude, so:
        # for an isotropic field, the transverse part carries 2/3 of the
        # total power (2 independent transverse directions vs. 1
        # longitudinal), so rescale by sqrt(3/2) to restore the original
        # variance.
        g = fa.solenoidal_component(g) * np.sqrt(1.5)
        if self._vector_potential:
            g = fa.potential_of_field(g)
        return g

    def _generate_field(self):
        g = self.grf.generate_vector_field_realization()
        g_avg = np.sqrt(np.mean(g * g))
        self._g_avg_low = g_avg

        g_rms = self._compute_g_rms(self.x, self.y, self.z) / g_avg
        g *= g_rms

        if self._divergence_clean:
            g = self._divergence_clean_field(g, self.width, self.ddims)
        self.g = g

        if self.refinement_regions:
            self._generate_patches()

    @staticmethod
    def _highpass_filter(field, deltas, k_cut):
        r"""
        Zero out all Fourier modes with :math:`|k| < k_{\rm cut}` in a
        real-space vector field, returning only its small-scale content.
        Implemented with plain ``numpy.fft`` (a self-contained round
        trip), independent of any of ``field_kit``'s internal Fourier
        scaling conventions.
        """
        ndim, nx, ny, nz = field.shape
        kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=deltas[0])
        ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=deltas[1])
        kz = 2.0 * np.pi * np.fft.fftfreq(nz, d=deltas[2])
        kk = np.sqrt(
            kx[:, np.newaxis, np.newaxis] ** 2
            + ky[np.newaxis, :, np.newaxis] ** 2
            + kz[np.newaxis, np.newaxis, :] ** 2
        )
        mask = kk < k_cut
        out = np.empty_like(field)
        for i in range(ndim):
            fhat = np.fft.fftn(field[i])
            fhat[mask] = 0.0
            out[i] = np.fft.ifftn(fhat).real
        return out

    def _generate_patch(self, region, seed):
        r"""
        Generate a single locally-refined :class:`FieldPatch`: an
        independent GRF realization at finer resolution than the coarse
        grid, high-pass filtered to keep only the power the coarse grid
        can't represent, amplitude-normalized so the combined
        coarse+patch field matches the target profile without
        double-counting power (see ``refinement_regions`` docs), and
        tapered to zero at the padded edges of the patch box so it
        blends smoothly into the coarse field.
        """
        center = parse_value(region["center"], "kpc").v
        width = np.atleast_1d(np.array(region["width"], dtype="float64"))
        if width.size == 1:
            width = np.repeat(width, 3)
        refine_by = int(region.get("refine_by", 4))
        padding = region.get("padding", 0.25)
        taper_alpha = region.get("taper_alpha", 0.3)

        coarse_delta = self.deltas[0]
        patch_delta = coarse_delta / refine_by

        pad_width = width * (1.0 + padding)
        patch_ddims = (2 * np.ceil(0.5 * pad_width / patch_delta)).astype("int")
        patch_left = center - 0.5 * patch_ddims * patch_delta
        patch_right = center + 0.5 * patch_ddims * patch_delta
        patch_width = patch_right - patch_left
        patch_deltas = patch_width / patch_ddims

        grf = GaussianRandomField(patch_left, patch_right, patch_ddims, self.power_spec, seed=seed)
        f_raw = grf.generate_vector_field_realization()

        k_cut = np.pi / coarse_delta
        f_high = self._highpass_filter(f_raw, patch_deltas, k_cut)
        g_avg_high = np.sqrt(np.mean(f_high * f_high))
        frac_low = self._g_avg_low**2 / (self._g_avg_low**2 + g_avg_high**2)

        le = patch_left + patch_deltas * 0.5
        re = patch_right - patch_deltas * 0.5
        x = np.linspace(le[0], re[0], patch_ddims[0])
        y = np.linspace(le[1], re[1], patch_ddims[1])
        z = np.linspace(le[2], re[2], patch_ddims[2])

        target_rms = self._compute_g_rms(x, y, z)
        f_high = f_high * (np.sqrt(1.0 - frac_low) / g_avg_high) * target_rms

        if self._divergence_clean:
            f_high = self._divergence_clean_field(f_high, patch_width, patch_ddims)

        fa = FourierAnalysis(patch_width, patch_ddims)
        window = np.ones(tuple(patch_ddims))
        fa.window_data(window, filter_function="tukey", alpha=taper_alpha)
        f_high = f_high * window

        return FieldPatch(
            left_edge=patch_left,
            right_edge=patch_right,
            ddims=patch_ddims,
            deltas=patch_deltas,
            x=x,
            y=y,
            z=z,
            g=f_high,
            window=window,
            frac_low=frac_low,
        )

    def _generate_patches(self):
        mylog.info("Generating %d refinement patch(es).", len(self.refinement_regions))
        self.patches = [
            self._generate_patch(region, seed)
            for region, seed in zip(self.refinement_regions, self._patch_seeds, strict=True)
        ]


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
        refinement_regions=None,
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
            refinement_regions=refinement_regions,
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
        refinement_regions=None,
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
            refinement_regions=refinement_regions,
        )


def refinement_regions_for_clusters(
    centers,
    profiles,
    profile_field,
    width_frac=0.1,
    refine_by=4,
    padding=0.25,
    taper_alpha=0.3,
    r_max=None,
    max_width=None,
):
    r"""
    Build a ``refinement_regions`` list (see :class:`RandomClusterField`) with
    one refinement patch centered on each cluster, sized automatically from
    that cluster's own profile rather than picked by hand.

    The width of each patch is set to twice the radius at which the
    cluster's profile first falls to ``width_frac`` of its central (innermost
    tabulated) value -- i.e. the patch spans out to where the field has
    dropped off, not an arbitrary fixed size. Placement uses ``centers``
    directly, so it stays in sync with whatever you pass as
    ``ctr1``/``ctr2``/``ctr3`` to the field class's constructor.

    A slowly-declining profile sampled out to large radii can produce a
    width far bigger than intended (and, in turn, a patch grid so large it
    exhausts memory) -- pass ``max_width`` (and/or ``r_max``) to guard
    against this, especially the first time you use a given profile.

    Parameters
    ----------
    centers : sequence of array-like
        The cluster centers [kpc], in the same order as ``ctr1``, ``ctr2``,
        ``ctr3`` passed to the field constructor.
    profiles : sequence of ClusterModel, string, or (r, g) array-like
        The corresponding profiles (``profile1``, ``profile2``, ``profile3``),
        one per center.
    profile_field : str
        The name of the field that is being profiled.
    width_frac : float or sequence of float, optional
        The fraction of each profile's central value used to define its
        patch width (see above). A single value applies to every cluster;
        a sequence must match ``centers`` in length. Default: 0.1.
    refine_by : int or sequence of int, optional
        The refinement factor relative to the coarse grid's cell size (see
        ``refinement_regions``). A single value applies to every cluster; a
        sequence must match ``centers`` in length. Default: 4.
    padding : float, optional
        Shared padding fraction for every patch (see ``refinement_regions``).
        Default: 0.25.
    taper_alpha : float, optional
        Shared Tukey taper parameter for every patch (see
        ``refinement_regions``). Default: 0.3.
    r_max : float, optional
        If given, ignore profile values beyond this radius [kpc] when
        determining each patch's width (matching the field constructor's own
        ``r_max``).
    max_width : float, optional
        If given, cap every computed width at this value [kpc] (a hard
        safety limit, applied after ``width_frac``/``r_max``). Default:
        None (no cap).

    Returns
    -------
    list of dict
        Ready to pass as ``refinement_regions`` to ``field_cls`` (or any
        other :class:`RandomClusterField` subclass).
    """
    n = len(centers)
    if len(profiles) != n:
        raise ValueError("centers and profiles must have the same length.")

    def _broadcast(value, name):
        if np.isscalar(value):
            return [value] * n
        value = list(value)
        if len(value) != n:
            raise ValueError(f"'{name}' must be a scalar or have the same length as 'centers'.")
        return value

    width_fracs = _broadcast(width_frac, "width_frac")
    refine_bys = _broadcast(refine_by, "refine_by")

    regions = []
    for center, profile, wf, rb in zip(centers, profiles, width_fracs, refine_bys, strict=True):
        r, g = _load_radial_profile(profile, profile_field)
        r = parse_value(r, "kpc").v
        g = np.abs(g.d if hasattr(g, "d") else np.asarray(g))
        if r_max is not None:
            keep = r <= r_max
            r, g = r[keep], g[keep]
        threshold = wf * g[0]
        below = g <= threshold
        idx = int(np.argmax(below)) if below.any() else len(r) - 1
        width = 2.0 * r[idx]
        if max_width is not None:
            width = min(width, max_width)
        regions.append(
            {
                "center": center,
                "width": width,
                "refine_by": rb,
                "padding": padding,
                "taper_alpha": taper_alpha,
            }
        )
    return regions
