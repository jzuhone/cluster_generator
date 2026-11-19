"""
3D fields for magnetic field initiation and other field based tasks.
"""

import os
from dataclasses import dataclass, field

import h5py
import numpy as np
from kspace import FourierAnalysis, GaussianRandomField
from unyt import unyt_array

from cluster_generator.amr_hierarchy import (
    check_no_overlap,
    load_radial_profile,
    region_chain_geometry,
)
from cluster_generator.utils import mylog


@dataclass
class FieldPatch:
    """
    One node of a :class:`RandomClusterField`'s coarse-to-fine patch tree.

    Every node -- the coarse grid itself (``ClusterField.root``) as well
    as every locally-refined sub-box (see ``refinement_regions`` on
    :class:`RandomClusterField`) -- is represented the same way, so a
    single recursive algorithm (:meth:`ClusterField._apply_patch`) can
    walk the whole tree uniformly: the root has ``frac_low=0`` and a
    ``window`` of all ones (it has no parent to blend into and carries
    the full target amplitude on its own), while every other node carries
    only the small-scale ("high-k") power beyond what its parent (the
    root, or an enclosing, coarser patch) resolves.

    ``children`` holds any further-refined sub-boxes nested inside this
    one (built from that region's own ``"children"`` entry, or --
    for the root -- from the top-level ``refinement_regions`` list),
    enabling an arbitrary number of refinement levels; it is empty for a
    node with no nested children.
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
    g_avg: float
    children: list = field(default_factory=list)


def _blend_patch_field(ancestors, patch):
    r"""
    Blend ``patch``'s own high-k-only field (see :class:`FieldPatch`) with
    the chain of its ancestors -- the root, then every enclosing, coarser
    patch, in order -- evaluated at ``patch``'s own grid points, using the
    same taper-based correction as :meth:`ClusterField._apply_patch`.

    Each ancestor's contribution is interpolated directly from *its own*
    native ``g``/``window`` arrays (never from an already-regridded
    intermediate), exactly mirroring what repeated calls to
    :meth:`ClusterField._apply_patch` would do at these same points -- so
    this reproduces that lazy, per-query-point algorithm exactly, just
    evaluated once over ``patch``'s whole grid, at write time (see
    :func:`_blend_patches`/:meth:`ClusterField.write_file`). What ends up
    on disk is already the final, fully-blended field, so a reader never
    needs ``window``/``frac_low`` (or any other blending metadata) at all
    -- just field data.
    """
    from scipy.interpolate import RegularGridInterpolator

    g1, g2, g3 = np.meshgrid(patch.x, patch.y, patch.z, indexing="ij")
    pts = np.stack([g1.ravel(), g2.ravel(), g3.ravel()], axis=-1)

    v = np.zeros((pts.shape[0], 3))
    for ancestor in ancestors:
        w_func = RegularGridInterpolator(
            (ancestor.x, ancestor.y, ancestor.z), ancestor.window, bounds_error=False, fill_value=0.0
        )
        correction = 1.0 - (1.0 - np.sqrt(ancestor.frac_low)) * w_func(pts)
        for i in range(3):
            g_func = RegularGridInterpolator(
                (ancestor.x, ancestor.y, ancestor.z), ancestor.g[i, ...], bounds_error=False, fill_value=0.0
            )
            v[:, i] = v[:, i] * correction + g_func(pts)

    # patch's own data, evaluated exactly at its own grid nodes -- plain
    # array lookups, no interpolation (and so no error) involved
    correction = 1.0 - (1.0 - np.sqrt(patch.frac_low)) * patch.window
    blended = np.empty_like(patch.g)
    for i in range(3):
        blended[i] = v[:, i].reshape(g1.shape) * correction + patch.g[i]
    return blended


def _blend_patches(patches, ancestors, level=1, flat=None):
    r"""
    Recursively blend a list of :class:`FieldPatch` trees (each node's own
    ``children``, arbitrarily nested; see :func:`_blend_patch_field`) and
    flatten them into a single depth-first list of
    ``(patch, level, blended_g)`` triples, ``level`` counting from 1 for a
    top-level patch (nested directly inside the root). ``ancestors`` is the
    chain of nodes (starting with the root) that every patch in ``patches``
    is nested inside.

    This lets every patch be written as its own flat HDF5 group, carrying
    only its own already-blended field and nesting depth -- no
    parent/child reference, ``window``, or ``frac_low`` at all (see
    :meth:`ClusterField.write_file`).
    """
    if flat is None:
        flat = []
    for patch in patches:
        blended = _blend_patch_field(ancestors, patch)
        flat.append((patch, level, blended))
        _blend_patches(patch.children, (*ancestors, patch), level + 1, flat)
    return flat


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


def _check_no_overlap(regions):
    r"""
    Raise a clear error if any two refinement regions overlap, after
    parsing each region's ``center`` into plain kpc floats (see
    :func:`~cluster_generator.amr_hierarchy.check_no_overlap`, which does
    the actual comparison -- using each region's *outermost*, level-1 box
    width, since that's the one that must not overlap another region's;
    see :func:`~cluster_generator.amr_hierarchy.region_outer_width`).

    ``RandomClusterField``'s patch combination (see
    ``_interpolate_at_points``) assumes non-overlapping regions -- for an
    overlapping pair, a query point in the overlap only ever sees
    whichever region happens to come first in the list, silently ignoring
    the other one there.

    Parameters
    ----------
    regions : list of dict
        A ``refinement_regions`` list, each with ``center`` and ``width``
        keys (and optionally ``num_levels``/``padding``).
    """
    parsed = [{**r, "center": parse_value(r["center"], "kpc").v} for r in regions]
    try:
        check_no_overlap(parsed)
    except ValueError as e:
        raise ValueError(
            f"{e} Shrink their widths, move the clusters apart, or build the "
            "list with refinement_regions_for_clusters(), which avoids this "
            "automatically."
        ) from e


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
        self.root = None

    @property
    def patches(self):
        r"""
        Top-level locally-refined patches (see ``RandomClusterField``'s
        ``refinement_regions``), i.e. ``self.root.children``. Kept as a
        convenience alias -- ``self.root`` (the coarse grid, itself the
        root of the same recursive patch tree; see :class:`FieldPatch`)
        is the more complete picture.
        """
        return self.root.children if self.root is not None else []

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
                flat_patches = _blend_patches(patches, (self.root,) if self.root is not None else ())
                f.attrs["num_patches"] = np.int32(len(flat_patches))
                for i, (patch, level, blended_g) in enumerate(flat_patches):
                    grp = f.create_group(f"patch_{i:04d}")
                    grp.attrs["level"] = np.int32(level)
                    self._write_patch(grp, patch, blended_g, length_unit, field_unit)

    def _write_patch(self, grp, patch, blended_g, length_unit, field_unit):
        r"""
        Write a single patch's own coordinates and its already-blended
        field (see :func:`_blend_patches`) to an open HDF5 group. Its
        place in the patch tree is recorded separately, as a flat
        ``level`` (see :meth:`write_file`), so this never recurses into
        ``patch.children`` itself.
        """
        for ax, arr in zip("xyz", (patch.x, patch.y, patch.z), strict=True):
            pd = unyt_array(arr, "kpc").to(length_unit)
            d = grp.create_dataset(ax, data=pd.d)
            d.attrs["units"] = str(pd.units)
        for j, ax in enumerate("xyz"):
            pfd = unyt_array(blended_g[j, ...], self.units)
            if field_unit is not None:
                pfd = pfd.to(f"{length_unit}*{field_unit}" if self._vector_potential else field_unit)
            d = grp.create_dataset(f"{self._name}_{ax}", data=pfd.d)
            d.attrs["units"] = str(pfd.units)

    def _interpolate_at_points(self, coords):
        r"""
        Evaluate the field at arbitrary points via tri-linear
        interpolation, walking ``self.root`` (the coarse grid, itself the
        root of the same recursive patch tree -- see :class:`FieldPatch`)
        down through any locally-refined ``patches`` (see
        ``RandomClusterField``'s ``refinement_regions``) whose box
        contains the point, recursing through an arbitrary number of
        nested refinement levels. Patches are assumed non-overlapping
        with their siblings.
        """
        coords = np.asarray(coords)
        v = np.zeros((coords.shape[0], 3))
        return self._apply_patch(v, self.root, coords)

    @staticmethod
    def _apply_patch(v, patch, coords):
        r"""
        Blend a single :class:`FieldPatch` into ``v`` -- the field values
        already evaluated at ``coords`` from ``patch``'s parent (the
        coarse grid, or an enclosing, coarser patch) -- then recurse into
        ``patch.children`` to blend in any further refinement nested
        inside it.
        """
        from scipy.interpolate import RegularGridInterpolator

        w_func = RegularGridInterpolator(
            (patch.x, patch.y, patch.z),
            patch.window,
            bounds_error=False,
            fill_value=0.0,
        )
        w = w_func(coords)
        # outside the patch box, w == 0 (fill_value), so this
        # correction is a no-op and the parent-only value is kept
        correction = 1.0 - (1.0 - np.sqrt(patch.frac_low)) * w
        for i in range(3):
            g_func = RegularGridInterpolator(
                (patch.x, patch.y, patch.z),
                patch.g[i, ...],
                bounds_error=False,
                fill_value=0.0,
            )
            v[:, i] = v[:, i] * correction + g_func(coords)
        for child in patch.children:
            v = ClusterField._apply_patch(v, child, coords)
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
    r"""
    Parameters
    ----------
    refinement_regions : list of dict, optional
        One entry per cluster: a telescoping chain of locally-refined
        boxes, layered on top of the coarse grid to add small-scale power
        it can't resolve (e.g. around a cluster core), each twice the
        resolution of the level just outside it (``refine_by`` is always
        2). Each entry is a dict with keys:

        center : array-like
            The box center [kpc], shared by every level of this region's
            chain.
        width : float
            The *finest* (innermost) level's side length [kpc] -- the one
            actually ``width`` wide, at full resolution; a single value
            applies to all three axes.
        num_levels : int, optional
            The number of refinement levels in this region's chain, each
            twice the resolution of the level just outside it. Going
            outward from the finest box, each coarser level is
            ``padding`` times wider than the level just inside it, down
            to level 1 (nested directly inside the coarse grid). Default:
            1 (a single level, ``width`` wide).
        padding : float, optional
            Width ratio between consecutive levels of this region's chain
            (see above); also leaves room for the taper below to roll off
            before each level's box edge. Default: 1.5.
        taper_alpha : float, optional
            Tukey-window taper parameter used to blend each level of this
            region smoothly into the level just outside it. Default: 0.3.

        Regions must not overlap (checked using each one's widest,
        level-1 box) -- :func:`_check_no_overlap` raises a clear error if
        they do; see :func:`refinement_regions_for_clusters` for a helper
        that sizes and places them automatically. Using
        ``refinement_regions`` requires an isotropic coarse grid.
    """

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
        self.power_spec = power_spec

        if self.refinement_regions:
            if isinstance(prng, np.random.Generator):
                raise ValueError(
                    "When using refinement_regions, 'prng' must be an int seed or "
                    "None (not an existing Generator instance), so independent "
                    "reproducible per-patch seeds can be derived from it."
                )
            seed_seq = np.random.SeedSequence(prng)
            coarse_seed, self._patches_seed_seq = seed_seq.spawn(2)
        else:
            coarse_seed = prng
            self._patches_seed_seq = None

        self.grf = GaussianRandomField(
            self.left_edge, self.right_edge, self.ddims, power_spec, seed=coarse_seed
        )
        self.x = self.grf.x
        self.y = self.grf.y
        self.z = self.grf.z

        num_halos = 1
        self.ctr1 = parse_value(ctr1, "kpc").v
        r1, g1 = load_radial_profile(profile1, self._profile_field)
        self.r1 = parse_value(r1, "kpc").v
        self.g1 = parse_value(g1, self._units)

        if profile2 is not None:
            num_halos += 1
            if ctr2 is None:
                raise RuntimeError("Need to specify 'ctr2' for the second halo!")
            self.ctr2 = parse_value(ctr2, "kpc").v
            r2, g2 = load_radial_profile(profile2, self._profile_field)
            self.r2 = parse_value(r2, "kpc").v
            self.g2 = parse_value(g2, self._units)
        if profile3 is not None:
            num_halos += 1
            if ctr3 is None:
                raise RuntimeError("Need to specify 'ctr3' for the second halo!")
            self.ctr3 = parse_value(ctr3, "kpc").v
            r3, g3 = load_radial_profile(profile3, self._profile_field)
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
        mylog.debug("Scaling the fields by cluster 1.")
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
            mylog.debug("Scaling the fields by cluster 2.")
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
            mylog.debug("Scaling the fields by cluster 3.")
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

        # The coarse grid is itself the root of the patch tree (see
        # FieldPatch): frac_low=0 and window=1 everywhere means
        # _apply_patch's blend reduces to "just use this grid's own data",
        # reproducing the (former) special-cased coarse-grid evaluation
        # exactly, with no separate code path needed.
        self.root = FieldPatch(
            left_edge=self.left_edge,
            right_edge=self.right_edge,
            ddims=self.ddims,
            deltas=self.deltas,
            x=self.x,
            y=self.y,
            z=self.z,
            g=self.g,
            window=np.ones(tuple(self.ddims)),
            frac_low=0.0,
            g_avg=self._g_avg_low,
        )

        if self.refinement_regions:
            self.root.children = self._generate_patches()

    @staticmethod
    def _highpass_filter(field, deltas, k_cut):
        r"""
        Zero out all Fourier modes with :math:`|k| < k_{\rm cut}` in a
        real-space vector field, returning only its small-scale content.
        Implemented with plain ``numpy.fft`` (a self-contained round
        trip), independent of any of ``kspace``'s internal Fourier
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

    def _generate_patch_chain(self, region, seed_seq):
        r"""
        Generate the full telescoping chain of locally-refined
        :class:`FieldPatch` nodes for one ``refinement_regions`` entry:
        level 1 (widest, nested directly inside ``self.root``) down to
        level ``region["num_levels"]`` (narrowest, finest -- the width
        given in ``region["width"]``), each an independent GRF
        realization at twice the resolution of the level just outside it
        (see :func:`~cluster_generator.amr_hierarchy.region_chain_geometry`
        for how each level's box is placed), high-pass filtered above
        that parent level's own Nyquist frequency to keep only the power
        it can't represent, amplitude-normalized so the combined chain
        matches the target profile without double-counting power, and
        tapered to zero at its own box edges so it blends smoothly into
        its parent.

        Parameters
        ----------
        region : dict
            This region's own entry from ``refinement_regions``.
        seed_seq : numpy.random.SeedSequence
            This region's own seed sequence, spawned from
            ``self._patches_seed_seq``; spawned further, here, into one
            seed per level of the chain.
        """
        center = parse_value(region["center"], "kpc").v
        taper_alpha = region.get("taper_alpha", 0.3)
        boxes = region_chain_geometry(self.root.left_edge, self.root.deltas, {**region, "center": center})
        seeds = seed_seq.spawn(len(boxes))

        parent = self.root
        top_patch = None
        for (patch_left, patch_right, patch_ddims, patch_deltas), seed in zip(boxes, seeds, strict=True):
            patch_width = patch_right - patch_left

            grf = GaussianRandomField(patch_left, patch_right, patch_ddims, self.power_spec, seed=seed)
            f_raw = grf.generate_vector_field_realization()
            g_avg_raw = np.sqrt(np.mean(f_raw * f_raw))

            k_cut = np.pi / parent.deltas[0]
            f_high = self._highpass_filter(f_raw, patch_deltas, k_cut)
            g_avg_high = np.sqrt(np.mean(f_high * f_high))
            frac_low = parent.g_avg**2 / (parent.g_avg**2 + g_avg_high**2)

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

            patch = FieldPatch(
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
                g_avg=g_avg_raw,
            )
            if top_patch is None:
                top_patch = patch
            else:
                parent.children = [patch]
            parent = patch

        return top_patch

    def _generate_patches(self):
        _check_no_overlap(self.refinement_regions)
        mylog.info("Generating %d refinement patch(es).", len(self.refinement_regions))
        seeds = self._patches_seed_seq.spawn(len(self.refinement_regions))
        return [
            self._generate_patch_chain(region, seed)
            for region, seed in zip(self.refinement_regions, seeds, strict=True)
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


_AXIS_NAMES = "xyz"


def _read_field_node(h5group, comps):
    r"""
    Read one node (the root, or a single patch group) of a field file
    written by :meth:`ClusterField.write_file` into a plain dict -- ``x``,
    ``y``, ``z`` coordinate arrays (converted to kpc using each dataset's
    own ``units`` attribute, since :meth:`ClusterField.write_file`'s
    ``length_unit`` need not be kpc -- e.g. GAMER's ``B_IC`` files are
    commonly written in Mpc) and ``g`` (stacked field components, shape
    ``(3, nx, ny, nz)``). Every node -- the root or any patch -- has
    exactly this shape: a patch's own field is already the final, blended
    one (see :func:`_blend_patches`), so there's no extra blending data
    (``window``/``frac_low``) to carry.
    """
    node = {ax: unyt_array(h5group[ax][:], h5group[ax].attrs["units"]).to_value("kpc") for ax in _AXIS_NAMES}
    node["g"] = np.stack([h5group[c][:] for c in comps], axis=0)
    return node


def _read_patches(f, comps):
    r"""
    Read the flat ``patch_NNNN`` groups written by
    :meth:`ClusterField.write_file` (see :func:`_blend_patches`) into a
    flat list of node dicts (see :func:`_read_field_node`), each tagged
    with its own ``"level"`` (1 for a patch nested directly inside the
    root, 2 for one nested inside a level-1 patch, etc.).

    Each patch's own field is already the fully-blended one -- there's no
    parent/child reference, ``window``, or ``frac_low`` to read back, and
    (unlike the pre-blended data itself) nothing here depends on knowing
    which patch sits inside which; :func:`plot_field_file` only needs to
    draw patches in increasing ``level`` order so finer ones land on top.
    This keeps the on-disk format itself down to "a flat list of boxes,
    each with a level and its own field data" -- something any tool can
    write, with no tree bookkeeping of its own.
    """
    n = int(f.attrs.get("num_patches", 0))
    groups = [f[f"patch_{i:04d}"] for i in range(n)]
    nodes = [_read_field_node(grp, comps) for grp in groups]
    for node, grp in zip(nodes, groups, strict=True):
        node["level"] = int(grp.attrs["level"])
    return nodes


def read_field_file(filename):
    r"""
    Read a field file written by :meth:`ClusterField.write_file` into a
    plain (root, patches, comps, units) tuple -- no yt or live
    :class:`ClusterField` involved, just ``h5py``.

    Parameters
    ----------
    filename : str
        Path to an HDF5 file written by :meth:`ClusterField.write_file`.

    Returns
    -------
    root : dict
        The coarse grid, as a node dict (see :func:`_read_field_node`).
    patches : list of dict
        Any locally-refined patches (see :func:`_read_patches`), as a flat
        list of node dicts, each also carrying its own nesting ``"level"``
        (1 for a patch nested directly inside the root, 2 for one nested
        inside a level-1 patch, etc.) -- empty if the file has none. Every
        patch's own field is already fully blended with its parent's (see
        :func:`_blend_patches`), so there's no tree to reconstruct here.
    comps : list of str
        The three field component dataset names, e.g.
        ``["magnetic_field_x", "magnetic_field_y", "magnetic_field_z"]``.
    units : str
        The field's units, as written by :meth:`ClusterField.write_file`.
    """
    with h5py.File(filename, "r") as f:
        name = f.attrs["name"]
        comps = [f"{name}_{ax}" for ax in _AXIS_NAMES]
        root = _read_field_node(f, comps)
        patches = _read_patches(f, comps)
        units = f.attrs["units"]
    return root, patches, comps, units


def plot_field_file(
    filename,
    axis="z",
    coord=0.0,
    fig=None,
    axes=None,
    cmap="RdBu_r",
    vmax=None,
    show_grids=True,
    grid_colors=("white", "orange", "red", "cyan"),
):
    r"""
    Read a field file written by :meth:`ClusterField.write_file` (see
    :func:`read_field_file`) and plot a slice through each component, with
    box outlines showing where any locally-refined patches sit.

    Every patch's own on-disk field is already fully blended with its
    parent's (see :func:`_blend_patches`), so this just draws the root's
    slice and then each patch's own slice directly on top, in increasing
    ``level`` order so finer patches land over coarser ones -- no
    additional blending at plot time.

    Parameters
    ----------
    filename : str
        Path to an HDF5 file written by :meth:`ClusterField.write_file`.
    axis : {'x', 'y', 'z'}, optional
        The axis to slice along. Default: 'z'.
    coord : float, optional
        The coordinate [kpc] to slice at, along ``axis``. Default: 0.0.
    fig : matplotlib Figure, optional
        The figure to plot in. Default: None, in which case one is created.
    axes : sequence of 3 matplotlib Axes, optional
        The axes to plot the x/y/z components in, in that order. Default:
        None, in which case three subplots are created.
    cmap : str, optional
        Colormap for the slice. Default: 'RdBu_r'.
    vmax : float, optional
        Symmetric color-scale limit (``-vmax`` to ``vmax``), shared across
        the root grid and every patch so they blend consistently rather
        than each rescaling to its own local range. Default: None, in
        which case half the root grid's own max absolute value is used.
    show_grids : bool, optional
        Whether to outline each grid's (the root's and every patch's) box.
        Default: True.
    grid_colors : sequence of str, optional
        Box outline color per nesting depth (cycled if there are more
        levels than colors); index 0 is the root. Default: ``("white",
        "orange", "red", "cyan")``.

    Returns
    -------
    (fig, axes)
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    axis_idx = _AXIS_NAMES.index(axis)
    plot_axes = [i for i in range(3) if i != axis_idx]
    axis1_name, axis2_name = _AXIS_NAMES[plot_axes[0]], _AXIS_NAMES[plot_axes[1]]

    root, patches, comps, units = read_field_file(filename)
    vmax = vmax if vmax is not None else 0.5 * np.max(np.abs(root["g"]))

    def _extent(c1, c2):
        d1, d2 = c1[1] - c1[0], c2[1] - c2[0]
        return (c1[0] - d1 / 2, c1[-1] + d1 / 2, c2[0] - d2 / 2, c2[-1] + d2 / 2)

    def _plot_node(node, depth):
        coord_arr = node[axis]
        if coord < coord_arr.min() or coord > coord_arr.max():
            return
        k = int(np.argmin(np.abs(coord_arr - coord)))
        c1 = node[axis1_name]
        c2 = node[axis2_name]
        slab = np.take(node["g"], k, axis=1 + axis_idx)  # (3, n1, n2)

        extent = _extent(c1, c2)
        for i, ax_ in enumerate(axes):
            ax_.imshow(
                slab[i].T, origin="lower", extent=extent, cmap=cmap, vmin=-vmax, vmax=vmax, zorder=depth
            )
            if show_grids:
                ax_.add_patch(
                    Rectangle(
                        (extent[0], extent[2]),
                        extent[1] - extent[0],
                        extent[3] - extent[2],
                        fill=False,
                        edgecolor=grid_colors[depth % len(grid_colors)],
                        linewidth=1.2,
                        zorder=10 + depth,
                    )
                )

    if fig is None or axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    _plot_node(root, depth=0)
    for patch in sorted(patches, key=lambda node: node["level"]):
        _plot_node(patch, depth=patch["level"])

    # Multiple overlaid imshow() extents otherwise leave the view autoscaled
    # to whichever patch was drawn last, not the full (root) domain.
    root_extent = _extent(root[axis1_name], root[axis2_name])

    for ax_, comp in zip(axes, comps, strict=True):
        ax_.set_title(comp)
        ax_.set_xlabel(f"{axis1_name} (kpc)")
        ax_.set_ylabel(f"{axis2_name} (kpc)")
        ax_.set_xlim(root_extent[0], root_extent[1])
        ax_.set_ylim(root_extent[2], root_extent[3])
        ax_.set_aspect("equal")

    fig.suptitle(f"Slice at {axis}={coord} kpc [{units}]")

    return fig, axes
