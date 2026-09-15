"""
Shared block-structured AMR hierarchy built on `pyAMReX
<https://github.com/AMReX-Codes/pyamrex>`_, used by both
:mod:`cluster_generator.datasets` (uniform/AMR grids built directly
from 1D radial profiles) and :mod:`cluster_generator.fields`
(locally-refined Gaussian random field patches).

``pyamrex`` is not distributed on PyPI -- install it with::

    conda install -c conda-forge pyamrex

(a ``nompi`` build is sufficient; no MPI is required for single-node
``cluster_generator`` use). The import is deferred so that the rest of the
package keeps working without it; only the AMR-specific code paths raise.
"""

import numpy as np
import pathlib as pt
from unyt import unyt_array

from cluster_generator.model import ClusterModel
from cluster_generator.utils import ensure_ytarray, mylog

try:
    import amrex.space3d as amr

    _HAVE_AMREX = True
except ImportError:
    amr = None
    _HAVE_AMREX = False

_AMREX_INITIALIZED = False


def require_amrex():
    """Raise a clear, actionable error if ``pyamrex`` isn't installed."""
    if not _HAVE_AMREX:
        raise ImportError(
            "This feature requires the 'pyamrex' package, which is not "
            "installable via pip. Install it with:\n"
            "    conda install -c conda-forge pyamrex\n"
            "(a 'nompi' build is sufficient; no MPI is required for "
            "single-node cluster_generator use)."
        )


def _ensure_initialized():
    global _AMREX_INITIALIZED
    require_amrex()
    if not _AMREX_INITIALIZED:
        amr.initialize([])
        _AMREX_INITIALIZED = True


def check_no_overlap(regions):
    r"""
    Raise a clear error if any two regions overlap, approximating each as
    a sphere of radius ``width/2`` centered on ``center``, where ``width``
    is each region's *outermost* (level-1, widest) box width -- see
    :func:`region_outer_width`. Used directly by :class:`AMRHierarchy`,
    and by :class:`cluster_generator.fields.RandomClusterField` (which
    parses ``center``'s/``width``'s units first).

    Every level of a given region's chain (see :func:`region_chain_geometry`)
    is centered on the same point and narrower than the level outside it,
    so checking only the widest (outermost) box per region is sufficient:
    if those don't overlap, no narrower, same-centered box nested inside
    them can overlap another region's either.
    """
    n = len(regions)
    for i in range(n):
        ci = np.asarray(regions[i]["center"], dtype=float)
        wi = np.max(region_outer_width(regions[i]))
        for j in range(i + 1, n):
            cj = np.asarray(regions[j]["center"], dtype=float)
            wj = np.max(region_outer_width(regions[j]))
            dist = np.linalg.norm(ci - cj)
            min_sep = 0.5 * (wi + wj)
            if dist < min_sep:
                raise ValueError(
                    f"refinement regions {i} and {j} overlap: their centers are "
                    f"{dist:.1f} kpc apart, but (outermost) widths {wi:.1f} and "
                    f"{wj:.1f} kpc require at least {min_sep:.1f} kpc of separation."
                )


def _snap_region_indices(parent_left_edge, parent_deltas, center, width):
    r"""
    Pure-numpy geometry core underlying :func:`snap_patch_geometry`: snap
    a region's physical ``center``/``width`` [kpc] onto
    ``parent_left_edge``/``parent_deltas``'s cell grid, returning the
    enclosing ``(lo_idx, hi_idx)`` cell-index range (inclusive) in the
    *parent*'s own index space.
    """
    parent_left_edge = np.asarray(parent_left_edge, dtype=float)
    parent_deltas = np.asarray(parent_deltas, dtype=float)
    center = np.asarray(center, dtype=float)
    width = np.atleast_1d(np.asarray(width, dtype=float))
    if width.size == 1:
        width = np.repeat(width, 3)

    lo_idx = np.floor((center - 0.5 * width - parent_left_edge) / parent_deltas).astype(int)
    hi_idx = np.ceil((center + 0.5 * width - parent_left_edge) / parent_deltas).astype(int) - 1
    hi_idx = np.maximum(hi_idx, lo_idx)
    return lo_idx, hi_idx


def snap_patch_geometry(parent_left_edge, parent_deltas, center, width, refine_by):
    r"""
    Snap a region's physical ``center``/``width`` [kpc] onto its parent
    grid's cell lines and refine by ``refine_by``, returning the resulting
    ``(left_edge, right_edge, ddims, deltas)`` as plain numpy arrays.

    Pure geometry math with **no pyamrex dependency** -- shared between
    :class:`AMRHierarchy` (via :func:`region_chain_geometry` and
    :func:`_edges_to_box`, which wraps the same snapped edges in an
    actual ``amrex.Box``) and
    :class:`cluster_generator.fields.RandomClusterField`'s
    ``refinement_regions``, which places patches the same way but doesn't
    need pyamrex at all for it.
    """
    parent_left_edge = np.asarray(parent_left_edge, dtype=float)
    parent_deltas = np.asarray(parent_deltas, dtype=float)
    lo_idx, hi_idx = _snap_region_indices(parent_left_edge, parent_deltas, center, width)

    ddims = (hi_idx - lo_idx + 1) * refine_by
    left_edge = parent_left_edge + lo_idx * parent_deltas
    right_edge = parent_left_edge + (hi_idx + 1) * parent_deltas
    deltas = (right_edge - left_edge) / ddims
    return left_edge, right_edge, ddims, deltas


def _edges_to_box(left_edge, ddims, domain_left_edge, deltas):
    r"""
    Convert an already-snapped ``(left_edge, ddims)`` (e.g. one entry of
    :func:`region_chain_geometry`'s return value) into an ``amrex.Box``,
    given the level's own ``deltas`` and the overall domain's
    ``domain_left_edge`` (the same for every level). ``left_edge`` is
    guaranteed (by construction) to fall exactly on a cell line of this
    resolution, up to floating point.
    """
    lo_idx = np.round((left_edge - domain_left_edge) / deltas).astype(int)
    hi_idx = lo_idx + np.asarray(ddims, dtype=int) - 1
    return amr.Box(amr.IntVect(*lo_idx.tolist()), amr.IntVect(*hi_idx.tolist()))


def _refine_domain_geometry(parent_geom, refine_by, real_box, coord_sys, periodic):
    domain = parent_geom.domain
    lo = np.array([domain.small_end[i] for i in range(3)])
    hi = np.array([domain.big_end[i] for i in range(3)])
    fine_lo = lo * refine_by
    fine_hi = (hi + 1) * refine_by - 1
    fine_box = amr.Box(amr.IntVect(*fine_lo.tolist()), amr.IntVect(*fine_hi.tolist()))
    return amr.Geometry(fine_box, real_box, coord_sys, periodic)


def region_outer_width(region):
    r"""
    The width [kpc] of ``region``'s widest (level-1, outermost) box.

    ``region["width"]`` is the width of the *finest* (innermost, level
    ``region["num_levels"]``) box -- the one actually centered on the
    cluster at full resolution. Going outward, each coarser level's box
    is ``region.get("padding", 1.5)`` times wider than the level just
    inside it, so the outermost (level 1) box -- the one that must not
    overlap any other region's -- is ``width * padding**(num_levels - 1)``.
    """
    num_levels = int(region.get("num_levels", 1))
    padding = region.get("padding", 1.5)
    width = np.atleast_1d(np.asarray(region["width"], dtype=float))
    return width * padding ** (num_levels - 1)


def region_chain_geometry(root_left_edge, root_deltas, region):
    r"""
    Compute the ``(left_edge, right_edge, ddims, deltas)`` box at every
    refinement level of one ``refinement_regions`` entry (the schema used
    by both :class:`cluster_generator.fields.RandomClusterField` and
    :class:`AMRHierarchy`): a cube of ``region["width"]`` [kpc] centered
    on ``region["center"]`` [kpc], refined ``region["num_levels"]`` times
    relative to the root grid, always by a factor of 2 per level.

    Going outward from the finest (innermost) level -- the one actually
    ``width`` wide -- each coarser level's box is ``region.get("padding",
    1.5)`` times wider than the level just inside it, so the whole chain
    telescopes: level 1 (the widest, nested directly inside the root) is
    ``width * padding**(num_levels - 1)`` wide, down to level
    ``num_levels`` (``width`` itself).

    Each level's box is snapped onto its own *immediate* parent's cell
    lines (level 1 onto the root's, level 2 onto level 1's, and so on --
    not all of them directly onto the root's), so it stays tight to the
    requested width even when that width is only a few root cells across.
    Containment then follows automatically for any ``padding >= 1``: each
    level's own edge is (by its own construction, one call up the chain)
    at or beyond its unsnapped target edge, and a strictly narrower,
    same-centered target snapped against that same edge can only land at
    or inside it.

    Pure geometry math with **no pyamrex dependency** -- shared between
    :class:`AMRHierarchy` (which wraps each level's box in an actual
    ``amrex.Box``) and :class:`cluster_generator.fields.RandomClusterField`'s
    ``refinement_regions``, which places patches the same way but doesn't
    need pyamrex at all for it.

    Returns
    -------
    list of (left_edge, right_edge, ddims, deltas)
        One entry per level, ordered level 1 (outermost) to level
        ``num_levels`` (innermost, finest).
    """
    num_levels = int(region.get("num_levels", 1))
    padding = region.get("padding", 1.5)
    if padding < 1:
        raise ValueError(
            f"refinement region centered at {region['center']} kpc: 'padding' "
            f"must be >= 1 (each level must be at least as wide as the level "
            f"just inside it); got {padding}."
        )
    center = np.asarray(region["center"], dtype=float)
    width = np.atleast_1d(np.asarray(region["width"], dtype=float))

    boxes = []
    parent_left_edge, parent_deltas = root_left_edge, root_deltas
    for depth in range(1, num_levels + 1):
        w = width * padding ** (num_levels - depth)
        box = snap_patch_geometry(parent_left_edge, parent_deltas, center, w, 2)
        boxes.append(box)
        parent_left_edge, parent_deltas = box[0], box[3]

    return boxes


class AMRLevel:
    """
    One level of an :class:`AMRHierarchy`: geometry, box layout, and field
    data (a ``MultiFab`` with one component per name in
    ``AMRHierarchy.field_names``).
    """

    def __init__(self, geom, box_array, ncomp, nghost=0):
        self.geom = geom
        self.box_array = box_array
        self.dmap = amr.DistributionMapping(box_array)
        self.multifab = amr.MultiFab(box_array, self.dmap, ncomp, nghost)
        self.multifab.set_val(0.0)

    @property
    def deltas(self):
        return np.array(self.geom.data().CellSize())

    @property
    def prob_lo(self):
        return np.array(self.geom.data().ProbLo())

    def fill_component(self, comp, func, accumulate=False):
        r"""
        Fill component ``comp`` of every box on this level by calling
        ``func(x, y, z)`` -- 1-D cell-center coordinate arrays [kpc], the
        same broadcasting convention used throughout
        :mod:`cluster_generator.fields` -- and assigning the (nx, ny, nz)
        result. If ``accumulate`` is True, the result is added to the
        component's existing values instead of overwriting them (e.g. to
        superpose multiple clusters' density onto the same grid).
        """
        dx = self.deltas
        plo = self.prob_lo
        for mfi in self.multifab:
            box = mfi.validbox()
            lo = np.array([box.small_end[i] for i in range(3)])
            hi = np.array([box.big_end[i] for i in range(3)])
            shape = hi - lo + 1
            x = plo[0] + dx[0] * (lo[0] + 0.5 + np.arange(shape[0]))
            y = plo[1] + dx[1] * (lo[1] + 0.5 + np.arange(shape[1]))
            z = plo[2] + dx[2] * (lo[2] + 0.5 + np.arange(shape[2]))
            arr = np.asarray(self.multifab.array(mfi), copy=False)
            # pyamrex's MultiFab array view is a numpy (C-order) reinterpretation
            # of AMReX's Fortran-ordered FArrayBox memory, so its spatial axes
            # come out reversed (z, y, x) relative to func's (x, y, z)-shaped
            # result -- transpose to match before assigning.
            values = np.transpose(func(x, y, z), (2, 1, 0))
            if accumulate:
                arr[comp, ...] += values
            else:
                arr[comp, ...] = values

    @property
    def num_cells(self):
        return self.box_array.numPts


class AMRHierarchy:
    r"""
    A block-structured multi-level grid hierarchy: a uniform base (level 0)
    domain plus zero or more locally-refined levels, each refined by a
    factor of 2 relative to its parent.

    Parameters
    ----------
    left_edge, right_edge : array-like, shape (3,)
        Physical domain bounds of the base level [kpc].
    base_dims : array-like of int, shape (3,)
        Number of cells along each axis at the base (level 0) resolution.
    field_names : list of str
        Component names carried by every level's ``MultiFab``, in order.
    regions : list of dict, optional
        Locally-refined regions, one per cluster, in the same schema as
        :class:`cluster_generator.fields.RandomClusterField`'s
        ``refinement_regions``: each a dict with ``center`` [kpc] and
        ``width`` [kpc] -- the size of the *finest* box, centered on
        ``center`` -- refined ``num_levels`` times relative to this
        hierarchy's root, always by a factor of 2 per level. Going
        outward from that finest box, each coarser level is
        ``padding`` (default 1.5) times wider than the level just inside
        it, down to level 1 (nested directly inside the root); see
        :func:`region_chain_geometry`. Regions must not overlap (checked
        using each one's widest, level-1 box).
    max_grid_size : int, optional
        Maximum box size along any axis; larger regions are automatically
        split into multiple boxes of at most this size. Default: 32.
    """

    def __init__(self, left_edge, right_edge, base_dims, field_names, regions=None, max_grid_size=32):
        _ensure_initialized()

        left_edge = np.asarray(left_edge, dtype=float)
        right_edge = np.asarray(right_edge, dtype=float)
        base_dims = np.asarray(base_dims, dtype=int)

        self.field_names = list(field_names)
        self.ncomp = len(self.field_names)
        self.max_grid_size = max_grid_size

        self.real_box = amr.RealBox(*left_edge.tolist(), *right_edge.tolist())
        self.coord_sys = amr.CoordSys.cartesian
        self.periodic = [0, 0, 0]

        base_box = amr.Box(amr.IntVect(0, 0, 0), amr.IntVect(*(base_dims - 1).tolist()))
        base_geom = amr.Geometry(base_box, self.real_box, self.coord_sys, self.periodic)
        base_ba = amr.BoxArray(base_box)
        base_ba.max_size(max_grid_size)
        base_left_edge = np.array(base_geom.data().ProbLo())
        base_deltas = np.array(base_geom.data().CellSize())

        self.levels = [AMRLevel(base_geom, base_ba, self.ncomp)]
        self.ref_ratios = []

        regions = regions or []
        if regions:
            check_no_overlap(regions)
            region_boxes = [region_chain_geometry(base_left_edge, base_deltas, r) for r in regions]
            num_levels = [int(r.get("num_levels", 1)) for r in regions]
            max_levels = max(num_levels)
            for depth in range(1, max_levels + 1):
                deltas = base_deltas / 2**depth
                geom = _refine_domain_geometry(
                    base_geom, 2**depth, self.real_box, self.coord_sys, self.periodic
                )
                boxes = [
                    _edges_to_box(boxes_i[depth - 1][0], boxes_i[depth - 1][2], base_left_edge, deltas)
                    for boxes_i, n in zip(region_boxes, num_levels, strict=True)
                    if n >= depth
                ]
                ba = amr.BoxArray(amr.Vector_Box(boxes))
                ba.max_size(max_grid_size)
                self.levels.append(AMRLevel(geom, ba, self.ncomp))
                self.ref_ratios.append(2)

    def fill_field(self, name, func, accumulate=False):
        """Fill the field ``name`` on every level by calling ``func(x, y, z)``."""
        comp = self.field_names.index(name)
        for level in self.levels:
            level.fill_component(comp, func, accumulate=accumulate)

    def write_plotfile(self, dirname, time=0.0, job_info=None):
        """
        Write this hierarchy to an AMReX plotfile directory, natively
        readable by ``yt.load()`` via yt's ``amrex``/``boxlib`` frontend.

        Parameters
        ----------
        dirname : str or path-like
            The plotfile directory to create.
        time : float, optional
            The simulation time to record in the plotfile. Default: 0.0.
        job_info : dict, optional
            If given, ``key = value`` lines are written to a ``job_info``
            file inside ``dirname`` -- yt's ``amrex``/``boxlib`` frontend
            subclasses (see :mod:`cluster_generator.frontend`) use this
            file to recognize and configure a dataset via their
            ``_subtype_keyword``/``_is_valid`` mechanism.
        """
        require_amrex()
        mfs = [level.multifab for level in self.levels]
        geoms = [level.geom for level in self.levels]
        level_steps = [0] * len(self.levels)
        ref_ratio_vecs = [amr.IntVect(r, r, r) for r in self.ref_ratios]
        amr.write_multi_level_plotfile(
            str(dirname), mfs, self.field_names, geoms, time, level_steps, ref_ratio_vecs
        )
        if job_info:
            with open(pt.Path(dirname) / "job_info", "w") as f:
                for key, value in job_info.items():
                    f.write(f"{key} = {value}\n")


def load_radial_profile(profile, profile_field):
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


def _shrink_widths_to_avoid_overlap(centers, widths, margin):
    r"""
    Shrink patch widths, pairwise, so that no two of the (spherical
    approximations of the) regions in ``centers``/``widths`` overlap,
    leaving ``margin`` as a fractional buffer on top of that. Widths only
    ever shrink, never grow, and clusters that never conflict with another
    are left untouched.

    Parameters
    ----------
    centers : list of ndarray
        Cluster centers [kpc].
    widths : list of float
        Initial (pre-overlap-avoidance) widths [kpc], one per center.
    margin : float
        Fractional buffer subtracted from each pairwise center-to-center
        distance before comparing against the sum of half-widths.

    Returns
    -------
    list of float
        The (possibly shrunk) widths.
    """
    n = len(centers)
    widths = list(widths)
    for _pass in range(n + 1):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(centers[i] - centers[j])
                allowed_sum = max(dist * (1.0 - margin), 0.0)
                total = widths[i] + widths[j]
                if total > allowed_sum:
                    scale = allowed_sum / total if total > 0 else 0.0
                    new_i, new_j = widths[i] * scale, widths[j] * scale
                    mylog.info(
                        "refinement_regions_for_clusters: clusters %d and %d are "
                        "%.1f kpc apart; shrinking their widths from (%.1f, %.1f) "
                        "to (%.1f, %.1f) kpc to avoid overlap.",
                        i,
                        j,
                        dist,
                        widths[i],
                        widths[j],
                        new_i,
                        new_j,
                    )
                    widths[i], widths[j] = new_i, new_j
                    changed = True
        if not changed:
            break
    return widths


def refinement_regions_for_clusters(
    centers,
    profiles,
    profile_field,
    width_frac=0.1,
    num_levels=2,
    padding=1.5,
    taper_alpha=0.3,
    r_max=None,
    max_width=None,
    overlap_margin=0.1,
):
    r"""
    Build a ``refinement_regions`` list (see
    :class:`cluster_generator.fields.RandomClusterField`, or
    :class:`AMRHierarchy`'s ``regions``) with one refinement region
    centered on each cluster, sized automatically from that cluster's own
    profile rather than picked by hand.

    The *outermost* (level-1, widest) box of each region -- the one that
    must not overlap another region's, and beyond which no refinement is
    applied at all -- is set to twice the radius at which the cluster's
    profile first falls to ``width_frac`` of its central (innermost
    tabulated) value, i.e. it spans out to where the field has dropped
    off, not an arbitrary fixed size. The corresponding *finest* box width
    (``region["width"]``, see :func:`region_chain_geometry`) is then this
    outer extent divided down by ``padding**(num_levels - 1)``. Placement
    uses ``centers`` directly, so it stays in sync with whatever you pass
    as ``ctr1``/``ctr2``/``ctr3`` to a field constructor, or ``center``/
    ``velocity`` to a dataset's ``add_model``.

    A slowly-declining profile sampled out to large radii can produce a
    width far bigger than intended (and, in turn, a patch grid so large it
    exhausts memory) -- pass ``max_width`` (and/or ``r_max``) to guard
    against this, especially the first time you use a given profile.

    Parameters
    ----------
    centers : sequence of array-like
        The cluster centers [kpc], in the same order as ``profiles``.
    profiles : sequence of ClusterModel, string, or (r, g) array-like
        The corresponding profiles, one per center.
    profile_field : str
        The name of the field that is being profiled.
    width_frac : float or sequence of float, optional
        The fraction of each profile's central value used to define each
        region's outermost extent (see above). A single value applies to
        every cluster; a sequence must match ``centers`` in length.
        Default: 0.1.
    num_levels : int or sequence of int, optional
        The number of refinement levels for each region, each twice the
        resolution of the level just outside it (``refine_by`` is always
        2). A single value applies to every cluster; a sequence must match
        ``centers`` in length. Default: 2.
    padding : float, optional
        Shared width ratio between consecutive levels for every region.
        Default: 1.5.
    taper_alpha : float, optional
        Shared Tukey taper parameter for every region's every level (only
        meaningful for :class:`cluster_generator.fields.RandomClusterField`'s
        ``refinement_regions``; ignored by :class:`AMRHierarchy`). Default:
        0.3.
    r_max : float, optional
        If given, ignore profile values beyond this radius [kpc] when
        determining each region's outer extent (matching the field
        constructor's own ``r_max``).
    max_width : float, optional
        If given, cap every region's outer extent at this value [kpc] (a
        hard safety limit, applied after ``width_frac``/``r_max``).
        Default: None (no cap).
    overlap_margin : float, optional
        After sizing, outer extents are shrunk (pairwise, never grown) so
        that no two regions overlap -- clusters close together relative to
        their computed extents get smaller regions instead of colliding.
        This is the fractional buffer left on top of that, e.g. 0.1 keeps
        regions at least 10% farther apart than the bare minimum. A shrink
        is logged whenever it happens. Default: 0.1.

    Returns
    -------
    list of dict
        Ready to pass as ``refinement_regions`` to a
        :class:`cluster_generator.fields.RandomClusterField` subclass, or
        as ``regions`` to :class:`AMRHierarchy`/
        :class:`cluster_generator.datasets.AMRClusterDataset`.
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
    num_levels_list = _broadcast(num_levels, "num_levels")
    centers_kpc = [np.asarray(ensure_ytarray(c, "kpc").d) for c in centers]

    outer_widths = []
    for profile, wf in zip(profiles, width_fracs, strict=True):
        r, g = load_radial_profile(profile, profile_field)
        r = ensure_ytarray(r, "kpc").d
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
        outer_widths.append(width)

    outer_widths = _shrink_widths_to_avoid_overlap(centers_kpc, outer_widths, overlap_margin)

    regions = []
    for center, outer_width, nl in zip(centers, outer_widths, num_levels_list, strict=True):
        if outer_width <= 0:
            mylog.warning(
                "refinement_regions_for_clusters: dropping the region at %s -- "
                "its width shrank to zero avoiding overlap with another cluster "
                "(they may coincide).",
                center,
            )
            continue
        regions.append(
            {
                "center": center,
                "width": outer_width / padding ** (nl - 1),
                "num_levels": nl,
                "padding": padding,
                "taper_alpha": taper_alpha,
            }
        )
    return regions
