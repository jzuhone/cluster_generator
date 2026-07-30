"""
Code-specific utilities for the ``cluster_generator`` library.
"""

import h5py
import numpy as np
from pathlib import Path
from unyt import uconcatenate, unyt_array, unyt_quantity

from cluster_generator.model import ClusterModel
from cluster_generator.particles import ClusterParticles
from cluster_generator.utils import mylog, parse_prng

# ---------------------------------------------------------------------------
# Optional Numba-accelerated nearest-particle search for Lloyd's relaxation.
# When Numba is available, _nb_find_nearest replaces scipy KDTree.query for
# the grid backend.  The algorithm builds a spatial bucket (uniform hash)
# over the particle positions and searches the 5×5×5 = 125 neighbouring
# buckets for each grid point — O(M × 125 / num_threads) rather than the
# O(M × log N) of a KDTree query, and fully parallel.
# ---------------------------------------------------------------------------
try:
    import numba as _numba

    @_numba.njit(parallel=True, cache=True)
    def _nb_find_nearest(grid_pts, pos, bucket_particles, bucket_offsets, nb, bucket_size):
        """Return nearest-particle index for each grid point (Numba parallel)."""
        M = grid_pts.shape[0]
        nearest = np.empty(M, dtype=_numba.int64)
        for m in _numba.prange(M):
            gx = grid_pts[m, 0]
            gy = grid_pts[m, 1]
            gz = grid_pts[m, 2]
            bi = min(int(gx / bucket_size), nb - 1)
            bj = min(int(gy / bucket_size), nb - 1)
            bk = min(int(gz / bucket_size), nb - 1)
            min_d2 = 1.0e200
            best = 0
            for di in range(-2, 3):
                ii = bi + di
                if ii < 0 or ii >= nb:
                    continue
                for dj in range(-2, 3):
                    jj = bj + dj
                    if jj < 0 or jj >= nb:
                        continue
                    for dk in range(-2, 3):
                        kk = bk + dk
                        if kk < 0 or kk >= nb:
                            continue
                        bidx = (ii * nb + jj) * nb + kk
                        for pi in range(bucket_offsets[bidx], bucket_offsets[bidx + 1]):
                            p = bucket_particles[pi]
                            dx = gx - pos[p, 0]
                            dy = gy - pos[p, 1]
                            dz = gz - pos[p, 2]
                            d2 = dx * dx + dy * dy + dz * dz
                            if d2 < min_d2:
                                min_d2 = d2
                                best = p
            nearest[m] = best
        return nearest

    _HAVE_NUMBA = True
    mylog.debug("Numba found; Lloyd's grid backend will use parallel bucket NNS.")
except ImportError:
    _HAVE_NUMBA = False
    mylog.debug("Numba not found; Lloyd's grid backend will use scipy KDTree.")


def _build_bucket_structure(pos, boxsize):
    """
    Build a CSR-style spatial hash over particle positions.

    Divides the box into a uniform grid with ~1 particle per bucket on
    average.  Returns ``(bucket_particles, bucket_offsets, nb, bucket_size)``
    suitable for passing to :func:`_nb_find_nearest`.
    """
    n = len(pos)
    # target ≈ 1 particle/bucket; cap nb at 400 to bound memory
    nb = min(max(int(round(n ** (1.0 / 3.0))), 2), 400)
    bucket_size = boxsize / nb

    bi = np.minimum((pos[:, 0] / bucket_size).astype(np.int64), nb - 1)
    bj = np.minimum((pos[:, 1] / bucket_size).astype(np.int64), nb - 1)
    bk = np.minimum((pos[:, 2] / bucket_size).astype(np.int64), nb - 1)
    bucket_idx = (bi * nb + bj) * nb + bk

    n_buckets = nb * nb * nb
    counts = np.bincount(bucket_idx, minlength=n_buckets)
    offsets = np.empty(n_buckets + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])

    # Fill bucket_particles using argsort (groups particles by bucket).
    order = np.argsort(bucket_idx, kind="stable").astype(np.int64)

    return order, offsets, nb, bucket_size


def write_amr_particles(
    particles,
    output_filename,
    ptypes,
    ptype_num,
    overwrite=True,
    in_cgs=False,
    format="hdf5",
):
    """
    Write the particles to an HDF5 file to be read in by the GAMER,
    FLASH, or RAMSES codes.

    Parameters
    ----------
    particles : :class:`cluster_generator.particles.ClusterParticles`
        The ClusterParticles instance which will be written.
    output_filename : string
        The file to write the particles to.
    overwrite : boolean, optional
        Overwrite an existing file with the same name. Default: False.
    """
    import h5py
    from scipy.io import FortranFile

    if Path(output_filename).exists() and not overwrite:
        raise OSError(f"Cannot create {output_filename}. It exists and overwrite=False.")
    nparts = [particles.num_particles[ptype] for ptype in ptypes]
    if format == "hdf5":
        write_class = h5py.File
    elif format == "fortran":
        write_class = FortranFile
    num_particles = 0
    with write_class(output_filename, "w") as f:
        pdata = []
        for field in ["particle_position", "particle_velocity", "particle_mass"]:
            fd = uconcatenate([particles[ptype, field] for ptype in ptypes], axis=0)
            if hasattr(fd, "units") and in_cgs:
                fd.convert_to_cgs()
            if format == "hdf5":
                f.create_dataset(field, data=np.asarray(fd))
            else:
                if field == "particle_mass":
                    num_particles = fd.size
                pdata.append(np.asarray(fd).astype("float64").T)
        if format == "hdf5":
            fd = np.concatenate([ptype_num[ptype] * np.ones(nparts[i]) for i, ptype in enumerate(ptypes)])
            f.create_dataset("particle_type", data=fd)
        else:
            f.write_record(num_particles)
            f.write_record(np.vstack(pdata).T)


def setup_gamer_ics(ics, regenerate_particles=False, use_tracers=False):
    r"""

    Generate the "Input_TestProb" lines needed for use
    with the ClusterMerger setup in GAMER. If the particles
    (dark matter and potentially star) have not been
    created yet, they will be created at this step. New profile
    files will also be created which have all fields in CGS units
    for reading into GAMER. If a magnetic field file is present
    in the ICs, a note will be given about how it should be named
    for GAMER to use it.

    Parameters
    ----------
    ics : ClusterICs object
        The ClusterICs object to generate the GAMER ICs from.
    regenerate_particles : boolean, optional
        If particle files have already been created and this
        flag is set to True, the particles will be
        re-created. Default: False
    use_tracers : boolean
        Set to True to add tracer particles. Default: False
    """
    gamer_ptypes = ["dm", "star"]
    if use_tracers:
        gamer_ptypes.insert(0, "tracer")
    gamer_ptype_num = {"tracer": 0, "dm": 2, "star": 3}
    hses = [ClusterModel.from_h5_file(hf) for hf in ics.profiles]
    parts = ics._generate_particles(regenerate_particles=regenerate_particles)
    outlines = [f"Merger_Coll_NumHalos\t\t{ics.num_halos}\t# number of halos"]
    for i in range(ics.num_halos):
        particle_file = f"{ics.basename}_gamerp_{i + 1}.h5"
        if ics.num_particles["star"][i] == 0:
            ptypes = gamer_ptypes[:-1]
        else:
            ptypes = gamer_ptypes
        write_amr_particles(parts[i], particle_file, ptypes, gamer_ptype_num, in_cgs=True, format="hdf5")
        hse_file_gamer = ics.profiles[i].replace(".h5", "_gamer.h5")
        hses[i].write_model_to_h5(hse_file_gamer, overwrite=True, in_cgs=True, r_max=ics.r_max[i])
        vel = ics.velocity[i].to_value("km/s")
        outlines += [
            f"Merger_File_Prof{i + 1}\t\t{hse_file_gamer}\t# profile table of cluster {i + 1}",
            f"Merger_File_Par{i + 1}\t\t{particle_file}\t# particle file of cluster {i + 1}",
            f"Merger_Coll_PosX{i + 1}\t\t{ics.center[i][0].v}\t# X-center of cluster {i + 1} in kpc",
            f"Merger_Coll_PosY{i + 1}\t\t{ics.center[i][1].v}\t# Y-center of cluster {i + 1} in kpc",
            f"Merger_Coll_PosZ{i + 1}\t\t{ics.center[i][2].v}\t# Z-center of cluster {i + 1} in kpc",
            f"Merger_Coll_VelX{i + 1}\t\t{vel[0]}\t# X-velocity of cluster {i + 1} in km/s",
            f"Merger_Coll_VelY{i + 1}\t\t{vel[1]}\t# Y-velocity of cluster {i + 1} in km/s",
            f"Merger_Coll_VelZ{i + 1}\t\t{vel[2]}\t# Z-velocity of cluster {i + 1} in km/s",
        ]
    mylog.info("Write the following lines to Input__TestProblem: ")
    for line in outlines:
        print(line)
    if ics.mag_file is not None:
        mylog.info(
            f"Rename the file '{ics.mag_file}' to 'B_IC' "
            f"and place it in the same directory as the "
            f"Input__* files, and set OPT__INIT_BFIELD_BYFILE "
            f"to 1 in Input__Parameter"
        )


def setup_flash_ics(ics, use_particles=True, regenerate_particles=False):
    r"""

    Generate the "flash.par" lines needed for use
    with the GalaxyClusterMerger setup in FLASH. If the particles
    (dark matter and potentially star) have not been
    created yet, they will be created at this step.

    Parameters
    ----------
    ics : ClusterICs object
        The ClusterICs object to generate the GAMER ICs from.
    use_particles : boolean, optional
        If True, set up particle distributions. Default: True
    regenerate_particles : boolean, optional
        If particle files have already been created, particles
        are being used, and this flag is set to True, the particles
        will be re-created. Default: False
    """
    if use_particles:
        ics._generate_particles(regenerate_particles=regenerate_particles)
    outlines = [f"testSingleCluster\t=\t{ics.num_halos} # number of halos"]
    for i in range(ics.num_halos):
        vel = ics.velocity[i].to("km/s")
        outlines += [
            f"profile{i + 1}\t=\t{ics.profiles[i]}\t# profile table of cluster {i + 1}",
            f"xInit{i + 1}\t=\t{ics.center[i][0]}\t# X-center of cluster {i + 1} in kpc",
            f"yInit{i + 1}\t=\t{ics.center[i][1]}\t# Y-center of cluster {i + 1} in kpc",
            f"vxInit{i + 1}\t=\t{vel[0]}\t# X-velocity of cluster {i + 1} in km/s",
            f"vyInit{i + 1}\t=\t{vel[1]}\t# Y-velocity of cluster {i + 1} in km/s",
        ]
        if use_particles:
            outlines.append(
                f"Merger_File_Par{i + 1}\t=\t{ics.particle_files[i]}\t# particle file of cluster {i + 1}",
            )
    mylog.info("Add the following lines to flash.par: ")
    for line in outlines:
        print(line)


def setup_athena_ics(ics):
    r"""
    Parameters
    ----------
    ics : ClusterICs object
        The ClusterICs object to generate the Athena ICs from.
    """
    mylog.info("Add the following lines to athinput.cluster3d: ")


def setup_enzo_ics(ics):
    r"""
    Parameters
    ----------
    ics : ClusterICs object
        The ClusterICs object to generate the Enzo ICs from.
    """
    pass


def setup_ramses_ics(ics, regenerate_particles=False):
    r"""
    Parameters
    ----------
    ics : ClusterICs object
        The ClusterICs object to generate the Ramses ICs from.
    regenerate_particles : boolean, optional
        If particle files have already been created, particles
        are being used, and this flag is set to True, the particles
        will be re-created. Default: False
    """
    names = ["Main", "Sub", "Third"]
    config_lines = ["# Merger Dynamics Setting, do not change the general format"]
    hses = [ClusterModel.from_h5_file(hf) for hf in ics.profiles]
    parts = ics._generate_particles(regenerate_particles=regenerate_particles)
    fields_to_write = ["radius", "density", "pressure"]
    for i in range(ics.num_halos):
        if i > 0:
            config_lines.append("#")
        config_lines += [f"# {names[i]}", "#", "#", f"Halo {i + 1}"]
        hses[i].write_model_to_binary(
            f"halo{i + 1}_prof.dat",
            overwrite=True,
            in_cgs=True,
            r_max=ics.r_max,
            fields_to_write=fields_to_write,
        )
        vel = ics.velocity[i].to_value("km/s")
        pos = ics.center[i].to_value("kpc")
        config_lines += [
            f"x_cen[kpc]     ={pos[0]:16.6e}",
            f"y_cen[kpc]     ={pos[1]:16.6e}",
            f"z_cen[kpc]     ={pos[2]:16.6e}",
            f"vx_cen[kms]    ={vel[0]:16.6e}",
            f"vy_cen[kms]    ={vel[1]:16.6e}",
            f"vz_cen[kms]    ={vel[2]:16.6e}",
        ]
        write_amr_particles(
            parts[i],
            f"halo{i + 1}_part.dat",
            ["dm"],
            {"dm": 1},
            format="fortran",
            in_cgs=True,
        )
    mylog.info("Simulation setups saved to Merger_Config.txt.")
    np.savetxt("Merger_Config.txt", config_lines, fmt="%s")


def _make_density_func(ics, bkg_density):
    """
    Build a density callable ``rho(pos)`` from a :class:`ClusterICs` object.

    The returned function evaluates the expected gas density (Msun/kpc³) at
    an array of 3-D positions by summing the interpolated density profiles of
    all clusters and applying ``bkg_density`` as a floor.  This is the same
    convention used by :func:`~cluster_generator.particles._sample_clusters`.

    Parameters
    ----------
    ics : ClusterICs
        The ICs object whose ``profiles`` and ``center`` attributes describe
        the cluster(s).
    bkg_density : float
        Background density floor in Msun/kpc³.

    Returns
    -------
    callable
        A function ``rho(pos)`` where *pos* is ``(M, 3)`` in kpc and the
        return value is ``(M,)`` in Msun/kpc³.
    """
    from scipy.interpolate import InterpolatedUnivariateSpline

    splines = []
    for pf, ctr in zip(ics.profiles, ics.center, strict=True):
        m = ClusterModel.from_h5_file(pf)
        spl = InterpolatedUnivariateSpline(m["radius"].v, m["density"].v)
        splines.append((ctr.v, spl))

    def density_func(pos):
        pos = np.asarray(pos)
        dens = np.zeros(len(pos))
        for ctr, spl in splines:
            r = np.sqrt(((pos - ctr[np.newaxis, :]) ** 2).sum(axis=1))
            dens += np.maximum(spl(r), 0.0)
        return np.maximum(dens, bkg_density)

    return density_func


def _lloyd_relax(
    positions,
    boxsize,
    num_iterations=50,
    method="grid",
    tol=1e-3,
    grid_oversample=8,
    density_func=None,
    step_damping=0.5,
):
    r"""
    Relax particle positions using Lloyd's algorithm for a non-periodic box.

    Each iteration moves every point towards the **density-weighted** centroid
    of its Voronoi cell (when ``density_func`` is provided) or the geometric
    centroid (when it is not).

    Using a density-weighted centroid is strongly recommended for non-uniform
    density distributions such as galaxy clusters: pure geometric Lloyd's
    drives cells toward equal *volume*, which in a steep density gradient
    pushes dense-core particles outward.  Density weighting drives cells
    toward equal *mass*, matching the behaviour of AREPO's
    ``REGULARIZE_MESH_CM_DRIFT`` option (which shifts the target centroid in
    the direction of the local density gradient).

    Two backends are available:

    * **grid** (default): Approximates the Voronoi diagram by seeding a
      regular grid of ``grid_oversample × N`` points and assigning each grid
      point to its nearest particle.  Each particle then moves to the
      (density-)weighted centroid of its owned grid points.  An adaptive
      coarse-to-fine schedule automatically uses a 2× coarser grid for early
      iterations (large displacement) and upgrades as displacement falls,
      reducing per-iteration cost without sacrificing final quality.

    Iteration stops after ``num_iterations`` steps or earlier when the
    maximum point displacement in a step falls below ``tol`` times the mean
    inter-particle spacing (whichever comes first).

    Parameters
    ----------
    positions : numpy.ndarray, shape (N, 3)
        Initial positions in kpc, assumed to lie within
        ``[0, boxsize]``\ :sup:`3`.
    boxsize : float
        Side length of the cubic box in kpc.
    num_iterations : int, optional
        Maximum number of Lloyd iterations. Default: 50
    method : str, optional
        Algorithm backend. Only ``'grid'`` is supported. Default: ``'grid'``
    tol : float or None, optional
        Convergence tolerance as a fraction of the mean inter-particle
        spacing.  Iteration stops early when the maximum displacement in a
        single step is less than ``tol × mean_spacing``.  Set to ``None``
        to always run all ``num_iterations`` steps. Default: 1e-3
    grid_oversample : int, optional
        Number of grid points per particle for the ``'grid'`` backend
        (total grid points ≈ ``grid_oversample × N``).  Higher values give
        more accurate centroid estimates at greater memory and time cost.
        Recommended values: 8 (fast), 64 (production quality). Default: 8
    density_func : callable or None, optional
        A function ``density_func(pos)`` that accepts an ``(M, 3)`` array
        of positions in kpc and returns ``(M,)`` gas densities.  When
        provided, each grid point is weighted by its local density, so the
        algorithm converges to **equal-mass** cells rather than equal-volume
        cells.  This is strongly recommended for non-uniform density
        distributions such as galaxy clusters; without it, the relaxation
        pushes dense-core particles outward.  When ``None``, all weights are
        equal (geometric centroid). Default: None
    step_damping : float, optional
        Fraction of the centroid displacement to apply each iteration,
        in ``(0, 1]``.  A value of 1.0 gives the classic Lloyd full step.
        Values smaller than 1 damp overshooting when centroid estimates are
        noisy (e.g. with a coarse grid) and match the behaviour of AREPO's
        ``CellShapingSpeed`` parameter (default 0.5 in AREPO).
        Default: 0.5

    Returns
    -------
    pos : numpy.ndarray, shape (N, 3)
        Relaxed positions, clipped to ``[0, boxsize]``\ :sup:`3`.
    vol_estimates : numpy.ndarray, shape (N,)
        Approximate Voronoi cell volumes in kpc\ :sup:`3`, estimated from
        the full-resolution grid by counting how many grid points each
        particle owns.  Pass these to ``setup_arepo_ics`` / ``relax_arepo_ics``
        so that particle masses are set as ``mass = density × vol_estimates``,
        ensuring AREPO's startup computation of
        ``density = mass / vol_voronoi`` reproduces the analytic profile.

    Notes
    -----
    AREPO ignores the ``Density`` field in the HDF5 IC file and always
    initialises cell density as ``mass / vol_voronoi`` at startup
    (``init.c``).  The ``vol_estimates`` return value is therefore critical:
    without correct mass assignment the resampled density profile will
    differ from the analytic profile by the ratio of old to new Voronoi
    volumes, generating pressure waves as soon as the simulation starts.
    """
    pos = np.asarray(positions, dtype=float).copy()
    n = len(pos)
    mean_spacing = boxsize / n ** (1.0 / 3.0)

    if method == "grid":
        # Adaptive coarse-to-fine grid schedule.
        # Minimum oversample=4 ensures each particle owns enough grid points
        # to get a meaningful centroid estimate (oversample<4 causes particles
        # to snap to grid positions and stop moving).
        # In early iterations the displacement is large; a 2× coarser grid
        # (8× fewer points) gives centroids accurate enough to make progress
        # at a fraction of the cost.  Thresholds (in mean_spacing units) are
        # chosen so the centroid approximation error is well below the current
        # displacement before each upgrade.
        _phases = [
            (max(4, grid_oversample // 4), 0.4),  # (oversample, upgrade_below_disp)
            (max(8, grid_oversample // 2), 0.05),
            (grid_oversample, None),  # final phase
        ]
        # Deduplicate consecutive phases with the same oversample value.
        seen = set()
        phases = []
        for s, thr in _phases:
            if s not in seen:
                seen.add(s)
                phases.append((s, thr))
        # Ensure the last phase always has threshold=None.
        phases[-1] = (phases[-1][0], None)

        def _build_grid(oversample):
            gn = max(2, int(round((oversample * n) ** (1.0 / 3.0))))
            e = (np.arange(gn) + 0.5) * (boxsize / gn)
            gx_, gy_, gz_ = np.meshgrid(e, e, e, indexing="ij")
            gp = np.column_stack([gx_.ravel(), gy_.ravel(), gz_.ravel()])
            gw = density_func(gp).astype(float) if density_func is not None else np.ones(len(gp))
            return gp, gw

        phase_idx = 0
        current_os = phases[0][0]
        grid_pts, grid_weights = _build_grid(current_os)
        mylog.info("Lloyd's grid: starting with oversample=%d (%d grid points).", current_os, len(grid_pts))

        for it in range(num_iterations):
            mylog.info(
                "Lloyd's relaxation (grid): iteration %d/%d (oversample=%d).",
                it + 1,
                num_iterations,
                current_os,
            )
            if _HAVE_NUMBA:
                bucket_particles, bucket_offsets, nb, bucket_size = _build_bucket_structure(pos, boxsize)
                nearest = _nb_find_nearest(grid_pts, pos, bucket_particles, bucket_offsets, nb, bucket_size)
            else:
                from scipy.spatial import KDTree

                _, nearest = KDTree(pos).query(grid_pts, workers=-1)
                nearest = nearest.astype(np.int64)

            wsum = np.bincount(nearest, weights=grid_weights, minlength=n).astype(float)
            valid = wsum > 0
            # Compute density-weighted centroid for each particle.
            centroid = pos.copy()
            for d in range(3):
                sums = np.bincount(nearest, weights=grid_weights * grid_pts[:, d], minlength=n)
                centroid[valid, d] = sums[valid] / wsum[valid]
            # Apply step damping: move only step_damping fraction toward centroid.
            # Matches AREPO's CellShapingSpeed parameter (default 0.5), which
            # prevents overshooting when centroid estimates are noisy.
            new_pos = pos + step_damping * (centroid - pos)
            new_pos = np.clip(new_pos, 0.0, boxsize)

            max_disp = np.max(np.linalg.norm(new_pos - pos, axis=1))
            pos = new_pos
            mylog.info(
                "Lloyd's relaxation (grid): max displacement = %.3e mean spacings.",
                max_disp / mean_spacing,
            )

            # Check convergence.
            if tol is not None and max_disp < tol * mean_spacing:
                mylog.info("Lloyd's relaxation (grid): converged after %d iterations.", it + 1)
                break

            # Upgrade to finer grid if displacement has fallen below threshold.
            upgrade_thr = phases[phase_idx][1]
            if upgrade_thr is not None and max_disp < upgrade_thr * mean_spacing:
                phase_idx += 1
                new_os = phases[phase_idx][0]
                if new_os != current_os:
                    current_os = new_os
                    grid_pts, grid_weights = _build_grid(current_os)
                    mylog.info(
                        "Lloyd's grid: upgraded to oversample=%d (%d grid points).",
                        current_os,
                        len(grid_pts),
                    )

        # Compute Voronoi volume estimates from the full-resolution grid.
        # AREPO ignores the Density field and always sets density = mass/vol_voronoi
        # at startup.  Returning these estimates lets the caller set masses as
        # mass = density × vol_estimate, so mass/vol_voronoi ≈ analytic_density.
        # Always use the full-resolution grid for the best volume accuracy,
        # even if convergence happened during a coarser phase.
        if current_os < grid_oversample:
            grid_pts_final, _ = _build_grid(grid_oversample)
        else:
            grid_pts_final = grid_pts  # already at full resolution
        if _HAVE_NUMBA:
            bp, bo, nb_f, bs_f = _build_bucket_structure(pos, boxsize)
            nearest_final = _nb_find_nearest(grid_pts_final, pos, bp, bo, nb_f, bs_f)
        else:
            from scipy.spatial import KDTree

            _, nearest_final = KDTree(pos).query(grid_pts_final, workers=-1)
            nearest_final = nearest_final.astype(np.int64)
        grid_n_final = max(2, int(round((grid_oversample * n) ** (1.0 / 3.0))))
        grid_dx = boxsize / grid_n_final
        vol_estimates = np.bincount(nearest_final, minlength=n).astype(float) * grid_dx**3
        mylog.info(
            "Lloyd's grid: volume estimates computed (grid_dx=%.3f kpc, median vol=%.3e kpc^3).",
            grid_dx,
            float(np.median(vol_estimates)),
        )
        return pos, vol_estimates

    raise ValueError(f"Unknown method {method!r}. Only 'grid' is supported.")


def setup_arepo_ics(
    ics,
    boxsize,
    ic_file,
    bkg_density,
    overwrite=False,
    regenerate_particles=False,
    num_lloyd_iterations=50,
    lloyd_method="grid",
    lloyd_tol=1e-3,
    grid_oversample=8,
    lloyd_damping=0.5,
    prng=None,
):
    r"""
    Generate initial conditions for the AREPO code.

    Creates cluster gas particles sampled from the density profile(s) and
    fills the remainder of the box with a uniform background of equal-mass
    gas cells. The combined distribution is written to a Gadget-HDF5 file
    suitable for AREPO. Optionally, the gas cell positions can be relaxed
    with Lloyd's algorithm before the final file is written.

    Parameters
    ----------
    ics : ClusterICs
        The :class:`~cluster_generator.ics.ClusterICs` object describing
        the halo(s).
    boxsize : float
        Side length of the cubic box in kpc.
    ic_file : str
        Path for the output Gadget-HDF5 IC file.
    bkg_density : float
        Uniform background gas density in g/cm**3.  Background cells are
        given equal mass to the cluster gas cells.
    overwrite : bool, optional
        Overwrite ``ic_file`` if it already exists. Default: False
    regenerate_particles : bool, optional
        Re-create particle files even if they already exist. Default: False
    num_lloyd_iterations : int, optional
        Maximum number of Lloyd's algorithm iterations used to regularise
        the Voronoi mesh before writing the IC file.  Set to 0 (default)
        to skip relaxation.  Iteration may stop earlier if ``lloyd_tol``
        is satisfied.  After relaxation the cluster profiles are resampled
        at the new cell positions.
    lloyd_method : str, optional
        Backend for Lloyd's algorithm. Only ``'grid'`` is supported. Ignored
        when ``num_lloyd_iterations`` is 0.
    lloyd_tol : float or None, optional
        Convergence tolerance for Lloyd's relaxation, as a fraction of the
        mean inter-particle spacing.  Iteration stops early when the
        maximum cell displacement falls below this threshold.  Set to
        ``None`` to always run all ``num_lloyd_iterations`` steps.
        Default: 1e-3
    grid_oversample : int, optional
        Number of grid points per particle for the ``'grid'`` backend.
        Higher values improve centroid accuracy at greater cost.
        Use 8 (default) for speed or 64 for production-quality ICs.
    lloyd_damping : float, optional
        Step-damping factor for Lloyd's relaxation (fraction of centroid
        displacement applied per iteration).  Matches AREPO's
        ``CellShapingSpeed`` parameter.  Default: 0.5
    prng : int, numpy.random.RandomState, or None, optional
        Pseudo-random number generator seed or state. Default: None
    """
    prng = parse_prng(prng)
    bkg_density = unyt_quantity(bkg_density, "g/cm**3").to("Msun/kpc**3")
    mylog.info("Background cell density is %g Msun/kpc**3.", bkg_density.value)

    parts = ics.setup_particle_ics(regenerate_particles=regenerate_particles, prng=prng)
    src_particle_mass = parts["gas", "particle_mass"][0].to_value("Msun")
    mylog.info("Source particle mass is %g Msun.", src_particle_mass)

    dV = src_particle_mass / bkg_density
    V = boxsize**3
    nnew = int(V / dV)
    posg = prng.uniform(low=0, high=boxsize, size=(nnew, 3))
    rmax2 = ics.r_max**2
    idxs = np.sum((posg - ics.center[0].v) ** 2, axis=1) > rmax2[0]
    if ics.num_halos > 1:
        idxs |= np.sum((posg - ics.center[1].v) ** 2, axis=1) > rmax2[1]
    if ics.num_halos > 2:
        idxs |= np.sum((posg - ics.center[2].v) ** 2, axis=1) > rmax2[2]
    nleft = idxs.sum()
    fields = {
        ("gas", "particle_position"): unyt_array(posg[idxs, :], "kpc"),
        ("gas", "particle_velocity"): unyt_array(np.zeros((nleft, 3)), "kpc/Myr"),
        ("gas", "density"): bkg_density * np.ones(nleft),
        ("gas", "thermal_energy"): unyt_array(np.zeros(nleft), "kpc**2/Myr**2"),
        ("gas", "particle_mass"): unyt_array(src_particle_mass * np.ones(nleft), "Msun"),
    }
    parts = parts + ClusterParticles.from_fields(fields)
    new_parts = ics.resample_particle_ics(parts, bkg_density=bkg_density.value)

    if num_lloyd_iterations > 0:
        gas_pos = new_parts["gas", "particle_position"].to_value("kpc")
        mylog.info(
            "Relaxing %d gas cells using Lloyd's algorithm (%d iterations).",
            len(gas_pos),
            num_lloyd_iterations,
        )
        density_func = _make_density_func(ics, bkg_density.value)
        gas_pos, vol_estimates = _lloyd_relax(
            gas_pos,
            boxsize,
            num_iterations=num_lloyd_iterations,
            method=lloyd_method,
            tol=lloyd_tol,
            grid_oversample=grid_oversample,
            density_func=density_func,
            step_damping=lloyd_damping,
        )
        new_parts["gas", "particle_position"] = unyt_array(gas_pos, "kpc")
        # Resample density, thermal energy, and velocity from analytic profiles.
        new_parts = ics.resample_particle_ics(new_parts, recalc_mass=False, bkg_density=bkg_density.value)
        # Set masses from grid volume estimates so that AREPO's startup
        # calculation of density = mass/vol_voronoi matches the analytic profile.
        # (AREPO ignores the Density field in the IC file; it always recomputes
        # density from mass/vol_voronoi at initialisation.)
        density_arr = new_parts["gas", "density"].to_value("Msun/kpc**3")
        new_parts["gas", "particle_mass"] = unyt_array(density_arr * vol_estimates, "Msun")

    new_parts.write_to_gadget_file(ic_file, boxsize, overwrite=overwrite, code="arepo")


def resample_arepo_ics(ics, infile, outfile, bkg_density, overwrite=False):
    parts = ClusterParticles.from_gadget_file(infile)
    bkg_density = unyt_quantity(bkg_density, "g/cm**3").to("Msun/kpc**3")
    new_parts = ics.resample_particle_ics(parts, recalc_mass=True, bkg_density=bkg_density.value)
    with h5py.File(infile, "r") as f:
        boxsize = f["Header"].attrs["BoxSize"]
    new_parts.write_to_gadget_file(outfile, boxsize, overwrite=overwrite, code="arepo")


def relax_arepo_ics(
    ics,
    infile,
    outfile,
    bkg_density,
    num_iterations=50,
    lloyd_method="grid",
    lloyd_tol=1e-3,
    grid_oversample=8,
    lloyd_damping=0.5,
    overwrite=False,
):
    r"""
    Relax the Voronoi mesh in a set of AREPO initial conditions using
    Lloyd's algorithm, then resample the cluster profiles onto the new mesh.

    This function reads an existing Gadget-HDF5 IC file produced by
    :func:`setup_arepo_ics`, iteratively moves each gas cell seed point to
    the centroid of its Voronoi cell (Lloyd's algorithm), and writes a new
    IC file with the updated positions. After relaxation the density,
    thermal energy, and velocity fields are resampled from the cluster
    profile(s) at the new cell positions.

    Parameters
    ----------
    ics : ClusterICs
        The :class:`~cluster_generator.ics.ClusterICs` object that was used
        to produce ``infile``.
    infile : str
        Path to the input Gadget-HDF5 IC file (e.g. from
        :func:`setup_arepo_ics`).
    outfile : str
        Path for the relaxed output IC file.
    bkg_density : float
        Background gas density in g/cm**3, used as a density floor when
        resampling the cluster profile onto the relaxed mesh.
    num_iterations : int, optional
        Maximum number of Lloyd relaxation iterations.  Iteration may stop
        earlier if ``lloyd_tol`` is satisfied. Default: 50
    lloyd_method : str, optional
        Backend for Lloyd's algorithm. Only ``'grid'`` is supported.
    lloyd_tol : float or None, optional
        Convergence tolerance as a fraction of the mean inter-particle
        spacing.  Iteration stops early when the maximum cell displacement
        falls below this threshold.  Set to ``None`` to always run all
        ``num_iterations`` steps. Default: 1e-3
    grid_oversample : int, optional
        Number of grid points per particle for the ``'grid'`` backend.
        Higher values improve centroid accuracy at greater cost.
        Use 8 (default) for speed or 64 for production-quality ICs.
    lloyd_damping : float, optional
        Step-damping factor for Lloyd's relaxation (fraction of centroid
        displacement applied per iteration).  Matches AREPO's
        ``CellShapingSpeed`` parameter.  Default: 0.5
    overwrite : bool, optional
        Overwrite ``outfile`` if it already exists. Default: False

    See Also
    --------
    setup_arepo_ics : Generate AREPO ICs with optional inline relaxation.
    resample_arepo_ics : Resample profiles onto a pre-relaxed mesh.
    """
    parts = ClusterParticles.from_gadget_file(infile)
    bkg_density = unyt_quantity(bkg_density, "g/cm**3").to("Msun/kpc**3")

    with h5py.File(infile, "r") as f:
        boxsize = float(f["Header"].attrs["BoxSize"])

    gas_pos = parts["gas", "particle_position"].to_value("kpc")
    mylog.info(
        "Relaxing %d gas cells using Lloyd's algorithm (up to %d iterations).",
        len(gas_pos),
        num_iterations,
    )
    gas_pos, vol_estimates = _lloyd_relax(
        gas_pos,
        boxsize,
        num_iterations=num_iterations,
        method=lloyd_method,
        tol=lloyd_tol,
        grid_oversample=grid_oversample,
        density_func=_make_density_func(ics, bkg_density.value),
        step_damping=lloyd_damping,
    )
    parts["gas", "particle_position"] = unyt_array(gas_pos, "kpc")

    new_parts = ics.resample_particle_ics(parts, recalc_mass=False, bkg_density=bkg_density.value)
    density_arr = new_parts["gas", "density"].to_value("Msun/kpc**3")
    new_parts["gas", "particle_mass"] = unyt_array(density_arr * vol_estimates, "Msun")
    new_parts.write_to_gadget_file(outfile, boxsize, overwrite=overwrite, code="arepo")


def setup_gizmo_ics(ics):
    r"""
    Parameters
    ----------
    ics : ClusterICs object
        The ClusterICs object to generate the GIZMO funcs from.
    """
    pass


def setup_art_ics(ics):
    pass
