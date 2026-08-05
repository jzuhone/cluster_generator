"""
Code-specific utilities for the ``cluster_generator`` library.
"""

import numpy as np
from pathlib import Path
from unyt import uconcatenate, unyt_array, unyt_quantity

from cluster_generator.model import ClusterModel
from cluster_generator.particles import ClusterParticles
from cluster_generator.utils import mylog, parse_prng


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


def _stream_grid_moments(pos, boxsize, grid_oversample, density_func=None, chunk_size=8_000_000):
    r"""
    Accumulate per-cell moments of a regular sampling grid, assigning each grid
    point to its nearest generator — the O(N)-memory core of the grid-based
    Lloyd relaxation and the cell-integrated mass.

    A regular ``gn**3`` grid (``gn = round((grid_oversample * N)**(1/3))``) is
    streamed in chunks and never materialised in full, so memory stays
    ``O(N + chunk_size)`` even for tens of millions of cells.

    Returns
    -------
    count : (N,) ndarray
        Number of grid points nearest to each generator.
    wsum : (N,) ndarray
        Sum of ``density_func`` over owned grid points (the count when
        ``density_func`` is None).
    wpos_sum : (N, 3) ndarray
        Sum of ``weight * grid_point`` over owned grid points.
    grid_dx : float
        Grid spacing in kpc (a grid cell has volume ``grid_dx**3``).
    """
    from scipy.spatial import cKDTree

    pos = np.asarray(pos, dtype=float)
    n = len(pos)
    boxsize = float(boxsize)
    gn = max(2, int(round((grid_oversample * n) ** (1.0 / 3.0))))
    grid_dx = boxsize / gn
    total = gn * gn * gn

    tree = cKDTree(pos)
    count = np.zeros(n, dtype=float)
    wsum = np.zeros(n, dtype=float)
    wpos_sum = np.zeros((n, 3), dtype=float)

    for start in range(0, total, chunk_size):
        stop = min(start + chunk_size, total)
        flat = np.arange(start, stop)
        # Unravel the flat lattice index into (i, j, k) cell-centre coordinates.
        i, rem = np.divmod(flat, gn * gn)
        j, k = np.divmod(rem, gn)
        gp = np.empty((len(flat), 3), dtype=float)
        gp[:, 0] = (i + 0.5) * grid_dx
        gp[:, 1] = (j + 0.5) * grid_dx
        gp[:, 2] = (k + 0.5) * grid_dx
        w = np.asarray(density_func(gp), dtype=float) if density_func is not None else np.ones(len(flat))
        nearest = tree.query(gp, workers=-1)[1]
        count += np.bincount(nearest, minlength=n)
        wsum += np.bincount(nearest, weights=w, minlength=n)
        for d in range(3):
            wpos_sum[:, d] += np.bincount(nearest, weights=w * gp[:, d], minlength=n)
    return count, wsum, wpos_sum, grid_dx


def _lloyd_relax(
    positions,
    boxsize,
    num_iterations=50,
    tol=1e-3,
    step_damping=0.5,
    density_func=None,
    grid_oversample=8,
    chunk_size=8_000_000,
):
    r"""
    Relax particle positions with a streaming grid Lloyd's algorithm for a
    non-periodic box.

    Each iteration moves every generator toward the (density-)weighted centroid
    of its Voronoi cell, approximated by assigning a regular sampling grid to
    the nearest generator.  The grid is streamed in chunks, so memory stays
    ``O(N + chunk_size)`` and the relaxation scales to tens of millions of
    cells.  When ``density_func`` is given, grid points are weighted by the
    local gas density, driving the mesh toward **equal-mass** cells (AREPO's
    ``REGULARIZE_MESH_CM_DRIFT`` behaviour); otherwise the geometric
    (equal-volume) centroid is used.

    Parameters
    ----------
    positions : (N, 3) ndarray
        Initial positions in kpc within ``[0, boxsize]**3``.
    boxsize : float
        Cubic box side length in kpc.
    num_iterations : int, optional
        Maximum number of Lloyd iterations. Default: 50
    tol : float or None, optional
        Convergence tolerance as a fraction of the mean inter-particle spacing;
        ``None`` runs all iterations. Default: 1e-3
    step_damping : float, optional
        Fraction of the centroid displacement applied per iteration (AREPO's
        ``CellShapingSpeed``). Default: 0.5
    density_func : callable or None, optional
        ``density_func(points)`` -> gas density (Msun/kpc**3); enables
        density-weighted (equal-mass) centroids. Default: None
    grid_oversample : int, optional
        Sampling grid points per generator (grid has ``~grid_oversample * N``
        points).  Higher is more accurate and more expensive. Default: 8
    chunk_size : int, optional
        Grid points processed per streamed chunk (bounds memory).
        Default: 8_000_000

    Returns
    -------
    pos : (N, 3) ndarray
        Relaxed positions, clipped to ``[0, boxsize]**3``.
    """
    pos = np.asarray(positions, dtype=float).copy()
    n = len(pos)
    mean_spacing = boxsize / n ** (1.0 / 3.0)

    for it in range(num_iterations):
        _, wsum, wpos_sum, _ = _stream_grid_moments(pos, boxsize, grid_oversample, density_func, chunk_size)
        centroid = pos.copy()
        valid = wsum > 0
        centroid[valid] = wpos_sum[valid] / wsum[valid, None]
        new_pos = np.clip(pos + step_damping * (centroid - pos), 0.0, boxsize)
        max_disp = np.max(np.linalg.norm(new_pos - pos, axis=1))
        pos = new_pos
        mylog.info(
            "Lloyd's relaxation: iteration %d/%d, max displacement = %.3e mean spacings.",
            it + 1,
            num_iterations,
            max_disp / mean_spacing,
        )
        if tol is not None and max_disp < tol * mean_spacing:
            mylog.info("Lloyd's relaxation: converged after %d iterations.", it + 1)
            break
    return pos


def _exact_voronoi_volumes(pos, boxsize):
    r"""
    Exact bounded Voronoi cell volumes (kpc**3) via voro++ (``pyvoro2``).

    Uses the lean ``cell_measures`` path — voro++ computes volumes in C++ and
    returns a flat ``(N,)`` array with **no** per-cell geometry, so this scales
    to tens of millions of cells (~0.5 GB per 5e5 cells).  Non-periodic box
    walls at ``[0, boxsize]**3`` clip boundary cells exactly as AREPO's mesh
    does.
    """
    pos = np.asarray(pos, dtype=float)
    boxsize = float(boxsize)

    try:
        import pyvoro2
        from pyvoro2.domains import Box
    except ImportError as e:
        raise ImportError(
            "Exact Voronoi volumes require the 'pyvoro2' package (a voro++ "
            "wrapper).  Install it, e.g. `pip install pyvoro2` or "
            "`pip install 'cluster_generator[arepo]'`."
        ) from e

    res = pyvoro2.compute(
        pos,
        domain=Box(((0.0, boxsize),) * 3),
        return_vertices=False,
        return_faces=False,
        return_adjacency=False,
        output="result",
    )
    volumes = np.empty(len(pos), dtype=float)
    volumes[np.asarray(res.ids)] = np.asarray(res.cell_measures, dtype=float)
    return volumes


def _compute_arepo_masses(
    pos,
    boxsize,
    density_func,
    point_density,
    mass_method="integrated",
    grid_oversample=8,
    chunk_size=8_000_000,
):
    r"""
    Per-cell gas masses (Msun) so AREPO's ``density = mass / V_voronoi``
    reproduces the target profile.

    AREPO recomputes cell density from ``mass / V_voronoi`` at startup, so we
    set ``mass_i = ρ_target,i × V_i`` with the **exact** voro++ volume ``V_i``
    (the lean ``cell_measures`` path, which scales to tens of millions of
    cells).  ``mass_method`` selects the density estimate:

    * ``'integrated'`` (default): the cell-averaged density from a streaming
      grid integral, ``(∫_cell ρ dV) / V_grid`` — more accurate in steep
      gradients such as the cluster core, and O(N) in memory.
    * ``'point'``: the density sampled at the generator, ``ρ(x_i)``.

    Exact volumes require ``pyvoro2``; the integrated estimate reuses the same
    streaming grid as :func:`_lloyd_relax`.
    """
    if mass_method not in ("point", "integrated"):
        raise ValueError(f"Unknown mass_method {mass_method!r}. Use 'point' or 'integrated'.")
    volume = _exact_voronoi_volumes(pos, boxsize)  # exact, lean, scales
    if mass_method == "point":
        return np.asarray(point_density, dtype=float) * volume
    if mass_method == "integrated":
        count, wsum, _, grid_dx = _stream_grid_moments(
            pos, boxsize, grid_oversample, density_func, chunk_size
        )
        vol_grid = count * grid_dx**3
        m_int = wsum * grid_dx**3
        # Cell-averaged density; fall back to the point value for empty cells.
        density = np.divide(
            m_int, vol_grid, out=np.asarray(point_density, dtype=float).copy(), where=vol_grid > 0
        )
        return density * volume


def setup_arepo_ics(
    ics,
    boxsize,
    nxb,
    bkg_density,
    ic_file,
    overwrite=False,
    regenerate_particles=False,
    num_lloyd_iterations=50,
    lloyd_tol=1e-3,
    lloyd_damping=0.5,
    grid_oversample=8,
    mass_method="integrated",
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
    nxb : integer
        The number of cells on a side to compute the background from.

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
    lloyd_tol : float or None, optional
        Convergence tolerance for Lloyd's relaxation, as a fraction of the
        mean inter-particle spacing.  Iteration stops early when the
        maximum cell displacement falls below this threshold.  Set to
        ``None`` to always run all ``num_lloyd_iterations`` steps.
        Default: 1e-3
    lloyd_damping : float, optional
        Step-damping factor for Lloyd's relaxation (fraction of centroid
        displacement applied per iteration).  Matches AREPO's
        ``CellShapingSpeed`` parameter.  Default: 0.5
    grid_oversample : int, optional
        Sampling grid points per cell for the streaming Lloyd relaxation and
        the ``'integrated'`` mass estimate (grid has ``~grid_oversample × N``
        points).  Higher is more accurate and more expensive. Default: 8
    mass_method : {'integrated', 'point'}, optional
        How the target density enters the exact-Voronoi mass assignment.
        ``mass = ρ × V_exact`` with the exact voro++ volume; ``'integrated'``
        (default) uses the cell-averaged density from a streaming grid integral
        (most accurate in steep gradients such as the cluster core), ``'point'``
        uses ``ρ(x_i)``.  Exact volumes require ``pyvoro2``.  Default:
        ``'integrated'``
    prng : int, numpy.random.RandomState, or None, optional
        Pseudo-random number generator seed or state. Default: None
    """
    prng = parse_prng(prng)

    parts = ics.setup_particle_ics(regenerate_particles=regenerate_particles, prng=prng)
    src_particle_mass = parts["gas", "particle_mass"][0].to_value("Msun")
    mylog.info("Source particle mass is %g Msun.", src_particle_mass)

    dx = boxsize / nxb
    dV = dx**3
    bkg_density = unyt_quantity(bkg_density, "g/cm**3").to_value("Msun/kpc**3")
    mylog.info("Background cell density is %g Msun/kpc**3.", bkg_density)
    lmin = 0.5 * dx
    lmax = boxsize - 0.5 * dx
    posg = (
        np.mgrid[lmin : lmax : nxb * 1j, lmin : lmax : nxb * 1j, lmin : lmax : nxb * 1j].reshape(3, nxb**3).T
    )
    bkg_particle_mass = dV * bkg_density
    mylog.info("Background particle mass is %g Msun.", bkg_particle_mass)
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
        ("gas", "density"): unyt_array(bkg_density * np.ones(nleft), "Msun/kpc**3"),
        ("gas", "thermal_energy"): unyt_array(np.zeros(nleft), "kpc**2/Myr**2"),
        ("gas", "particle_mass"): unyt_array(bkg_particle_mass * np.ones(nleft), "Msun"),
    }
    parts = parts + ClusterParticles.from_fields(fields)
    new_parts = ics.resample_particle_ics(parts, bkg_density=bkg_density)

    if num_lloyd_iterations > 0:
        gas_pos = new_parts["gas", "particle_position"].to_value("kpc")
        mylog.info(
            "Relaxing %d gas cells using Lloyd's algorithm (%d iterations).",
            len(gas_pos),
            num_lloyd_iterations,
        )
        density_func = _make_density_func(ics, bkg_density)
        gas_pos = _lloyd_relax(
            gas_pos,
            boxsize,
            num_iterations=num_lloyd_iterations,
            tol=lloyd_tol,
            step_damping=lloyd_damping,
            density_func=density_func,
            grid_oversample=grid_oversample,
        )
        new_parts["gas", "particle_position"] = unyt_array(gas_pos, "kpc")
        # Resample density, thermal energy, and velocity from analytic profiles.
        new_parts = ics.resample_particle_ics(new_parts, recalc_mass=False, bkg_density=bkg_density)
        # Set masses so that AREPO's startup calculation of
        # density = mass/vol_voronoi matches the analytic profile.  (AREPO
        # ignores the Density field in the IC file; it always recomputes
        # density from mass/vol_voronoi at initialisation.)  See
        # _compute_arepo_masses for the mass_method options.
        density_arr = new_parts["gas", "density"].to_value("Msun/kpc**3")
        masses = _compute_arepo_masses(
            gas_pos,
            boxsize,
            density_func,
            density_arr,
            mass_method=mass_method,
            grid_oversample=grid_oversample,
        )
        new_parts["gas", "particle_mass"] = unyt_array(masses, "Msun")

    new_parts.write_to_gadget_file(ic_file, boxsize, overwrite=overwrite, code="arepo")


def setup_gizmo_ics(
    ics,
    boxsize,
    bkg_density,
    ic_file,
    overwrite=False,
    regenerate_particles=False,
    num_lloyd_iterations=50,
    lloyd_tol=1e-3,
    lloyd_damping=0.5,
    grid_oversample=8,
    prng=None,
):
    r"""
    Generate initial conditions for the GIZMO code (MFM/MFV/SPH).

    GIZMO is **meshless**: there is no Voronoi tessellation, and each
    particle's density is a kernel estimate over its neighbours,
    ``ρ_i = Σ_j m_j W(|r_i - r_j|, h_i)``.  The right IC is therefore
    **equal-mass** particles whose *number* density follows the target gas
    density, so the kernel estimate reconstructs ``ρ_target`` automatically —
    no per-particle volume or mass assignment is needed (unlike AREPO, this
    path uses no exact Voronoi volumes and does not require ``pyvoro2``).

    Cluster gas particles (equal mass, sampled ``∝ ρ``) are combined with an
    equal-mass background filling the box outside ``r_max``, and the whole
    distribution is relaxed toward a density-weighted centroidal-Voronoi
    ("weighted-Voronoi-tessellation") glass with :func:`_lloyd_relax`, which
    minimises the particle-configuration noise that meshless/SPH density and
    gradient estimators are sensitive to.  Internal energy and velocity are
    resampled from the analytic profiles onto the relaxed positions; masses
    stay equal.

    Parameters
    ----------
    ics : ClusterICs
        The :class:`~cluster_generator.ics.ClusterICs` object describing
        the halo(s).
    boxsize : float
        Side length of the cubic box in kpc.
    bkg_density : float
        Uniform background gas density in g/cm**3.  Sets the background
        *number* density (equal-mass particles), so background cells blend
        smoothly with the cluster sampling where the profile falls to this
        floor near ``r_max``.
    ic_file : str
        Path for the output Gadget-HDF5 IC file.
    overwrite : bool, optional
        Overwrite ``ic_file`` if it already exists. Default: False
    regenerate_particles : bool, optional
        Re-create particle files even if they already exist. Default: False
    num_lloyd_iterations : int, optional
        Maximum number of relaxation iterations. Set to 0 to skip relaxation.
        Default: 50
    lloyd_tol : float or None, optional
        Convergence tolerance as a fraction of the mean inter-particle
        spacing; ``None`` runs all iterations. Default: 1e-3
    lloyd_damping : float, optional
        Step-damping factor for the relaxation. Default: 0.5
    grid_oversample : int, optional
        Sampling grid points per particle for the streaming relaxation.
        Default: 8
    prng : int, numpy.random.RandomState, or None, optional
        Pseudo-random number generator seed or state. Default: None

    Notes
    -----
    HSE is not exact on the meshless discretisation; a short GIZMO settling
    run with velocity damping typically removes any residual startup motions.
    """
    prng = parse_prng(prng)
    bkg_density = unyt_quantity(bkg_density, "g/cm**3").to_value("Msun/kpc**3")
    mylog.info("Background gas density is %g Msun/kpc**3.", bkg_density)

    parts = ics.setup_particle_ics(regenerate_particles=regenerate_particles, prng=prng)
    src_particle_mass = parts["gas", "particle_mass"][0].to_value("Msun")
    mylog.info("Gas particle mass is %g Msun.", src_particle_mass)

    # Equal-mass background on a regular grid: number density = bkg/m0, so the
    # grid spacing satisfies dx**3 = m0 / bkg_density.  All particles share the
    # same mass (what MFM/MFV want); the grid resolution follows from that.
    nxb = max(1, int(round(boxsize / (src_particle_mass / bkg_density) ** (1.0 / 3.0))))
    dx = boxsize / nxb
    lmin, lmax = 0.5 * dx, boxsize - 0.5 * dx
    posg = (
        np.mgrid[lmin : lmax : nxb * 1j, lmin : lmax : nxb * 1j, lmin : lmax : nxb * 1j].reshape(3, nxb**3).T
    )
    mylog.info("Background grid: %d^3 cells, spacing %g kpc.", nxb, dx)

    rmax2 = ics.r_max**2
    idxs = np.sum((posg - ics.center[0].v) ** 2, axis=1) > rmax2[0]
    if ics.num_halos > 1:
        idxs |= np.sum((posg - ics.center[1].v) ** 2, axis=1) > rmax2[1]
    if ics.num_halos > 2:
        idxs |= np.sum((posg - ics.center[2].v) ** 2, axis=1) > rmax2[2]
    nleft = int(idxs.sum())
    fields = {
        ("gas", "particle_position"): unyt_array(posg[idxs, :], "kpc"),
        ("gas", "particle_velocity"): unyt_array(np.zeros((nleft, 3)), "kpc/Myr"),
        ("gas", "density"): unyt_array(bkg_density * np.ones(nleft), "Msun/kpc**3"),
        ("gas", "thermal_energy"): unyt_array(np.zeros(nleft), "kpc**2/Myr**2"),
        # Equal mass to the cluster gas particles (meshless: no volume weighting).
        ("gas", "particle_mass"): unyt_array(src_particle_mass * np.ones(nleft), "Msun"),
    }
    parts = parts + ClusterParticles.from_fields(fields)
    new_parts = ics.resample_particle_ics(parts, bkg_density=bkg_density)

    if num_lloyd_iterations > 0:
        gas_pos = new_parts["gas", "particle_position"].to_value("kpc")
        mylog.info(
            "Relaxing %d gas particles toward a density-weighted glass (%d iterations).",
            len(gas_pos),
            num_lloyd_iterations,
        )
        density_func = _make_density_func(ics, bkg_density)
        gas_pos = _lloyd_relax(
            gas_pos,
            boxsize,
            num_iterations=num_lloyd_iterations,
            tol=lloyd_tol,
            step_damping=lloyd_damping,
            density_func=density_func,
            grid_oversample=grid_oversample,
        )
        new_parts["gas", "particle_position"] = unyt_array(gas_pos, "kpc")
        # Resample density, internal energy, and velocity onto the relaxed
        # positions.  recalc_mass=False keeps the equal particle masses — GIZMO
        # reconstructs density from the kernel, so no mass/volume step is needed.
        new_parts = ics.resample_particle_ics(new_parts, recalc_mass=False, bkg_density=bkg_density)

    new_parts.write_to_gadget_file(ic_file, boxsize, overwrite=overwrite, code="gizmo")


def setup_art_ics(ics):
    pass
