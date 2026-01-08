"""
Code-specific utilities for the ``cluster_generator`` library.
"""

import h5py
import numpy as np
from pathlib import Path
from unyt import uconcatenate, unyt_array

from cluster_generator.model import ClusterModel
from cluster_generator.particles import ClusterParticles
from cluster_generator.utils import mylog


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
            f"Merger_Coll_VelX{i + 1}\t\t{vel[0]}\t# X-velocity of cluster {i + 1} in km/s",
            f"Merger_Coll_VelY{i + 1}\t\t{vel[1]}\t# Y-velocity of cluster {i + 1} in km/s",
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


def setup_arepo_ics(ics, boxsize, nx, ic_file, overwrite=False, regenerate_particles=False, prng=None):
    fields = {}
    cparts = ics.setup_particle_ics(regenerate_particles=regenerate_particles, prng=prng)
    ngrid = nx**3
    dx = boxsize / nx
    le = 0.5 * dx
    re = boxsize - 0.5 * dx
    posg = np.mgrid[le : re : nx * 1j, le : re : nx * 1j, le : re : nx * 1j].reshape(3, ngrid).T
    rmax2 = ics.r_max**2
    idxs = np.sum((posg - ics.center[0].v) ** 2, axis=1) > rmax2[0]
    if ics.num_halos > 1:
        idxs |= np.sum((posg - ics.center[1].v) ** 2, axis=1) > rmax2[1]
    if ics.num_halos > 2:
        idxs |= np.sum((posg - ics.center[2].v) ** 2, axis=1) > rmax2[2]
    dV = dx**3
    nleft = idxs.sum()
    idens = np.argmin(cparts["gas", "density"].d)
    dens = cparts["gas", "density"].d[idens] * np.ones(nleft)
    eint = cparts["gas", "thermal_energy"].d[idens] * np.ones(nleft)
    fields["gas", "particle_position"] = unyt_array(posg[idxs, :], "kpc")
    fields["gas", "particle_velocity"] = unyt_array(np.zeros((nleft, 3)), "kpc/Myr")
    fields["gas", "particle_mass"] = unyt_array(dens * dV, "Msun")
    fields["gas", "density"] = unyt_array(dens, "Msun/kpc**3")
    fields["gas", "thermal_energy"] = unyt_array(eint, "kpc**2/Myr**2")
    mylog.info(
        "Background cell density is %g g/cm**3.",
        fields["gas", "density"][0].to_value("g/cm**3"),
    )
    mylog.info(
        "Background cell mass is %g Msun.",
        fields["gas", "particle_mass"][0].to_value("Msun"),
    )
    all_parts = cparts + ClusterParticles.from_fields(fields)
    all_parts.write_to_gadget_file(ic_file, boxsize, overwrite=overwrite, code="arepo")


def resample_arepo_ics(ics, infile, outfile, overwrite=False):
    parts = ClusterParticles.from_gadget_file(infile)
    new_parts = ics.resample_particle_ics(parts)
    with h5py.File(infile, "r") as f:
        boxsize = f["Header"].attrs["BoxSize"]
    new_parts.write_to_gadget_file(outfile, boxsize, overwrite=overwrite)


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
