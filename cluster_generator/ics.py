import os
from collections.abc import Collection
from numbers import Number

import numpy as np
from pathlib import Path
from ruamel.yaml import YAML
from unyt import unyt_array

from cluster_generator.model import ClusterModel
from cluster_generator.particles import (
    ClusterParticles,
    combine_three_clusters,
    combine_two_clusters,
    resample_one_cluster,
    resample_three_clusters,
    resample_two_clusters,
)
from cluster_generator.utils import ensure_list, ensure_ytarray, parse_prng


def compute_centers_for_binary(center, d, b, a=0.0):
    """
    Given a common center and distance parameters, calculate the
    central positions of two clusters.

    First, the separation along the x-direction is determined
    by:

    sep_x = sqrt(d**2-b**2-a**2)

    where d is the distance between the two clusters, b is the
    impact parameter in the y-direction, and a is the impact
    parameter in the z-direction. So the resulting centers are
    calculated as:

    center1 = [center-0.5*sep_x, center-0.5*b, center-0.5*a]
    center2 = [center+0.5*sep_x, center+0.5*b, center+0.5*a]

    Parameters
    ----------
    center : array-like
        The center from which the distance parameters for
        the two clusters will be calculated.
    d : float
        The distance between the two clusters.
    b : float
        The impact parameter in the y-direction, in kpc.
    a : float, optional
        The impact parameter in the z-direction, in kpc.
        Default: 0.0
    """
    d = np.sqrt(d * d - b * b - a * a)
    diff = np.array([d, b, a])
    center1 = center - 0.5 * diff
    center2 = center + 0.5 * diff
    return center1, center2


class ClusterICs:
    def __init__(
        self,
        basename,
        num_halos,
        profiles,
        center,
        velocity,
        num_particles=None,
        mag_file=None,
        particle_files=None,
        r_max=20000.0,
        r_max_tracer=None,
    ):
        self.basename = basename
        self.num_halos = num_halos
        self.profiles = ensure_list(profiles)
        self.center = ensure_ytarray(center, "kpc")
        self.velocity = ensure_ytarray(velocity, "kpc/Myr")
        if self.num_halos == 1:
            self.center = self.center.reshape(1, 3)
            self.velocity = self.velocity.reshape(1, 3)
        self.mag_file = mag_file
        if isinstance(r_max, Number):
            r_max = [r_max] * num_halos
        self.r_max = np.array(r_max)
        if r_max_tracer is None:
            r_max_tracer = r_max
        if isinstance(r_max_tracer, Number):
            r_max_tracer = [r_max_tracer] * num_halos
        self.r_max_tracer = np.array(r_max_tracer)
        if num_particles is None:
            self.tot_np = {"dm": 0, "gas": 0, "star": 0, "tracer": 0}
        else:
            self.tot_np = num_particles
        self._determine_num_particles()
        self.particle_files = [None] * 3
        if particle_files is not None:
            self.particle_files[:num_halos] = particle_files[:]

    def _determine_num_particles(self):
        from collections import defaultdict

        dm_masses = []
        gas_masses = []
        star_masses = []
        tracer_masses = []
        for i, pf in enumerate(self.profiles):
            p = ClusterModel.from_h5_file(pf)
            idxs = p["radius"] < self.r_max[i]
            dm_masses.append(p["dark_matter_mass"][idxs][-1].value)
            if "gas_mass" in p:
                gmass = p["gas_mass"][idxs][-1].value
            else:
                gmass = 0.0
            gas_masses.append(gmass)
            if "stellar_mass" in p:
                smass = p["stellar_mass"][idxs][-1].value
            else:
                smass = 0.0
            star_masses.append(smass)
            if self.tot_np.get("tracer", 0) > 0:
                idxst = p["radius"] < self.r_max_tracer[i]
                tmass = p["gas_mass"][idxst][-1].value
            else:
                tmass = 0.0
            tracer_masses.append(tmass)
        tot_dm_mass = np.sum(dm_masses)
        tot_gas_mass = np.sum(gas_masses)
        tot_star_mass = np.sum(star_masses)
        tot_tracer_mass = np.sum(tracer_masses)
        self.num_particles = defaultdict(list)
        for i in range(self.num_halos):
            if self.tot_np.get("dm", 0) > 0:
                ndp = np.rint(self.tot_np["dm"] * dm_masses[i] / tot_dm_mass).astype("int")
            else:
                ndp = 0
            self.num_particles["dm"].append(ndp)
            if self.tot_np.get("gas", 0) > 0:
                ngp = np.rint(self.tot_np["gas"] * gas_masses[i] / tot_gas_mass).astype("int")
            else:
                ngp = 0
            self.num_particles["gas"].append(ngp)
            if self.tot_np.get("star", 0) > 0:
                nsp = np.rint(self.tot_np["star"] * star_masses[i] / tot_star_mass).astype("int")
            else:
                nsp = 0
            self.num_particles["star"].append(nsp)
            if self.tot_np.get("tracer", 0) > 0:
                ntp = np.rint(self.tot_np["tracer"] * tracer_masses[i] / tot_tracer_mass).astype("int")
            else:
                ntp = 0
            self.num_particles["tracer"].append(ntp)

    def _generate_particles(self, regenerate_particles=False, prng=None):
        prng = parse_prng(prng)
        parts = []
        for i, pf in enumerate(self.profiles):
            if regenerate_particles or self.particle_files[i] is None:
                m = ClusterModel.from_h5_file(pf)
                p = m.generate_dm_particles(self.num_particles["dm"][i], r_max=self.r_max[i], prng=prng)
                if self.num_particles["star"][i] > 0:
                    sp = m.generate_star_particles(
                        self.num_particles["star"][i], r_max=self.r_max[i], prng=prng
                    )
                    p = p + sp
                if self.num_particles["gas"][i] > 0:
                    gp = m.generate_gas_particles(
                        self.num_particles["gas"][i], r_max=self.r_max[i], prng=prng
                    )
                    p = p + gp
                if self.num_particles["tracer"][i] > 0:
                    tp = m.generate_tracer_particles(
                        self.num_particles["tracer"][i],
                        r_max=self.r_max_tracer[i],
                        prng=prng,
                    )
                    p = p + tp
                parts.append(p)
                outfile = f"{self.basename}_{i}_particles.h5"
                p.write_particles(outfile, overwrite=True)
                self.particle_files[i] = outfile
            else:
                p = ClusterParticles.from_file(self.particle_files[i])
                parts.append(p)
        return parts

    def to_file(self, filename, overwrite=False):
        r"""
        Write the initial conditions information to a file.

        Parameters
        ----------
        filename : string
            The file to write the initial conditions information to.
        overwrite : boolean, optional
            If True, overwrite a file with the same name. Default: False
        """
        if os.path.exists(filename) and not overwrite:
            raise RuntimeError(f"{filename} exists and overwrite=False!")
        from ruamel.yaml.comments import CommentedMap

        out = CommentedMap()
        out["basename"] = self.basename
        out.yaml_add_eol_comment("base name for ICs", key="basename")
        out["num_halos"] = self.num_halos
        out.yaml_add_eol_comment("number of halos", key="num_halos")
        out["profile1"] = self.profiles[0]
        out.yaml_add_eol_comment("profile for cluster 1", key="profile1")
        out["center1"] = self.center[0].tolist()
        out.yaml_add_eol_comment("center for cluster 1", key="center1")
        out["velocity1"] = self.velocity[0].tolist()
        out.yaml_add_eol_comment("velocity for cluster 1", key="velocity1")
        if self.particle_files[0] is not None:
            out["particle_file1"] = self.particle_files[0]
            out.yaml_add_eol_comment("particle file for cluster 1", key="particle_file1")
        if self.num_halos > 1:
            out["profile2"] = self.profiles[1]
            out.yaml_add_eol_comment("profile for cluster 2", key="profile2")
            out["center2"] = self.center[1].tolist()
            out.yaml_add_eol_comment("center for cluster 2", key="center2")
            out["velocity2"] = self.velocity[1].tolist()
            out.yaml_add_eol_comment("velocity for cluster 2", key="velocity2")
            if self.particle_files[1] is not None:
                out["particle_file2"] = self.particle_files[1]
                out.yaml_add_eol_comment("particle file for cluster 2", key="particle_file2")
        if self.num_halos == 3:
            out["profile3"] = self.profiles[2]
            out.yaml_add_eol_comment("profile for cluster 3", key="profile3")
            out["center3"] = self.center[2].tolist()
            out.yaml_add_eol_comment("center for cluster 3", key="center3")
            out["velocity3"] = self.velocity[2].tolist()
            out.yaml_add_eol_comment("velocity for cluster 3", key="velocity3")
            if self.particle_files[2] is not None:
                out["particle_file3"] = self.particle_files[2]
                out.yaml_add_eol_comment("particle file for cluster 3", key="particle_file3")
        if self.tot_np.get("dm", 0) > 0:
            out["num_dm_particles"] = self.tot_np["dm"]
            out.yaml_add_eol_comment("number of DM particles", key="num_dm_particles")
        if self.tot_np.get("gas", 0) > 0:
            out["num_gas_particles"] = self.tot_np["gas"]
            out.yaml_add_eol_comment("number of gas particles", key="num_gas_particles")
        if self.tot_np.get("star", 0) > 0:
            out["num_star_particles"] = self.tot_np["star"]
            out.yaml_add_eol_comment("number of star particles", key="num_star_particles")
        if self.tot_np.get("tracer", 0) > 0:
            out["num_tracer_particles"] = self.tot_np["tracer"]
            out.yaml_add_eol_comment("number of tracer particles", key="num_tracer_particles")
        if self.mag_file is not None:
            out["mag_file"] = self.mag_file
            out.yaml_add_eol_comment("3D magnetic field file", key="mag_file")
        out["r_max"] = self.r_max.tolist()
        out.yaml_add_eol_comment("Maximum radii of particles", key="r_max")
        if self.tot_np.get("tracer", 0) > 0:
            out["r_max_tracer"] = self.r_max_tracer.tolist()
            out.yaml_add_eol_comment("Maximum radii of tracer particles", key="r_max")
        yaml = YAML()
        with open(filename, "w") as f:
            yaml.dump(out, f)

    @classmethod
    def from_file(cls, filename):
        r"""
        Read the initial conditions information
        from a YAML-formatted `filename`.
        """
        from ruamel.yaml import YAML

        yaml = YAML()
        with open(filename) as f:
            params = yaml.load(f)
        basename = params["basename"]
        num_halos = params["num_halos"]
        profiles = [params[f"profile{i}"] for i in range(1, num_halos + 1)]
        center = [np.array(params[f"center{i}"]) for i in range(1, num_halos + 1)]
        velocity = [np.array(params[f"velocity{i}"]) for i in range(1, num_halos + 1)]
        num_particles = {k: params.get(f"num_{k}_particles", 0) for k in ["gas", "dm", "star"]}
        mag_file = params.get("mag_file", None)
        particle_files = [params.get(f"particle_file{i}", None) for i in range(1, num_halos + 1)]
        r_max = params.get("r_max", 20000.0)
        r_max_tracer = params.get("r_max_tracer", r_max)
        return cls(
            basename,
            num_halos,
            profiles,
            center,
            velocity,
            num_particles=num_particles,
            mag_file=mag_file,
            particle_files=particle_files,
            r_max=r_max,
            r_max_tracer=r_max_tracer,
        )

    def setup_particle_ics(self, regenerate_particles=False, prng=None):
        r"""
        From a set of cluster models and their relative positions and
        velocities, set up initial conditions for use with SPH codes.

        This routine will either generate a single cluster or will combine
        two or three clusters together. If more than one cluster is
        generated, the gas particles will have their densities set by
        adding the densities from the overlap of the two particles
        together, and will have their thermal energies and velocities
        set by mass-weighting them from the two profiles.

        Parameters
        ----------
        """
        profiles = [ClusterModel.from_h5_file(hf) for hf in self.profiles]
        parts = self._generate_particles(regenerate_particles=regenerate_particles, prng=prng)
        if self.num_halos == 1:
            all_parts = parts[0]
            all_parts.add_offsets(self.center[0], self.velocity[0])
        elif self.num_halos == 2:
            all_parts = combine_two_clusters(
                parts[0],
                parts[1],
                profiles[0],
                profiles[1],
                self.center[0],
                self.center[1],
                self.velocity[0],
                self.velocity[1],
            )
        else:
            all_parts = combine_three_clusters(
                parts[0],
                parts[1],
                parts[2],
                profiles[0],
                profiles[1],
                profiles[2],
                self.center[0],
                self.center[1],
                self.center[2],
                self.velocity[0],
                self.velocity[1],
                self.velocity[2],
            )
        return all_parts

    def resample_particle_ics(self, parts, passive_scalars=None):
        r"""
        Given a Gadget-HDF5-like initial conditions file which has been
        output from some type of relaxation process (such as making a
        glass or using MESHRELAX in the case of Arepo), resample the density,
        thermal energy, and velocity fields onto the gas particles/cells from
        the initial hydrostatic profiles.

        Parameters
        ----------
        filename : string
            The name of file to output the resampled ICs to.
        """
        profiles = [ClusterModel.from_h5_file(hf) for hf in self.profiles]
        if self.num_halos == 1:
            new_parts = resample_one_cluster(
                parts,
                profiles[0],
                self.center[0],
                self.velocity[0],
                passive_scalars=passive_scalars,
            )
        elif self.num_halos == 2:
            new_parts = resample_two_clusters(
                parts,
                profiles[0],
                profiles[1],
                self.center[0],
                self.center[1],
                self.velocity[0],
                self.velocity[1],
                self.r_max,
                passive_scalars=passive_scalars,
            )
        else:
            new_parts = resample_three_clusters(
                parts,
                profiles[0],
                profiles[1],
                profiles[2],
                self.center[0],
                self.center[1],
                self.center[2],
                self.velocity[0],
                self.velocity[1],
                self.velocity[2],
                self.r_max,
                passive_scalars=passive_scalars,
            )
        return new_parts

    def create_dataset(
        self,
        filename: str | Path,
        domain_dimensions: Collection[int] = (512, 512, 512),
        left_edge: Collection[Number] | unyt_array | None = None,
        box_size: Collection[Number] | unyt_array | None = None,
        overwrite: bool = False,
        chunksize: int = 64,
    ) -> str | Path:
        r"""
        Construct a ``yt`` dataset object from this model on a uniformly spaced grid.

        Parameters
        ----------
        filename : str or :py:class:`pathlib.Path`
            The path at which to generate the underlying HDF5 datafile.
        domain_dimensions : Collection of int, optional
            The size of the uniform grid along each axis of the domain.
            If specified, the argument must be an iterable type with
            shape ``(3,)``. Each element should be an ``int`` specifying
            the number of grid cells to place along that axis. By default,
            the selected value is ``(512,512,512)``.
        left_edge : Collection of float or :py:class:`unyt.unyt_array`, optional
            The left-most edge of the uniform grid's domain. In conjunction
            with ``box_size``, this attribute specifies the position of
            the model in the box and the amount of the model which is
            actually written to the disk. If specified, ``left_edge`` should
            be a length 3 iterable with each of the entries representing the
            minimum value of the respective axis. If elements of the iterable
            have units, or the array is a :py:class:`unyt.unyt_array` instance,
            then the units will be interpreted automatically; otherwise, units
            are assumed to be kpc. By default, the left edge is determined such
            that the resulting grid contains the full radial domain of the
            :py:class:`ClusterModel`.
        box_size : Collection of float or :py:class:`unyt.unyt_array`, optional
            The length of the grid along each of the physical axes. Along with
            ``left_edge``, this argument determines the positioning of the grid
            and the model within it. If specified, ``box_size`` should be a length
            3 iterable with each of the entries representing the length of the
            grid along the respective axis. If elements of the iterable have units,
            or the array is a :py:class:`unyt.unyt_array` instance, then the units
            will be interpreted automatically; otherwise, units are assumed to be kpc.
            By default, the ``box_size`` is determined such that the resulting grid
            contains the full radial domain of the :py:class:`ClusterModel`.
        overwrite : bool, optional
            If ``False`` (default), the an error is raised if ``filename`` already
            exists. Otherwise, ``filename`` will be deleted and overwritten by this
            method.
        chunksize : int, optional
            The maximum chunksize for subgrid operations. Lower values with increase
            the execution time but save memory. By default, chunks contain no more
            than :math:`64^3` cells (``chunksize=64``).

        Returns
        -------
        str
            The path to the output dataset file.

        Notes
        -----

        Generically, converting a :py:class:`ClusterModel` instance to a valid
        ``yt`` dataset occurs in two steps. In the first step, the dataset is written
        to disk on a uniform grid (or, more generally, an AMR grid). From this grid,
        ``yt`` can then interpret the data and construct a dataset from there.

        Because constructing the underlying grid is a memory intensive procedure, this
        method utilizes the HDF5 structure as an intermediary (effectively using the
        disk for VRAM).

        """
        from cluster_generator.data_structures import YTHDF5

        if left_edge is None:
            left_edge = 3 * [-np.amax(self.r_max)]
        if box_size is None:
            box_size = 2 * np.amax(self.r_max)

        bbox = [[le, le + box_size] for le in left_edge]

        ds_obj = YTHDF5.build(
            filename,
            domain_dimensions,
            bbox,
            chunksize=chunksize,
            overwrite=overwrite,
        )
        ds_obj.add_ICs(self)

        return ds_obj.filename
