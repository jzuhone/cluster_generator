import os
from collections import OrderedDict

import h5py
import numpy as np
from scipy.integrate import cumtrapz, quad
from scipy.interpolate import InterpolatedUnivariateSpline
from unyt import unyt_array, unyt_quantity

from cluster_generator.particles import ClusterParticles
from cluster_generator.utils import (
    G,
    ensure_ytquantity,
    field_label_map,
    generate_particle_radii,
    integrate,
    integrate_mass,
    kpc_to_cm,
    mp,
    mu,
    mue,
    mylog,
)
from cluster_generator.virial import VirialEquilibrium

tt = 2.0 / 3.0
mtt = -tt
ft = 5.0 / 3.0
tf = 3.0 / 5.0
mtf = -tf
gamma = ft
et = 8.0 / 3.0
te = 3.0 / 8.0


class ClusterModel:
    default_fields = [
        "density",
        "temperature",
        "pressure",
        "total_density",
        "gravitational_potential",
        "gravitational_field",
        "total_mass",
        "gas_mass",
        "dark_matter_mass",
        "dark_matter_density",
        "stellar_density",
        "stellar_mass",
    ]

    _keep_units = ["entropy", "electron_number_density", "magnetic_field_strength"]

    def __init__(self, num_elements, fields):
        self.num_elements = num_elements
        self.fields = fields

    _dm_virial = None
    _star_virial = None

    @property
    def dm_virial(self):
        if self._dm_virial is None:
            self._dm_virial = VirialEquilibrium(self, "dark_matter")
        return self._dm_virial

    @property
    def star_virial(self):
        if self._star_virial is None and "stellar_density" in self:
            self._star_virial = VirialEquilibrium(self, "stellar")
        return self._star_virial

    @classmethod
    def from_arrays(cls, fields):
        return ClusterModel(fields["radius"].size, fields)

    @classmethod
    def from_h5_file(cls, filename, r_min=None, r_max=None):
        r"""
        Generate an equilibrium model from an HDF5 file.

        Parameters
        ----------
        filename : string
            The name of the file to read the model from.

        Examples
        --------
        >>> from cluster_generator import ClusterModel
        >>> hse_model = ClusterModel.from_h5_file("hse_model.h5")
        """
        from cluster_generator.virial import VirialEquilibrium

        with h5py.File(filename, "r") as f:
            fnames = list(f["fields"].keys())
            get_dm_virial = "dm_df" in f
            get_star_virial = "star_df" in f

        fields = OrderedDict()
        for field in fnames:
            a = unyt_array.from_hdf5(filename, dataset_name=field, group_name="fields")
            fields[field] = unyt_array(a.d, str(a.units))
            if field not in cls._keep_units:
                fields[field].convert_to_base("galactic")
        if r_min is None:
            r_min = 0.0
        if r_max is None:
            r_max = fields["radius"][-1].d * 2
        mask = np.logical_and(fields["radius"].d >= r_min, fields["radius"].d <= r_max)
        for field in fnames:
            fields[field] = fields[field][mask]
        num_elements = mask.sum()

        model = cls(num_elements, fields)

        if get_dm_virial:
            mask = np.logical_and(
                fields["radius"].d >= r_min, fields["radius"].d <= r_max
            )
            df = unyt_array.from_hdf5(filename, dataset_name="dm_df")[mask]
            model._dm_virial = VirialEquilibrium(model, ptype="dark_matter", df=df)

        if get_star_virial:
            mask = np.logical_and(
                fields["radius"].d >= r_min, fields["radius"].d <= r_max
            )
            df = unyt_array.from_hdf5(filename, dataset_name="star_df")[mask]
            model._star_virial = VirialEquilibrium(model, ptype="stellar", df=df)

        return model

    @classmethod
    def _from_scratch(cls, fields, stellar_density=None):
        rr = fields["radius"].d
        mylog.info("Integrating gravitational potential profile.")
        tdens_func = InterpolatedUnivariateSpline(rr, fields["total_density"].d)
        gpot_profile = lambda r: tdens_func(r) * r
        gpot1 = fields["total_mass"] / fields["radius"]
        gpot2 = unyt_array(4.0 * np.pi * integrate(gpot_profile, rr), "Msun/kpc")
        fields["gravitational_potential"] = -G * (gpot1 + gpot2)
        fields["gravitational_potential"].convert_to_units("kpc**2/Myr**2")

        if "density" in fields and "gas_mass" not in fields:
            mylog.info("Integrating gas mass profile.")
            m0 = fields["density"].d[0] * rr[0] ** 3 / 3.0
            fields["gas_mass"] = unyt_array(
                4.0 * np.pi * cumtrapz(fields["density"] * rr * rr, x=rr, initial=0.0)
                + m0,
                "Msun",
            )

        if stellar_density is not None:
            fields["stellar_density"] = unyt_array(stellar_density(rr), "Msun/kpc**3")
            mylog.info("Integrating stellar mass profile.")
            fields["stellar_mass"] = unyt_array(
                integrate_mass(stellar_density, rr), "Msun"
            )

        mdm = fields["total_mass"].copy()
        ddm = fields["total_density"].copy()
        if "density" in fields:
            mdm -= fields["gas_mass"]
            ddm -= fields["density"]
        if "stellar_mass" in fields:
            mdm -= fields["stellar_mass"]
            ddm -= fields["stellar_density"]
        mdm[ddm.v < 0.0] = mdm.max()
        ddm[ddm.v < 0.0] = 0.0

        if ddm.sum() < 0.0 or mdm.sum() < 0.0:
            mylog.warning("The total dark matter mass is either zero or negative!!")
        fields["dark_matter_density"] = ddm
        fields["dark_matter_mass"] = mdm

        if "density" in fields:
            fields["gas_fraction"] = fields["gas_mass"] / fields["total_mass"]
            fields["electron_number_density"] = fields["density"].to(
                "cm**-3", "number_density", mu=mue
            )
            fields["entropy"] = (
                fields["temperature"] * fields["electron_number_density"] ** mtt
            )

        return cls(rr.size, fields)

    def set_rmax(self, r_max):
        mask = self.fields["radius"].d <= r_max
        fields = {}
        for field in self.fields:
            fields[field] = self.fields[field][mask]
        num_elements = mask.sum()
        return ClusterModel(
            num_elements, fields, dm_virial=self.dm_virial, star_virial=self.star_virial
        )

    def __getitem__(self, key):
        return self.fields[key]

    def __contains__(self, key):
        return key in self.fields

    def keys(self):
        return self.fields.keys()

    def write_model_to_ascii(self, output_filename, in_cgs=False, overwrite=False):
        r"""
        Write the equilibrium model to an ascii text file. Uses
        AstroPy's `QTable` to write the file, so that units are
        included.

        Parameters
        ----------
        output_filename : string
            The file to write the model to.
        in_cgs : boolean, optional
            Whether to convert the units to cgs before writing. Default: False.
        overwrite : boolean, optional
            Overwrite an existing file with the same name. Default: False.
        """
        from astropy.table import QTable

        fields = {}
        for k, v in self.fields.items():
            if in_cgs:
                if k == "temperature":
                    fd = v.to_equivalent("K", "thermal")
                elif k not in self._keep_units:
                    fd = v.in_cgs()
                else:
                    fd = v
            else:
                fd = v
            fields[k] = fd.to_astropy()
        t = QTable(fields)
        t.meta["comments"] = f"unit_system={'cgs' if in_cgs else 'galactic'}"
        t.write(output_filename, overwrite=overwrite)

    def write_model_to_h5(
        self, output_filename, in_cgs=False, r_min=None, r_max=None, overwrite=False
    ):
        r"""
        Write the equilibrium model to an HDF5 file.

        Parameters
        ----------
        output_filename : string
            The file to write the model to.
        in_cgs : boolean, optional
            Whether to convert the units to cgs before writing. Default False.
        overwrite : boolean, optional
            Overwrite an existing file with the same name. Default False.
        """
        if os.path.exists(output_filename) and not overwrite:
            raise IOError(
                f"Cannot create {output_filename}. It exists and overwrite=False."
            )
        f = h5py.File(output_filename, "w")
        f.create_dataset("num_elements", data=self.num_elements)
        f.attrs["unit_system"] = "cgs" if in_cgs else "galactic"
        f.close()
        if r_min is None:
            r_min = 0.0
        if r_max is None:
            r_max = self.fields["radius"][-1].d * 2
        mask = np.logical_and(
            self.fields["radius"].d >= r_min, self.fields["radius"].d <= r_max
        )
        for k, v in self.fields.items():
            if in_cgs:
                if k == "temperature":
                    fd = v[mask].to_equivalent("K", "thermal")
                elif k not in self._keep_units:
                    fd = v[mask].in_cgs()
                else:
                    fd = v[mask]
            else:
                fd = v[mask]
            fd.write_hdf5(output_filename, dataset_name=k, group_name="fields")
        if getattr(self, "_dm_virial", None):
            fd = self.dm_virial.df
            fd.write_hdf5(output_filename, dataset_name="dm_df")
        if getattr(self, "_star_virial", None):
            fd = self.star_virial.df
            fd.write_hdf5(output_filename, dataset_name="star_df")

    def write_model_to_binary(
        self,
        output_filename,
        fields_to_write=None,
        in_cgs=False,
        r_min=None,
        r_max=None,
        overwrite=False,
    ):
        if fields_to_write is None:
            fields_to_write = list(self.fields.keys())
        from scipy.io import FortranFile

        if os.path.exists(output_filename) and not overwrite:
            raise IOError(
                f"Cannot create {output_filename}. It exists and " f"overwrite=False."
            )
        if r_min is None:
            r_min = 0.0
        if r_max is None:
            r_max = self.fields["radius"][-1].d * 2
        mask = np.logical_and(
            self.fields["radius"].d >= r_min, self.fields["radius"].d <= r_max
        )
        with FortranFile(output_filename, "w") as f:
            f.write_record(self.fields["radius"][mask].size)
            prof_rec = []
            for k in fields_to_write:
                v = self.fields[k]
                if in_cgs:
                    if k == "temperature":
                        fd = v[mask].to_equivalent("K", "thermal")
                    elif k not in self._keep_units:
                        fd = v[mask].in_cgs()
                    else:
                        fd = v[mask]
                else:
                    fd = v[mask]
                prof_rec.append(fd)
            f.write_record(np.array(prof_rec).T)

    def set_field(self, name, value):
        r"""
        Set a field with name `name` to value `value`, which is an `unyt_array`.
        The array will be checked to make sure that it has the appropriate size.
        """
        if not isinstance(value, unyt_array):
            raise TypeError("value needs to be an unyt_array")
        if value.size == self.num_elements:
            if name in self.fields:
                mylog.warning("Overwriting field %s." % name)
            self.fields[name] = value
        else:
            raise ValueError(
                f"The length of the array needs to be " f"{self.num_elements} elements!"
            )

    @classmethod
    def from_dens_and_temp(
        cls, rmin, rmax, density, temperature, stellar_density=None, num_points=1000
    ):
        """
        Construct a hydrostatic equilibrium model using gas density
        and temperature profiles.

        Parameters
        ----------
        rmin : float
            Minimum radius of profiles in kpc.
        rmax : float
            Maximum radius of profiles in kpc.
        density : :class:`~cluster_generator.radial_profiles.RadialProfile`
            A radial profile describing the gas mass density.
        temperature : :class:`~cluster_generator.radial_profiles.RadialProfile`
            A radial profile describing the gas temperature.
        stellar_density : :class:`~cluster_generator.radial_profiles.RadialProfile`, optional
            A radial profile describing the stellar mass density, if desired.
        num_points : integer, optional
            The number of points the profiles are evaluated at.
        """
        mylog.info("Computing the profiles from density and temperature.")
        rr = np.logspace(np.log10(rmin), np.log10(rmax), num_points, endpoint=True)
        fields = OrderedDict()
        fields["radius"] = unyt_array(rr, "kpc")
        fields["density"] = unyt_array(density(rr), "Msun/kpc**3")
        fields["temperature"] = unyt_array(temperature(rr), "keV")
        fields["pressure"] = fields["density"] * fields["temperature"]
        fields["pressure"] /= mu * mp
        fields["pressure"].convert_to_units("Msun/(Myr**2*kpc)")
        pressure_spline = InterpolatedUnivariateSpline(rr, fields["pressure"].d)
        dPdr = unyt_array(pressure_spline(rr, 1), "Msun/(Myr**2*kpc**2)")
        fields["gravitational_field"] = dPdr / fields["density"]
        fields["gravitational_field"].convert_to_units("kpc/Myr**2")
        fields["gas_mass"] = unyt_array(integrate_mass(density, rr), "Msun")
        fields["total_mass"] = (
            -fields["radius"] ** 2 * fields["gravitational_field"] / G
        )
        total_mass_spline = InterpolatedUnivariateSpline(rr, fields["total_mass"].v)
        dMdr = unyt_array(total_mass_spline(rr, nu=1), "Msun/kpc")
        fields["total_density"] = dMdr / (4.0 * np.pi * fields["radius"] ** 2)
        return cls._from_scratch(fields, stellar_density=stellar_density)

    @classmethod
    def from_dens_and_entr(
        cls, rmin, rmax, density, entropy, stellar_density=None, num_points=1000
    ):
        n_e = density / (mue * mp * kpc_to_cm**3)
        temperature = entropy * n_e**tt
        return cls.from_dens_and_temp(
            rmin,
            rmax,
            density,
            temperature,
            stellar_density=stellar_density,
            num_points=num_points,
        )

    @classmethod
    def from_dens_and_tden(
        cls, rmin, rmax, density, total_density, stellar_density=None, num_points=1000
    ):
        """
        Construct a hydrostatic equilibrium model using gas density
        and total density profiles

        Parameters
        ----------
        rmin : float
            Minimum radius of profiles in kpc.
        rmax : float
            Maximum radius of profiles in kpc.
        density : :class:`~cluster_generator.radial_profiles.RadialProfile`
            A radial profile describing the gas mass density.
        total_density : :class:`~cluster_generator.radial_profiles.RadialProfile`
            A radial profile describing the total mass density.
        stellar_density : :class:`~cluster_generator.radial_profiles.RadialProfile`, optional
            A radial profile describing the stellar mass density, if desired.
        num_points : integer, optional
            The number of points the profiles are evaluated at.
        """
        mylog.info("Computing the profiles from density and total density.")
        rr = np.logspace(np.log10(rmin), np.log10(rmax), num_points, endpoint=True)
        fields = OrderedDict()
        fields["radius"] = unyt_array(rr, "kpc")
        fields["density"] = unyt_array(density(rr), "Msun/kpc**3")
        fields["total_density"] = unyt_array(total_density(rr), "Msun/kpc**3")
        mylog.info("Integrating total mass profile.")
        fields["total_mass"] = unyt_array(integrate_mass(total_density, rr), "Msun")
        fields["gas_mass"] = unyt_array(integrate_mass(density, rr), "Msun")
        fields["gravitational_field"] = (
            -G * fields["total_mass"] / (fields["radius"] ** 2)
        )
        fields["gravitational_field"].convert_to_units("kpc/Myr**2")
        g = fields["gravitational_field"].in_units("kpc/Myr**2").v
        g_r = InterpolatedUnivariateSpline(rr, g)
        dPdr_int = lambda r: density(r) * g_r(r)
        mylog.info("Integrating pressure profile.")
        P = -integrate(dPdr_int, rr)
        dPdr_int2 = lambda r: density(r) * g[-1] * (rr[-1] / r) ** 2
        P -= quad(dPdr_int2, rr[-1], np.inf, limit=100)[0]
        fields["pressure"] = unyt_array(P, "Msun/kpc/Myr**2")
        fields["temperature"] = fields["pressure"] * mu * mp / fields["density"]
        fields["temperature"].convert_to_units("keV")

        return cls._from_scratch(fields, stellar_density=stellar_density)

    @classmethod
    def no_gas(cls, rmin, rmax, total_density, stellar_density=None, num_points=1000):
        rr = np.logspace(np.log10(rmin), np.log10(rmax), num_points, endpoint=True)
        fields = OrderedDict()
        fields["radius"] = unyt_array(rr, "kpc")
        fields["total_density"] = unyt_array(total_density(rr), "Msun/kpc**3")
        mylog.info("Integrating total mass profile.")
        fields["total_mass"] = unyt_array(integrate_mass(total_density, rr), "Msun")
        fields["gravitational_field"] = (
            -G * fields["total_mass"] / (fields["radius"] ** 2)
        )
        fields["gravitational_field"].convert_to_units("kpc/Myr**2")

        return cls._from_scratch(fields, stellar_density=stellar_density)

    def find_field_at_radius(self, field, r):
        """
        Find the value of a *field* in the profiles
        at radius *r*.
        """
        return unyt_array(np.interp(r, self["radius"], self[field]), self[field].units)

    def check_hse(self):
        r"""
        Determine the deviation of the model from hydrostatic equilibrium.

        Returns
        -------
        chk : NumPy array
            An array containing the relative deviation from hydrostatic
            equilibrium as a function of radius.
        """
        if "pressure" not in self.fields:
            raise RuntimeError("This ClusterModel contains no gas!")
        rr = self.fields["radius"].v
        pressure_spline = InterpolatedUnivariateSpline(rr, self.fields["pressure"].v)
        dPdx = unyt_array(pressure_spline(rr, 1), "Msun/(Myr**2*kpc**2)")
        rhog = self.fields["density"] * self.fields["gravitational_field"]
        chk = dPdx - rhog
        chk /= rhog
        mylog.info(
            "The maximum relative deviation of this profile from "
            "hydrostatic equilibrium is %g",
            np.abs(chk).max(),
        )
        return chk

    def check_dm_virial(self):
        return self.dm_virial.check_virial()

    def check_star_virial(self):
        if self._star_virial is None:
            raise RuntimeError(
                "Cannot check the virial equilibrium of "
                "the stars because there are no stars in "
                "this model!"
            )
        return self.star_virial.check_virial()

    def set_magnetic_field_from_beta(self, beta, gaussian=True):
        """
        Set a magnetic field radial profile from
        a plasma beta parameter, assuming beta = p_th/p_B.
        The field can be set in Gaussian or Lorentz-Heaviside
        (dimensionless) units.

        Parameters
        ----------
        beta : float
            The ratio of the thermal pressure to the
            magnetic pressure.
        gaussian : boolean, optional
            Set the field in Gaussian units such that
            p_B = B^2/(8*pi), otherwise p_B = B^2/2.
            Default: True
        """
        B = np.sqrt(2.0 * self["pressure"] / beta)
        if gaussian:
            B *= np.sqrt(4.0 * np.pi)
        B.convert_to_units("gauss")
        self.set_field("magnetic_field_strength", B)

    def set_magnetic_field_from_density(self, B0, eta=2.0 / 3.0, gaussian=True):
        """
        Set a magnetic field radial profile assuming it is proportional
        to some power of the density, usually 2/3. The field can be set
        in Gaussian or Lorentz-Heaviside (dimensionless) units.

        Parameters
        ----------
        B0 : float
            The central magnetic field strength in units of
            gauss.
        eta : float, optional
            The power of the density which the field is
            proportional to. Default: 2/3.
        gaussian : boolean, optional
            Set the field in Gaussian units such that
            p_B = B^2/(8*pi), otherwise p_B = B^2/2.
            Default: True
        """
        B0 = ensure_ytquantity(B0, "gauss")
        B = B0 * (self["density"] / self["density"][0]) ** eta
        if not gaussian:
            B /= np.sqrt(4.0 * np.pi)
        self.set_field("magnetic_field_strength", B)

    def generate_tracer_particles(
        self, num_particles, r_max=None, sub_sample=1, prng=None
    ):
        """
        Generate a set of tracer particles based on the gas distribution.

        Parameters
        ----------
        num_particles : integer
            The number of particles to generate.
        r_max : float, optional
            The maximum radius in kpc within which to generate
            particle positions. If not supplied, it will generate
            positions out to the maximum radius available. Default: None
        sub_sample : integer, optional
            This option allows one to generate a sub-sample of unique
            particle radii, densities, and energies which will then be
            repeated to fill the required number of particles. Default: 1,
            which means no sub-sampling.
        prng : :class:`~numpy.random.RandomState` object, integer, or None
            A pseudo-random number generator. Typically will only
            be specified if you have a reason to generate the same
            set of random numbers, such as for a test. Default is None,
            which sets the seed based on the system time.
        """
        from cluster_generator.utils import parse_prng

        prng = parse_prng(prng)
        mylog.info("We will be assigning %d tracer particles.", num_particles)
        mylog.info("Compute particle positions.")

        num_particles_sub = num_particles // sub_sample

        radius_sub, mtot = generate_particle_radii(
            self["radius"].d,
            self["gas_mass"].d,
            num_particles_sub,
            r_max=r_max,
            prng=prng,
        )

        if sub_sample > 1:
            radius = np.tile(radius_sub, sub_sample)[:num_particles]
        else:
            radius = radius_sub

        theta = np.arccos(prng.uniform(low=-1.0, high=1.0, size=num_particles))
        phi = 2.0 * np.pi * prng.uniform(size=num_particles)

        fields = OrderedDict()

        fields["tracer", "particle_position"] = unyt_array(
            [
                radius * np.sin(theta) * np.cos(phi),
                radius * np.sin(theta) * np.sin(phi),
                radius * np.cos(theta),
            ],
            "kpc",
        ).T

        fields["tracer", "particle_velocity"] = unyt_array(
            np.zeros(fields["tracer", "particle_position"].shape), "kpc/Myr"
        )

        fields["tracer", "particle_mass"] = unyt_array(np.zeros(num_particles), "Msun")

        return ClusterParticles("tracer", fields)

    def generate_gas_particles(
        self,
        num_particles,
        r_max=None,
        sub_sample=1,
        compute_potential=False,
        prng=None,
    ):
        """
        Generate a set of gas particles in hydrostatic equilibrium.

        Parameters
        ----------
        num_particles : integer
            The number of particles to generate.
        r_max : float, optional
            The maximum radius in kpc within which to generate
            particle positions. If not supplied, it will generate
            positions out to the maximum radius available. Default: None
        sub_sample : integer, optional
            This option allows one to generate a sub-sample of unique
            particle radii, densities, and energies which will then be
            repeated to fill the required number of particles. Default: 1,
            which means no sub-sampling.
        compute_potential : boolean, optional
            If True, the gravitational potential for each particle will
            be computed. Default: False
        prng : :class:`~numpy.random.RandomState` object, integer, or None
            A pseudo-random number generator. Typically will only
            be specified if you have a reason to generate the same
            set of random numbers, such as for a test. Default is None,
            which sets the seed based on the system time.
        """
        from cluster_generator.utils import parse_prng

        prng = parse_prng(prng)
        mylog.info("We will be assigning %d gas particles.", num_particles)
        mylog.info("Compute particle positions.")

        num_particles_sub = num_particles // sub_sample

        radius_sub, mtot = generate_particle_radii(
            self["radius"].d,
            self["gas_mass"].d,
            num_particles_sub,
            r_max=r_max,
            prng=prng,
        )

        if sub_sample > 1:
            radius = np.tile(radius_sub, sub_sample)[:num_particles]
        else:
            radius = radius_sub

        theta = np.arccos(prng.uniform(low=-1.0, high=1.0, size=num_particles))
        phi = 2.0 * np.pi * prng.uniform(size=num_particles)

        fields = OrderedDict()

        fields["gas", "particle_position"] = unyt_array(
            [
                radius * np.sin(theta) * np.cos(phi),
                radius * np.sin(theta) * np.sin(phi),
                radius * np.cos(theta),
            ],
            "kpc",
        ).T

        mylog.info("Compute particle thermal energies, densities, and masses.")

        e_arr = 1.5 * self.fields["pressure"] / self.fields["density"]
        get_energy = InterpolatedUnivariateSpline(self.fields["radius"], e_arr)

        if sub_sample > 1:
            energy = np.tile(get_energy(radius_sub), sub_sample)[:num_particles]
        else:
            energy = get_energy(radius)

        fields["gas", "thermal_energy"] = unyt_array(energy, "kpc**2/Myr**2")
        fields["gas", "particle_mass"] = unyt_array(
            [mtot / num_particles] * num_particles, "Msun"
        )

        get_density = InterpolatedUnivariateSpline(
            self.fields["radius"], self.fields["density"]
        )

        if sub_sample > 1:
            density = np.tile(get_density(radius_sub), sub_sample)[:num_particles]
        else:
            density = get_density(radius)

        fields["gas", "density"] = unyt_array(density, "Msun/kpc**3")

        mylog.info("Set particle velocities to zero.")

        fields["gas", "particle_velocity"] = unyt_array(
            np.zeros((num_particles, 3)), "kpc/Myr"
        )

        if compute_potential:
            energy_spline = InterpolatedUnivariateSpline(
                self["radius"].d, -self["gravitational_potential"]
            )
            phi = -energy_spline(radius_sub)
            if sub_sample > 1:
                phi = np.tile(phi, sub_sample)
            fields["gas", "particle_potential"] = unyt_array(phi, "kpc**2/Myr**2")

        return ClusterParticles("gas", fields)

    def generate_dm_particles(
        self,
        num_particles,
        r_max=None,
        sub_sample=1,
        compute_potential=False,
        prng=None,
    ):
        """
        Generate a set of dark matter particles in virial equilibrium.

        Parameters
        ----------
        num_particles : integer
            The number of particles to generate.
        r_max : float, optional
            The maximum radius in kpc within which to generate
            particle positions. If not supplied, it will generate
            positions out to the maximum radius available. Default: None
        sub_sample : integer, optional
            This option allows one to generate a sub-sample of unique
            particle radii and velocities which will then be repeated
            to fill the required number of particles. Default: 1, which
            means no sub-sampling.
        compute_potential : boolean, optional
            If True, the gravitational potential for each particle will
            be computed. Default: False
        prng : :class:`~numpy.random.RandomState` object, integer, or None
            A pseudo-random number generator. Typically will only
            be specified if you have a reason to generate the same
            set of random numbers, such as for a test. Default is None,
            which sets the seed based on the system time.

        Returns
        -------
        particles : :class:`~cluster_generator.particles.ClusterParticles`
            A set of dark matter particles.
        """
        return self.dm_virial.generate_particles(
            num_particles,
            r_max=r_max,
            sub_sample=sub_sample,
            compute_potential=compute_potential,
            prng=prng,
        )

    def generate_star_particles(
        self,
        num_particles,
        r_max=None,
        sub_sample=1,
        compute_potential=False,
        prng=None,
    ):
        """
        Generate a set of star particles in virial equilibrium.

        Parameters
        ----------
        num_particles : integer
            The number of particles to generate.
        r_max : float, optional
            The maximum radius in kpc within which to generate
            particle positions. If not supplied, it will generate
            positions out to the maximum radius available. Default: None
        sub_sample : integer, optional
            This option allows one to generate a sub-sample of unique
            particle radii and velocities which will then be repeated
            to fill the required number of particles. Default: 1, which
            means no sub-sampling.
        compute_potential : boolean, optional
            If True, the gravitational potential for each particle will
            be computed. Default: False
        prng : :class:`~numpy.random.RandomState` object, integer, or None
            A pseudo-random number generator. Typically will only
            be specified if you have a reason to generate the same
            set of random numbers, such as for a test. Default is None,
            which sets the seed based on the system time.

        Returns
        -------
        particles : :class:`~cluster_generator.particles.ClusterParticles`
            A set of star particles.
        """
        return self.star_virial.generate_particles(
            num_particles,
            r_max=r_max,
            sub_sample=sub_sample,
            compute_potential=compute_potential,
            prng=prng,
        )

    def plot(self, field, r_min=None, r_max=None, fig=None, ax=None, **kwargs):
        """
        Plot a field vs radius from this model using Matplotlib.

        Parameters
        ----------
        field : string
            The field to plot.
        r_min : float
            The minimum radius of the plot in kpc.
        r_max : float
            The maximum radius of the plot in kpc.
        fig : Matplotlib Figure
            The figure to plot in. Default; None, in which case
            one will be generated.
        ax : Matplotlib Axes
            The axes to plot in. Default: None, in which case
            one will be generated.

        Returns
        -------
        The Figure, Axes tuple used for the plot.
        """
        import matplotlib.pyplot as plt

        plt.rc("font", size=18)
        plt.rc("axes", linewidth=2)
        if fig is None:
            fig = plt.figure(figsize=(10, 10))
        if ax is None:
            ax = fig.add_subplot(111)
        ax.loglog(self["radius"], self[field], **kwargs)
        ax.set_xlim(r_min, r_max)
        ax.set_xlabel("Radius (kpc)")
        ax.set_ylabel(field_label_map.get(field, ""))
        ax.tick_params(which="major", width=2, length=6)
        ax.tick_params(which="minor", width=2, length=3)
        return fig, ax

    def mass_in_radius(self, radius):
        masses = {}
        r = self.fields["radius"].to_value("kpc")
        for mtype in ["total", "gas", "dark_matter", "stellar"]:
            if f"{mtype}_mass" in self.fields:
                masses[mtype] = self.fields[f"{mtype}_mass"][r < radius][-1]
        return masses

    def find_radius_for_density(self, density):
        density = ensure_ytquantity(density, "Msun/kpc**3").value
        r = self.fields["radius"].to_value("kpc")[::-1]
        d = self.fields["density"].to_value("Msun/kpc**3")[::-1]
        return unyt_quantity(np.interp(density, d, r), "kpc")

    def create_dateset(
        self,
        filename,
        domain_dimensions=(512, 512, 512),
        left_edge=None,
        box_size=None,
        overwrite=False,
        chunksize=64,
    ):
        r"""
        Generate a :py:mod:`yt` dataset grid for the :py:class:`model.ClusterModel` instance.

        Parameters
        ----------
        filename: str
            The filename at which to write the HDF5 formatted grid dataset. By default, this is ``None``, and a
            temporary directory is generated which is deleted after runtime has concluded.
        domain_dimensions: tuple, optional
            Length 3 tuple of integers. Represents the number of cells to place on each of the axes of the
            grid. By default, the grid is :math:`512^3` cells.
        left_edge: tuple, optional
            The left edge of the grid. By default, this is :math:`-r_{\mathrm{max}}` (for each entry), so as to contain the entire
            model within the grid domain. In conjunction with ``box_size``, this serves to determine the geometry of the
            output dataset.
        box_size: tuple, optional
            The size of each of the box axes. By default, these are each twice the maximum radius of the model to ensure that
            the entire model is included in the dataset.
        overwrite: bool, optional
            If ``True``, the dataset creation process will attempt to overwrite an existing data file if necessary. Default is ``False``.
        chunksize: int, optional
            The maximum chunksize for subgrid operations. Lower values with increase the execution time but save memory. By default,
            chunks contain no more that :math:`64^3` cells (``chunksize=64``).

        Returns
        -------
        yt_dataset
        """
        from data_structures import YTHDF5

        if not left_edge:
            left_edge = 3 * [np.amin(self["radius"].to_value("kpc"))]
        if not box_size:
            box_size = 2 * np.amax(self["radius"].to_value("kpc"))

        bbox = [le + box_size for le in left_edge]

        ds_obj = YTHDF5.build(
            filename,
            domain_dimensions,
            bbox,
            chunksize=chunksize,
            overwrite=overwrite,
        )
        ds_obj.add_model(self, [0, 0, 0], [0, 0, 0])

        return ds_obj.create_dataset()


# This is only for backwards-compatibility
class HydrostaticEquilibrium(ClusterModel):
    pass
