import numpy as np
from collections import OrderedDict
from yt import YTArray, YTQuantity
from scipy.integrate import cumtrapz, quad
from scipy.interpolate import InterpolatedUnivariateSpline
from cluster_generator.utils import \
    integrate, integrate_toinf, mylog, \
    integrate_mass, kboltz, \
    mp, G, generate_particle_radii
from cluster_generator.cluster_model import ClusterModel
from cluster_generator.cluster_particles import \
    ClusterParticles

X_H = 0.76
mu = 1.0/(2.0*X_H + 0.75*(1.0-X_H))
mue = 1.0/(X_H+0.5*(1.0-X_H))
tt = 2.0/3.0
mtt = -tt
ft = 5.0/3.0
tf = 3.0/5.0
mtf = -tf
gamma = ft
et = 8.0/3.0
te = 3.0/8.0


class HydrostaticEquilibrium(ClusterModel):

    _type_name = "hydrostatic"

    default_fields = ["density","temperature","pressure","total_density",
                      "gravitational_potential","gravitational_field",
                      "total_mass","gas_mass","dark_matter_mass",
                      "dark_matter_density","stellar_density","stellar_mass"]

    @classmethod
    def _from_scratch(cls, fields, stellar_density=None, parameters=None):
        rr = fields["radius"].d
        mylog.info("Integrating gravitational potential profile.")
        tdens_func = InterpolatedUnivariateSpline(rr, fields["total_density"].d)
        gpot_profile = lambda r: tdens_func(r)*r
        gpot1 = fields["total_mass"]/fields["radius"]
        gpot2 = YTArray(4.*np.pi*integrate(gpot_profile, rr), "Msun/kpc")
        fields["gravitational_potential"] = -G*(gpot1 + gpot2)
        fields["gravitational_potential"].convert_to_units("kpc**2/Myr**2")

        if "density" in fields and "gas_mass" not in fields:
            mylog.info("Integrating gas mass profile.")
            m0 = fields["density"].d[0]*rr[0]**3/3.
            fields["gas_mass"] = YTArray(
                4.0*np.pi*cumtrapz(fields["density"]*rr*rr,
                                   x=rr, initial=0.0)+m0, "Msun")

        if stellar_density is not None:
            fields["stellar_density"] = YTArray(stellar_density(rr),
                                                "Msun/kpc**3")
            mylog.info("Integrating stellar mass profile.")
            fields["stellar_mass"] = YTArray(integrate_mass(stellar_density, rr),
                                             "Msun")

        mdm = fields["total_mass"].copy()
        ddm = fields["total_density"].copy()
        if "density" in fields:
            mdm -= fields["gas_mass"]
            ddm -= fields["density"]
        if "stellar_mass" in fields:
            mdm -= fields["stellar_mass"]
            ddm -= fields["stellar_density"]
        mdm[ddm.v < 0.0][:] = mdm.max()
        ddm[ddm.v < 0.0][:] = 0.0

        if ddm.sum() > 0.0 and mdm.sum() > 0.0:
            fields["dark_matter_density"] = ddm
            fields["dark_matter_mass"] = mdm
        else:
            raise RuntimeError("The total dark matter mass is either zero or negative!!")

        if "density" in fields:
            fields["gas_fraction"] = fields["gas_mass"]/fields["total_mass"]
            fields["electron_number_density"] = \
                fields["density"].to("cm**-3", "number_density", mu=mue)
            fields["entropy"] = \
                fields["temperature"]*fields["electron_number_density"]**mtt

        return cls(rr.size, fields, parameters=parameters)

    @classmethod
    def from_dens_and_temp(cls, rmin, rmax, density, temperature,
                           stellar_density=None, num_points=1000, 
                           parameters=None):
        mylog.info("Computing the profiles from density and temperature.")
        rr = np.logspace(np.log10(rmin), np.log10(rmax), num_points,
                         endpoint=True)
        fields = OrderedDict()
        fields["radius"] = YTArray(rr, "kpc")
        fields["density"] = YTArray(density(rr), "Msun/kpc**3")
        fields["temperature"] = YTArray(temperature(rr), "keV")
        fields["pressure"] = fields["density"]*fields["temperature"]
        fields["pressure"] /= mu*mp
        fields["pressure"].convert_to_units("Msun/(Myr**2*kpc)")
        pressure_spline = InterpolatedUnivariateSpline(rr, fields["pressure"].d)
        dPdr = YTArray(pressure_spline(rr, 1), "Msun/(Myr**2*kpc**2)")
        fields["gravitational_field"] = dPdr/fields["density"]
        fields["gravitational_field"].convert_to_units("kpc/Myr**2")
        fields["gas_mass"] = YTArray(integrate_mass(density, rr), "Msun")
        fields["total_mass"] = -fields["radius"]**2*fields["gravitational_field"]/G
        total_mass_spline = InterpolatedUnivariateSpline(rr, fields["total_mass"].v)
        dMdr = YTArray(total_mass_spline(rr, nu=1), "Msun/kpc")
        fields["total_density"] = dMdr/(4.*np.pi*fields["radius"]**2)
        return cls._from_scratch(fields, stellar_density=stellar_density,
                                 parameters=parameters)

    @classmethod
    def from_dens_and_tden(cls, rmin, rmax, density, total_density,
                           stellar_density=None, num_points=1000, 
                           parameters=None):
        mylog.info("Computing the profiles from density and total density.")
        rr = np.logspace(np.log10(rmin), np.log10(rmax), num_points,
                         endpoint=True)
        fields = OrderedDict()
        fields["radius"] = YTArray(rr, "kpc")
        fields["density"] = YTArray(density(rr), "Msun/kpc**3")
        fields["total_density"] = YTArray(total_density(rr), "Msun/kpc**3")
        mylog.info("Integrating total mass profile.")
        fields["total_mass"] = YTArray(integrate_mass(total_density, rr), "Msun")
        fields["gas_mass"] = YTArray(integrate_mass(density, rr), "Msun")
        fields["gravitational_field"] = -G*fields["total_mass"]/(fields["radius"]**2)
        fields["gravitational_field"].convert_to_units("kpc/Myr**2")
        g = fields["gravitational_field"].in_units("kpc/Myr**2").v
        g_r = InterpolatedUnivariateSpline(rr, g)
        dPdr_int = lambda r: density(r)*g_r(r)
        mylog.info("Integrating pressure profile.")
        P = -integrate(dPdr_int, rr)
        dPdr_int2 = lambda r: density(r)*g[-1]*(rr[-1]/r)**2
        P -= quad(dPdr_int2, rr[-1], np.inf, limit=100)[0]
        fields["pressure"] = YTArray(P, "Msun/kpc/Myr**2")
        fields["temperature"] = fields["pressure"]*mu*mp/fields["density"]
        fields["temperature"].convert_to_units("keV")

        return cls._from_scratch(fields, stellar_density=stellar_density, 
                                 parameters=parameters)

    @classmethod
    def no_gas(cls, rmin, rmax, total_density, stellar_density=None,
               num_points=1000, parameters=None):
        rr = np.logspace(np.log10(rmin), np.log10(rmax), num_points,
                         endpoint=True)
        fields = OrderedDict()
        fields["radius"] = YTArray(rr, "kpc")
        fields["total_density"] = YTArray(total_density(rr), "Msun/kpc**3")
        mylog.info("Integrating total mass profile.")
        fields["total_mass"] = YTArray(integrate_mass(total_density, rr), "Msun")
        fields["gravitational_field"] = -G*fields["total_mass"]/(fields["radius"]**2)
        fields["gravitational_field"].convert_to_units("kpc/Myr**2")

        return cls._from_scratch(fields, stellar_density=stellar_density,
                                 parameters=parameters)

    @classmethod
    def from_entr_and_tden(cls, rmin, rmax, entropy, total_density,
                           rfgas, fgas, stellar_density=None,
                           parameters=None, num_points=1000):
        rr = np.logspace(np.log10(rmin), np.log10(rmax), num_points,
                         endpoint=True)
        fields = OrderedDict()
        mylog.info("Computing the profiles from density and entropy.")
        fields["radius"] = YTArray(rr, "kpc")
        fields["total_density"] = YTArray(total_density(rr), "Msun/kpc**3")
        mylog.info("Integrating total mass profile.")
        fields["total_mass"] = YTArray(integrate_mass(total_density, rr), "Msun")
        fields["gravitational_field"] = -G*fields["total_mass"]/(fields["radius"]**2)
        fields["gravitational_field"].convert_to_units("kpc/Myr**2")
        g = fields["gravitational_field"].d
        g_r = InterpolatedUnivariateSpline(rr, g)
        K = YTArray(entropy(rr), "keV*cm**2").in_base("galactic")
        K *= (mp*mue)**-ft
        Kr = InterpolatedUnivariateSpline(rr, K.d)
        integrand = lambda r: Kr(r)**mtf*g_r(r)
        I = integrate(integrand, rr)
        integrand2 = lambda r: Kr(rr[-1])**mtf*g[-1]*(rr[-1]/r)**2
        I += quad(integrand2, rr[-1], np.inf, limit=100)[0]
        P = (-0.4*I)**2.5
        rho = (P/K.d)**tf
        m0 = rho[0]*rr[0]**3/3.
        m_g = 4.0*np.pi*cumtrapz(rho*rr*rr, x=rr, initial=0.0)+m0
        mgval = np.interp(rfgas, rr, m_g)
        mtval = np.interp(rfgas, rr, fields["total_mass"].d)
        x = fgas * mtval / mgval
        P *= x
        rho *= x
        fields["pressure"] = YTArray(P, "Msun/kpc/Myr**2")
        fields["density"] = YTArray(rho, "Msun/kpc**3")
        density = InterpolatedUnivariateSpline(rr, rho)
        fields["gas_mass"] = YTArray(integrate_mass(density, rr), "Msun")
        fields["temperature"] = fields["pressure"]*mu*mp/fields["density"]
        fields["temperature"].convert_to_units("keV")
        return cls._from_scratch(fields, stellar_density=stellar_density,
                                 parameters=parameters)

    def check_model(self):
        r"""
        Determine the deviation of the model from hydrostatic equilibrium. 

        Returns
        -------
        chk : NumPy array
            An array containing the relative deviation from hydrostatic
            equilibrium as a function of radius.
        """
        rr = self.fields["radius"].v
        pressure_spline = InterpolatedUnivariateSpline(rr, self.fields["pressure"].v)
        dPdx = YTArray(pressure_spline(rr, 1), "Msun/(Myr**2*kpc**2)")
        rhog = self.fields["density"]*self.fields["gravitational_field"]
        chk = dPdx - rhog
        chk /= rhog
        mylog.info("The maximum relative deviation of this profile from "
                   "hydrostatic equilibrium is %g" % np.abs(chk).max())
        return chk

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
        B = np.sqrt(2.0*self["pressure"]/beta)
        if gaussian:
            B *= np.sqrt(4.0*np.pi)
        B.convert_to_units("gauss")
        self.set_field("magnetic_field_strength", B)

    def set_magnetic_field_from_density(self, B0, eta=2./3., gaussian=True):
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
        if not hasattr(B0, "units"):
            B0 = YTQuantity(B0, "gauss")
        else:
            B0.convert_to_units("gauss")
        B = B0*(self["density"]/self["density"][0])**eta
        if not gaussian:
            B /= np.sqrt(4.0*np.pi)
        self.set_field("magnetic_field_strength", B)

    def generate_particles(self, num_particles, r_max=None, sub_sample=1):
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
        """
        mylog.info("We will be assigning %d particles." % num_particles)
        mylog.info("Compute particle positions.")

        num_particles_sub = num_particles // sub_sample

        radius_sub, mtot = generate_particle_radii(self["radius"].d,
                                                   self["gas_mass"].d,
                                                   num_particles_sub, r_max=r_max)

        if sub_sample > 1:
            radius = np.tile(radius_sub, sub_sample)[:num_particles]
        else:
            radius = radius_sub

        theta = np.arccos(np.random.uniform(low=-1., high=1., size=num_particles))
        phi = 2.*np.pi*np.random.uniform(size=num_particles)

        fields = OrderedDict()

        fields["gas", "particle_position"] = YTArray([radius*np.sin(theta)*np.cos(phi),
                                                      radius*np.sin(theta)*np.sin(phi),
                                                      radius*np.cos(theta)], "kpc").T

        mylog.info("Compute particle thermal energies, densities, and masses.")

        e_arr = 1.5*self.fields["pressure"]/self.fields["density"]
        get_energy = InterpolatedUnivariateSpline(self.fields["radius"], e_arr)

        if sub_sample > 1:
            energy = np.tile(get_energy(radius_sub), sub_sample)[:num_particles]
        else:
            energy = get_energy(radius)

        fields["gas", "thermal_energy"] = YTArray(energy, "kpc**2/Myr**2")
        fields["gas", "particle_mass"] = YTArray([mtot/num_particles]*num_particles, "Msun")

        get_density = InterpolatedUnivariateSpline(self.fields["radius"], 
                                                   self.fields["density"])

        if sub_sample > 1:
            density = np.tile(get_density(radius_sub), sub_sample)[:num_particles]
        else:
            density = get_density(radius)

        fields["gas", "density"] = YTArray(density, "Msun/kpc**3")

        mylog.info("Set particle velocities to zero.")

        fields["gas", "particle_velocity"] = YTArray(np.zeros((num_particles, 3)), "kpc/Myr")

        return ClusterParticles("gas", fields)
