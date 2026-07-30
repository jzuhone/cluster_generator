"""
Module containing methods for particle virialization.
"""

from collections import OrderedDict

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from tqdm.auto import tqdm
from unyt import unyt_array

from cluster_generator.opt.cython_utils import generate_velocities
from cluster_generator.particles import ClusterParticles
from cluster_generator.utils import cgparams, generate_particle_radii, mylog, quad


class VirialEquilibrium:
    def __init__(self, model, ptype="dark_matter", df=None):
        r"""
        Generate a virial equilibrium model from a profile.

        Parameters
        ----------
        model : :class:`~cluster_generator.model.ClusterModel`
            The cluster model which will be used to
            construct the virial equilibrium.
        ptype : string, optional
            The type of the particles which can be generated from this
            object, either "dark_matter" or "stellar". Default: "dark_matter"
        df : unyt_array
            The particle distribution function. If not supplied, it will
            be generated.
        """
        self.num_elements = model.num_elements
        self.ptype = ptype
        self.model = model
        if df is None:
            self._generate_df()
        else:
            self.df = df
            f = df.d[::-1]
            self.f = InterpolatedUnivariateSpline(self.ee, f)

    def _generate_df(self):
        pden = self.model[f"{self.ptype}_density"][::-1]
        density_spline = InterpolatedUnivariateSpline(self.ee, pden)
        g = np.zeros(self.num_elements)

        def dgdp(t, e):
            return 2 * density_spline(e - t * t, 1)

        pbar = tqdm(
            leave=True,
            total=self.num_elements,
            desc="Computing particle DF ",
            disable=(~cgparams["system"]["display"]["progress_bars"]),
        )
        for i in range(self.num_elements):
            g[i] = quad(
                dgdp,
                0.0,
                np.sqrt(self.ee[i]),
                epsabs=1.49e-05,
                epsrel=1.49e-05,
                args=(self.ee[i]),
            )[0]
            pbar.update()
        pbar.close()
        g_spline = InterpolatedUnivariateSpline(self.ee, g)
        ff = g_spline(self.ee, 1) / (np.sqrt(8.0) * np.pi**2)
        self.f = InterpolatedUnivariateSpline(self.ee, ff)
        self.df = unyt_array(ff[::-1], "Msun*Myr**3/kpc**6")

    @property
    def ee(self):
        return -self.model["gravitational_potential"].d[::-1]

    @property
    def ff(self):
        return self.df.d[::-1]

    def check_virial(self):
        r"""
        Computes the radial density profile for the collisionless
        particles computed from integrating over the distribution
        function, and the relative difference between this and the
        input density profile.

        Returns
        -------
        rho : NumPy array
            The density profile computed from integrating the
            distribution function.
        chk : NumPy array
            The relative difference between the input density
            profile and the one calculated using this method.
        """
        n = self.num_elements
        rho = np.zeros(n)
        pden = self.model[f"{self.ptype}_density"].d

        def rho_int(e, psi):
            return self.f(e) * np.sqrt(2 * (psi - e))

        for i, e in enumerate(self.ee):
            rho[i] = 4.0 * np.pi * quad(rho_int, 0.0, e, args=(e,))[0]
        chk = (rho[::-1] - pden) / pden
        mylog.info(
            "The maximum relative deviation of this profile from virial equilibrium is %g",
            np.abs(chk).max(),
        )
        return rho[::-1], chk

    def generate_particles(
        self,
        num_particles,
        r_max=None,
        sub_sample=1,
        compute_potential=False,
        prng=None,
    ):
        """
        Generate a set of dark matter or star particles in virial equilibrium.

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
            A set of dark matter or star particles.
        """
        from cluster_generator.utils import parse_prng

        num_particles_sub = num_particles // sub_sample
        key = {"dark_matter": "dm", "stellar": "star"}[self.ptype]
        density = f"{self.ptype}_density"
        mass = f"{self.ptype}_mass"
        energy_spline = InterpolatedUnivariateSpline(self.model["radius"].d, self.ee[::-1])

        prng = parse_prng(prng)

        mylog.info("We will be assigning %s %s particles.", num_particles, self.ptype)
        mylog.info("Compute %s particle positions.", num_particles)

        nonzero = self.model[density] > 0.0
        radius_sub, mtot = generate_particle_radii(
            self.model["radius"].d[nonzero],
            self.model[mass].d[nonzero],
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

        fields[key, "particle_position"] = unyt_array(
            [
                radius * np.sin(theta) * np.cos(phi),
                radius * np.sin(theta) * np.sin(phi),
                radius * np.cos(theta),
            ],
            "kpc",
        ).T

        mylog.info("Compute %s particle velocities.", self.ptype)

        psi = energy_spline(radius_sub)
        vesc = 2.0 * psi
        fv2esc = vesc * self.f(psi)
        vesc = np.sqrt(vesc)

        velocity_sub = generate_velocities(
            psi,
            vesc,
            fv2esc,
            self.f.get_knots(),
            self.f.get_coeffs(),
            self.f._eval_args[2],
            int(~cgparams["system"]["display"]["progress_bars"]),
        )

        if sub_sample > 1:
            velocity = np.tile(velocity_sub, sub_sample)[:num_particles]
        else:
            velocity = velocity_sub

        theta = np.arccos(prng.uniform(low=-1.0, high=1.0, size=num_particles))
        phi = 2.0 * np.pi * prng.uniform(size=num_particles)

        fields[key, "particle_velocity"] = unyt_array(
            [
                velocity * np.sin(theta) * np.cos(phi),
                velocity * np.sin(theta) * np.sin(phi),
                velocity * np.cos(theta),
            ],
            "kpc/Myr",
        ).T

        fields[key, "particle_mass"] = unyt_array([mtot / num_particles] * num_particles, "Msun")

        if compute_potential:
            if sub_sample > 1:
                phi = -np.tile(psi, sub_sample)
            else:
                phi = -psi
            fields[key, "particle_potential"] = unyt_array(phi, "kpc**2/Myr**2")

        return ClusterParticles(key, fields)
