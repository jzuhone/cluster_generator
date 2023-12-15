"""
:py:class:`radial_profiles.RadialProfile` objects and associated tools for working with analytically defined profiles.
"""
import inspect

import numpy as np

#: Alternative factor for rho(r) in NFW profiles. See `the wiki<https://en.wikipedia.org/wiki/Navarro%E2%80%93Frenk%E2%80%93White_profile>`_
_nfw_factor = lambda conc: 1.0 / (np.log(conc + 1.0) - conc / (1.0 + conc))


class RadialProfile:
    r"""
    The :py:class:`radial_profiles.RadialProfile` class acts as a wrapper on standard functions to represent radial profiles
    for different physical variables.

    Parameters
    ----------
    profile: :py:class:`radial_profiles.RadialProfile` or callable
        The radial profile to attribute to the object. The radial profile must be callable (i.e. ``lambda`` function) or
        another instance of :py:class:`radial_profiles.RadialProfile`.

    """

    #: The built-in options for :py:class:`~radial_profiles.RadialProfile` objects.
    builtin = [
        "constant_profile",
        "power_law_profile",
        "beta_model_profile",
        "hernquist_density_profile",
        "cored_hernquist_density_profile",
        "hernquist_mass_profile",
        "nfw_density_profile",
        "nfw_mass_profile",
        "tnfw_density_profile",
        "tnfw_mass_profile",
        "snfw_density_profile",
        "snfw_mass_profile",
        "cored_snfw_density_profile",
        "cored_snfw_mass_profile",
        "cored_snfw_total_mass",
        "einasto_density_profile",
        "einasto_mass_profile",
        "am06_density_profile",
        "vikhlinin_density_profile",
        "vikhlinin_temperature_profile",
        "am06_temperature_profile",
        "baseline_entropy_profile",
        "broken_entropy_profile",
        "walker_entropy_profile",
    ]
    _characteristic_range = [
        1,
        10000,
    ]  # Used for defining equality (needed for testing consistency)

    def __init__(self, profile, name=None):
        #: The profile name.
        self.name = None
        if isinstance(profile, RadialProfile):
            self.profile = profile.profile
        else:
            self.profile = profile

    def __call__(self, r):
        return self.profile(r)

    def __str__(self):
        if self.name is None:
            return object.__str__(self)
        else:
            return f"RadialProfile; type={self.name}."

    def __repr__(self):
        if self.name is None:
            return object.__repr__(self)
        else:
            return f"RadialProfile; type={self.name}."

    def _do_op(self, other, op):
        if hasattr(other, "profile"):
            p = lambda r: op(self.profile(r), other.profile(r))
        else:
            p = lambda r: op(self.profile(r), other)
        return p

    def __add__(self, other):
        p = self._do_op(other, np.add)
        return RadialProfile(p)

    def __mul__(self, other):
        p = self._do_op(other, np.multiply)
        return RadialProfile(p)

    __radd__ = __add__
    __rmul__ = __mul__

    def __pow__(self, power):
        p = lambda r: self.profile(r) ** power
        return RadialProfile(p)

    def __eq__(self, other):
        ar = np.linspace(*self._characteristic_range, 1000)
        return np.array_equal(self(ar), other(ar))

    def add_core(self, r_core, alpha):
        r"""
        Add a core to the pre-existing profile.

        Parameters
        ----------
        r_core : float
            The core radius in kpc.
        alpha : float
            The power-low index inside the exponential.

        Notes
        -----
        ``add_core`` is implemented by taking the existing profile :math:`f(r)` and altering it such that

        .. math::

            f'(r) = \left(1-\exp\left(\frac{-r}{r_{core}}\right)^\alpha\right) f(r).

        This will cause any cuspy profile (i.e. one for which :math:`\left.\frac{d}{dr} f(r)\right|_{r=0} > 0` and which grows
        faster than the exponential term added to instead contain a core and go to 0 in its limit.

        """

        def _core(r):
            x = r / r_core
            ret = 1.0 - np.exp(-(x**alpha))
            return self.profile(r) * ret

        return RadialProfile(_core)

    def cutoff(self, r_cut, k=5):
        r"""
        Generate a truncated form of the profile.

        Parameters
        ----------
        r_cut: float or int
            The cutoff radius beyond which the truncation should dominate the profile behavior [kpc].
        k: int
            The truncation rate. Higher ``k`` will cause the truncation to go to zero faster.

        Returns
        -------
        RadialProfile
            The corresponding :py:class:`~radial_profiles.RadialProfile` object with the truncated profile.

        Notes
        -----
        The truncation is achieved by multiplying the profile by the factor

        .. math::

            1-\frac{1}{1+\exp\left(-2k\left(\frac{r}{r_{cut}}\right)\right)}.

        """

        def _cutoff(r):
            x = r / r_cut
            step = 1.0 / (1.0 + np.exp(-2 * k * (x - 1)))
            p = self.profile(r) * (1.0 - step)
            return p

        return RadialProfile(_cutoff)

    @classmethod
    def from_array(cls, r, f_r):
        """
        Generate a callable radial profile using an array of radii
        and an array of values.

        Parameters
        ----------
        r : array-like
            Array of radii in kpc.
        f_r : array-like
            Array of profile values in the appropriate units.

        Returns
        -------
        RadialProfile
            The corresponding radial profile.

        Notes
        -----
        This function uses ``scipy.interpolate.UnivariateSpline`` to generate a continuous spectrum. May lead to problematic behavior
        beyond the intended radii.

        """
        from scipy.interpolate import UnivariateSpline

        f = UnivariateSpline(r, f_r)
        return cls(f)

    @classmethod
    def built_in(cls, name, *args):
        """Initialize a :py:class:`~radial_profiles.RadialProfile` from the specified name and given args."""
        if name in cls.builtin:
            return globals()[name](*args)
        else:
            raise ValueError(f"The name {name} is either not builtin or is incorrect.")

    @classmethod
    def from_binary(cls, f):
        """
        Load a specific instance of a :py:class:`~radial_profiles.RadialProfile` object from the serialized version of the instance saved to disk.

        Parameters
        ----------
        f: str
            The filename to open. Should be a valid ``.rp`` file type.

        Returns
        -------
        RadialProfile
            The :py:class:`~radial_profiles.RadialProfile` object on disk.

        """
        import dill as pickle

        with open(f, "rb") as bf:
            return pickle.load(bf)

    def to_binary(self, f):
        """
        Send the :py:class:`~radial_profiles.RadialProfile` instance to a serialized binary file.

        Parameters
        ----------
        f: str
            The preferred filename. For consistency, binary files should have ``.rp`` extension; however, this is not required.

        Returns
        -------
        None

        """
        import dill as pickle

        with open(f, "wb") as bf:
            pickle.dump(self, bf)

    def plot(self, rmin, rmax, num_points=1000, fig=None, ax=None, **kwargs):
        """
        Make a quick plot of a profile using Matplotlib.

        Parameters
        ----------
        rmin : float
            The minimum radius of the plot in kpc.
        rmax : float
            The maximum radius of the plot in kpc.
        num_points : integer, optional
            The number of logspaced points between rmin
            and rmax to use when making the plot. Default: 1000
        fig : :class:`~matplotlib.figure.Figure`, optional
            A Figure instance to plot in. Default: None, one will be
            created if not provided.
        ax : :class:`~matplotlib.axes.Axes`, optional
            An Axes instance to plot in. Default: None, one will be
            created if not provided.

        """
        import matplotlib.pyplot as plt

        plt.rc("font", size=18)
        plt.rc("axes", linewidth=2)
        if fig is None:
            fig = plt.figure(figsize=(10, 10))
        if ax is None:
            ax = fig.add_subplot(111)
        rr = np.logspace(np.log10(rmin), np.log10(rmax), num_points, endpoint=True)
        ax.loglog(rr, self(rr), **kwargs)
        ax.set_xlabel("Radius (kpc)")
        ax.tick_params(which="major", width=2, length=6)
        ax.tick_params(which="minor", width=2, length=3)
        return fig, ax


def constant_profile(const):
    """
    Provide constant profile.

    Parameters
    ----------
    const : float
        The value of the constant.

    """
    p = lambda r: const
    return RadialProfile(p, name=inspect.stack()[0][3])


def power_law_profile(A, r_s, alpha):
    """
    A profile which is a power-law with radius, scaled
    so that it has a certain value ``A`` at a scale
    radius ``r_s``. Can be used as a density, temperature,
    mass, or entropy profile (or whatever else one may
    need).

    Parameters
    ----------
    A : float
        Scale value of the profile at r = r_s.
    r_s : float
        Scale radius in kpc.
    alpha : float
        Power-law index of the profile.

    """
    p = lambda r: A * (r / r_s) ** alpha
    return RadialProfile(p, name=inspect.stack()[0][3])


def beta_model_profile(rho_c, r_c, beta):
    """
    A beta-model density profile [CaFu76]_.

    Parameters
    ----------
    rho_c : float
        The core density in Msun/kpc**3.
    r_c : float
        The core radius in kpc.
    beta : float
        The beta parameter.

    Returns
    -------
    RadialProfile
        The corresponding radial profile object.

    References
    ----------
    .. [CaFu76] (Cavaliere A.,Fusco-Femiano R., 1976, A&A, 49, 137).

    """
    p = lambda r: rho_c * ((1 + (r / r_c) ** 2) ** (-1.5 * beta))
    return RadialProfile(p, name=inspect.stack()[0][3])


def hernquist_density_profile(M_0, a):
    """
    A Hernquist density profile [Hern90]_.

    Parameters
    ----------
    M_0 : float
        The total mass in Msun.
    a : float
        The scale radius in kpc.

    Returns
    -------
    RadialProfile
        The corresponding radial profile object.

    References
    ----------
    .. [Hern90] (Hernquist, L. 1990, ApJ, 356, 359).

    """
    p = lambda r: M_0 / (2.0 * np.pi * a**3) / ((r / a) * (1.0 + r / a) ** 3)
    return RadialProfile(p, name=inspect.stack()[0][3])


def cored_hernquist_density_profile(M_0, a, b):
    """
    A Hernquist density profile [Hern90]_ with a core radius.

    Parameters
    ----------
    M_0 : float
        The total mass in Msun.
    a : float
        The scale radius in kpc.
    b : float
        The core radius in kpc.

    Returns
    -------
    RadialProfile
        The corresponding radial profile object.

    """
    p = (
        lambda r: M_0
        * b
        / (2.0 * np.pi * a**3)
        / ((1.0 + b * r / a) * (1.0 + r / a) ** 3)
    )
    return RadialProfile(p, name=inspect.stack()[0][3])


def hernquist_mass_profile(M_0, a):
    """
    A Hernquist mass profile [Hern90]_.

    Parameters
    ----------
    M_0 : float
        The total mass in Msun.
    a : float
        The scale radius in kpc.

    Returns
    -------
    RadialProfile
        The corresponding radial profile object.

    """
    p = lambda r: M_0 * r**2 / (r + a) ** 2
    return RadialProfile(p, name=inspect.stack()[0][3])


def convert_nfw_to_hernquist(M_200, r_200, conc):
    """
    Given M200, r200, and a concentration parameter for an
    NFW profile [NaFrW90]_, return the Hernquist ([Hern90]_) mass and scale radius
    parameters.

    Parameters
    ----------
    M_200 : float
        The mass of the halo at r200 in Msun.
    r_200 : float
        The radius corresponding to the overdensity of 200 times the
        critical density of the universe in kpc.
    conc : float
        The concentration parameter r200/r_s for the NFW profile.

    Returns
    -------
    RadialProfile
        The corresponding radial profile object.

    References
    ----------
    .. [NaFrW90] (Navarro, Julio F.; Frenk, Carlos S.; White, Simon D. M.; 1997ApJ...490..493N)

    """
    a = r_200 / (np.sqrt(0.5 * conc * conc * _nfw_factor(conc)) - 1.0)
    M0 = M_200 * (r_200 + a) ** 2 / r_200**2
    return M0, a


def nfw_density_profile(rho_s, r_s):
    """
    An NFW density profile [NaFrW90]_.

    Parameters
    ----------
    rho_s : float
        The scale density in Msun/kpc**3.
    r_s : float
        The scale radius in kpc.

    Returns
    -------
    RadialProfile
        The corresponding radial profile object.

    """
    p = lambda r: rho_s / ((r / r_s) * (1.0 + r / r_s) ** 2)
    return RadialProfile(p, name=inspect.stack()[0][3])


def nfw_mass_profile(rho_s, r_s):
    """
    An NFW mass profile [NaFrW90]_.

    Parameters
    ----------
    rho_s : float
        The scale density in Msun/kpc**3.
    r_s : float
        The scale radius in kpc.

    Returns
    -------
    RadialProfile
        The corresponding radial profile object.

    """

    def _nfw(r):
        x = r / r_s
        return 4 * np.pi * rho_s * r_s**3 * (np.log(1 + x) - x / (1 + x))

    return RadialProfile(_nfw, name=inspect.stack()[0][3])


def nfw_scale_density(conc, z=0.0, delta=200.0, cosmo=None):
    """
    Compute a scale density parameter for an NFW profile [NaFrW90]_
    given a concentration parameter, and optionally
    a redshift, overdensity, and cosmology.

    Parameters
    ----------
    conc : float
        The concentration parameter for the halo, which should
        correspond the selected overdensity (which has a default
        of 200).
    z : float, optional
        The redshift of the halo formation. Default: 0.0
    delta : float, optional
        The overdensity parameter for which the concentration
        is defined. Default: 200.0
    cosmo : yt Cosmology object
        The cosmology to be used when computing the critical
        density. If not supplied, a default one from yt will
        be used.

    Returns
    -------
    RadialProfile
        The corresponding radial profile object.

    """
    from yt.utilities.cosmology import Cosmology

    if cosmo is None:
        cosmo = Cosmology()
    rho_crit = cosmo.critical_density(z).to_value("Msun/kpc**3")
    rho_s = delta * rho_crit * conc**3 * _nfw_factor(conc) / 3.0
    return rho_s


def tnfw_density_profile(rho_s, r_s, r_t):
    """
    A truncated NFW [NaFrW90]_ (tNFW) density profile [BaMaO09]_.

    Parameters
    ----------
    rho_s : float
        The scale density in Msun/kpc**3.
    r_s : float
        The scale radius in kpc.
    r_t : float
        The truncation radius in kpc.

    Returns
    -------
    RadialProfile
        The corresponding radial profile object.

    References
    ----------
    .. [BaMaO09]  (Baltz, E.A.,Marshall, P., & Oguri, M. 2009, JCAP, 2009, 015)

    """

    def _tnfw(r):
        profile = rho_s / ((r / r_s) * (1 + r / r_s) ** 2)
        profile /= 1 + (r / r_t) ** 2
        return profile

    return RadialProfile(_tnfw, name=inspect.stack()[0][3])


def tnfw_mass_profile(rho_s, r_s, r_t):
    """
    A truncated NFW (tNFW) mass profile  [BaMaO09]_.

    Parameters
    ----------
    rho_s : float
        The scale density in Msun/kpc**3.
    r_s : float
        The scale radius in kpc.
    r_t : float
        The truncation radius in kpc.

    Returns
    -------
    RadialProfile
        The corresponding radial profile object.

    """
    from sympy import Symbol, integrate, lambdify

    xx = Symbol("x")
    aa = Symbol("a")
    yy = Symbol("y")
    f = integrate(xx**2 / (xx * (1 + xx) ** 2) / (1 + (xx / aa) ** 2), (xx, 0, yy))
    fl = lambdify((yy, aa), f, modules="numpy")

    def _tnfw(r):
        x = r / r_s
        a = r_t / r_s
        return 4 * np.pi * rho_s * r_s**3 * fl(x, a).astype("float64")

    return RadialProfile(_tnfw, name=inspect.stack()[0][3])


def snfw_density_profile(M, a):
    """
    A "super-NFW" density profile [LiWyS18]_.

    Parameters
    ----------
    M : float
        The total mass in Msun.
    a : float
        The scale radius in kpc.

    Returns
    -------
    RadialProfile
        The corresponding radial profile object.

    References
    ----------
    .. [LiWyS18] (Lilley, E. J., Wyn Evans, N., & Sanders, J.L. 2018, MNRAS)

    """

    def _snfw(r):
        x = r / a
        return 3.0 * M / (16.0 * np.pi * a**3) / (x * (1.0 + x) ** 2.5)

    return RadialProfile(_snfw, name=inspect.stack()[0][3])


def snfw_mass_profile(M, a):
    """
    A "super-NFW" mass profile [LiWyS18]_.

    Parameters
    ----------
    M : float
        The total mass in Msun.
    a : float
        The scale radius in kpc.

    Returns
    -------
    RadialProfile
        The corresponding radial profile object.

    """

    def _snfw(r):
        x = r / a
        return M * (1.0 - (2.0 + 3.0 * x) / (2.0 * (1.0 + x) ** 1.5))

    return RadialProfile(_snfw, name=inspect.stack()[0][3])


def snfw_total_mass(mass, radius, a):
    """
    Find the total mass parameter for the super-NFW
    model [LiWyS18]_ by inputting a reference mass and radius
    (say, M200c and R200c), along with the scale radius.

    Parameters
    ----------
    mass : float
        The input mass in Msun.
    radius : float
        The input radius that the input ``mass`` corresponds to in kpc.
    a : float
        The scale radius in kpc.

    Returns
    -------
    RadialProfile
        The corresponding radial profile object.

    """
    mp = snfw_mass_profile(1.0, a)
    return mass / mp(radius)


def cored_snfw_density_profile(M, a, r_c):
    """
    A cored "super-NFW" density profile [LiWyS18]_.

    Parameters
    ----------
    M : float
        The total mass in Msun.
    a : float
        The scale radius in kpc.
    r_c : float
        The core radius in kpc.

    Returns
    -------
    RadialProfile
        The corresponding radial profile object.

    """
    b = a / r_c

    def _snfw(r):
        x = r / a
        return (
            3.0 * M * b / (16.0 * np.pi * a**3) / ((1.0 + b * x) * (1.0 + x) ** 2.5)
        )

    return RadialProfile(_snfw, name=inspect.stack()[0][3])


def cored_snfw_mass_profile(M, a, r_c):
    """
    A cored "super-NFW" mass profile [LiWyS18]_.

    Parameters
    ----------
    M : float
        The total mass in Msun.
    a : float
        The scale radius in kpc.
    r_c : float
        The core radius in kpc.

    Returns
    -------
    RadialProfile
        The corresponding radial profile object.

    """
    b = a / r_c

    def _snfw(r):
        x = r / a
        y = np.complex128(np.sqrt(x + 1.0))
        d = np.sqrt(np.complex128(b / (1.0 - b)))
        e = b * (b - 1.0) ** 2
        ret = (1.0 - 1.0 / y) * (b - 2.0) / (b - 1.0) ** 2
        ret += (1.0 / y**3 - 1.0) / (3.0 * (b - 1.0))
        ret += d * (np.arctan(y * d) - np.arctan(d)) / e
        return 1.5 * M * b * ret.astype("float64")

    return RadialProfile(_snfw, name=inspect.stack()[0][3])


def snfw_conc(conc_nfw):
    """
    Given an NFW concentration parameter, calculate the
    corresponding sNFW concentration parameter. This comes
    from Equation 31 of [LiWyS18]_.

    Parameters
    ----------
    conc_nfw : float
        NFW concentration for r200c.

    Returns
    -------
    RadialProfile
        The corresponding radial profile object.

    """
    return 0.76 * conc_nfw + 1.36


def cored_snfw_total_mass(mass, radius, a, r_c):
    """
    Find the total mass parameter for the cored super-NFW
    model [LiWyS18]_ by inputting a reference mass and radius
    (say, M200c and R200c), along with the scale radius.

    Parameters
    ----------
    mass : float
        The input mass in Msun.
    radius : float
        The input radius that the input ``mass`` corresponds to in kpc.
    a : float
        The scale radius in kpc.
    r_c : float
        The core radius in kpc.

    Returns
    -------
    RadialProfile
        The corresponding radial profile object.

    """
    mp = cored_snfw_mass_profile(1.0, a, r_c)
    return mass / mp(radius)


_dn = lambda n: 3.0 * n - 1.0 / 3.0 + 8.0 / (1215.0 * n) + 184.0 / (229635.0 * n * n)


def einasto_density_profile(M, r_s, n):
    """
    A density profile where the logarithmic slope is a
    power-law [Eina65]_. The form here is that given in Section 2 of
    [RvGB12]_.

    Parameters
    ----------
    M : float
        The total mass of the profile in M.
    r_s : float
        The scale radius in kpc.
    n : float
        The inverse power-law index.

    Returns
    -------
    RadialProfile
        The corresponding radial profile object.

    References
    ----------
    .. [Eina65] J. Einasto (1965), Kinematics and dynamics of stellar systems, Trudy Inst. Astrofiz. Alma-Ata 5, 87
    .. [RvGB12] (Retana-Montenegro, E; et. al. 2012A&A...540A..70R)

    """
    from scipy.special import gamma

    alpha = 1.0 / n
    h = r_s / _dn(n) ** n
    rho_0 = M / (4.0 * np.pi * h**3 * n * gamma(3.0 * n))

    def _einasto(r):
        s = r / h
        return rho_0 * np.exp(-(s**alpha))

    return RadialProfile(_einasto, name=inspect.stack()[0][3])


def einasto_mass_profile(M, r_s, n):
    """
    A mass profile where the logarithmic slope is a
    power-law [Eina65]_. The form here is that given in Section 2 of
    [RvGB12]_.

    Parameters
    ----------
    M : float
        The total mass of the profile in M.
    r_s : float
        The scale radius in kpc.
    n : float
        The inverse power-law index.

    Returns
    -------
    RadialProfile
        The corresponding radial profile object.

    """
    from scipy.special import gammaincc

    alpha = 1.0 / n
    h = r_s / _dn(n) ** n

    def _einasto(r):
        s = r / h
        return M * (1.0 - gammaincc(3.0 * n, s**alpha))

    return RadialProfile(_einasto, name=inspect.stack()[0][3])


def am06_density_profile(rho_0, a, a_c, c, n):
    """
    The density profile for galaxy clusters suggested by [AsMa06]_.
    Works best in concert with the :py:func:`radial_profiles.am06_temperature_profile`.

    Parameters
    ----------
    rho_0 : float
        The scale density of the profile in Msun/kpc**3.
    a : float
        The scale radius in kpc.
    a_c : float
        The scale radius of the cool core in kpc.
    c : float
        The scale of the temperature drop of the cool core.
    n : float
        The integer scaling on alpha and beta.

    Returns
    -------
    RadialProfile
        The corresponding radial profile object.

    References
    ----------
    .. [AsMa06] Ascasibar, Y., & Markevitch, M. 2006, ApJ, 650, 102.

    """
    alpha = -1.0 - n * (c - 1.0) / (c - a / a_c)
    beta = 1.0 - n * (1.0 - a / a_c) / (c - a / a_c)
    p = (
        lambda r: rho_0
        * (1.0 + r / a_c)
        * (1.0 + r / a_c / c) ** alpha
        * (1.0 + r / a) ** beta
    )
    return RadialProfile(p, name=inspect.stack()[0][3])


def ad07_density_profile(
    T0, t, a, alpha, f, n=4, mu=0.6, omega_b=0.048, omega_dm=0.262
):
    """
    Pseudo-polytropic gas density profile from [AsDi08]_.

    Parameters
    ----------
    T0: float
        (keV) The core temperature of the gas distribution.
    t: float
        (Dimensionless) value representing the degree of cooling in the cluster core.
    a: float
        (kpc) Scale length of both the temperature and density profiles.
    alpha: float
        (dimensionless) Ratio of a_c/a, a_c is the cooling radius.
    f: float
        (dimensionless) The gas fraction.
    n: int, optional
        (dimensionless) the polytropic index. Default is 4.
    mu: float, optional
        (dimensionless) the mean molecular mass of the gas. Default is 0.6
    omega_b: float, optional
        (dimensionless) the cosmic baryon fraction parameter. Default is 0.048.
    omega_dm: float, optional
        (dimensionless) the cosmic dark matter fraction parameter. Default is 0.262
    Returns
    -------
    :py:class:`radial_profiles.RadialProfile`
        The corresponding radial profile.

    References
    ----------
    .. [AsDi08] Ascasibar & Diego, 2008MNRAS.383..369A

    """
    from unyt import physical_constants as const
    from unyt import unyt_quantity

    # - computing the mass norm -#
    M = (
        unyt_quantity(a, "kpc")
        * (n + 1)
        * unyt_quantity(T0, "keV")
        / (mu * const.mp * const.G)
    )
    M = M.to("Msun")

    # - computing the density norm -#
    rho0 = f * (omega_b / omega_dm) * (M / (2 * np.pi * unyt_quantity(a, "kpc") ** 3))
    rho0 = rho0.to("Msun/kpc**3")

    function = (
        lambda r, A=a, T=t, ALPHA=alpha, N=4, RHO=rho0.d: RHO
        * ((1 + (r / A)) / (T * ALPHA + (r / A)))
        ** (1 + ((ALPHA - T * ALPHA) * (1 - T * ALPHA)) * (N + 1))
        * (ALPHA + (r / A))
        / ((1 + (r / A)) ** (N + 1))
    )
    return RadialProfile(function, name=inspect.stack()[0][3])


def vikhlinin_density_profile(rho_0, r_c, r_s, alpha, beta, epsilon, gamma=None):
    """
    A modified beta-model density profile for galaxy
    clusters from [ViKrF06]_.

    Parameters
    ----------
    rho_0 : float
        The scale density in Msun/kpc**3.
    r_c : float
        The core radius in kpc.
    r_s : float
        The scale radius in kpc.
    alpha : float
        The inner logarithmic slope parameter.
    beta : float
        The middle logarithmic slope parameter.
    epsilon : float
        The outer logarithmic slope parameter.
    gamma : float
        This parameter controls the width of the outer
        transition. If None, it will be gamma = 3 by default.

    Returns
    -------
    RadialProfile
        The corresponding radial profile object.

    """
    if gamma is None:
        gamma = 3.0
    profile = (
        lambda r: rho_0
        * (r / r_c) ** (-0.5 * alpha)
        * (1.0 + (r / r_c) ** 2) ** (-1.5 * beta + 0.25 * alpha)
        * (1.0 + (r / r_s) ** gamma) ** (-0.5 * epsilon / gamma)
    )
    return RadialProfile(profile, name=inspect.stack()[0][3])


def ad07_temperature_profile(T0, t, a, alpha):
    """
    Pseudo-polytropic gas temperature profile from [AsDi08]_.

    Parameters
    ----------
    T0: float
        (keV) The core temperature of the gas distribution.
    t: float
        (Dimensionless) value representing the degree of cooling in the cluster core.
    a: float
        (kpc) Scale length of both the temperature and density profiles.
    alpha: float
        (dimensionless) a_c/a

    Returns
    -------
    :py:class:`radial_profiles.RadialProfile`
        The corresponding radial profile.

    """
    function = lambda r, A=a, ALPHA=alpha, T=t, TEMP0=T0: (TEMP0 / (1 + (r / A))) * (
        (T + (r / (ALPHA * A))) / (1 + (r / (ALPHA * A)))
    )
    return RadialProfile(function, name=inspect.stack()[0][3])


def vikhlinin_temperature_profile(T_0, a, b, c, r_t, T_min, r_cool, a_cool):
    """
    A temperature profile for galaxy clusters from [ViKrF06]_.


    Parameters
    ----------
    T_0 : float
        The scale temperature of the profile in keV.
    a : float
        The inner logarithmic slope.
    b : float
        The width of the transition region.
    c : float
        The outer logarithmic slope.
    r_t : float
        The scale radius kpc.
    T_min : float
        The minimum temperature in keV.
    r_cool : float
        The cooling radius in kpc.
    a_cool : float
        The logarithmic slope in the cooling region.

    Returns
    -------
    RadialProfile
        The corresponding radial profile object.

    References
    ----------
    .. [ViKrF06] Vikhlinin, A., Kravtsov, A., Forman, W., et al. 2006, ApJ, 640, 691.

    """

    def _temp(r):
        x = (r / r_cool) ** a_cool
        t = (r / r_t) ** (-a) / ((1.0 + (r / r_t) ** b) ** (c / b))
        return T_0 * t * (x + T_min / T_0) / (x + 1)

    return RadialProfile(_temp, name=inspect.stack()[0][3])


def am06_temperature_profile(T_0, a, a_c, c):
    """
    The temperature profile for galaxy clusters suggested by [AsMa06]_.
    Works best in concert with the :py:func:`radial_profiles.am06_density_profile`.

    Parameters
    ----------
    T_0 : float
        The scale temperature of the profile in keV.
    a : float
        The scale radius in kpc.
    a_c : float
        The cooling radius in kpc.
    c : float
        The scale of the temperature drop of the cool core.

    Returns
    -------
    RadialProfile
        The corresponding radial profile object.

    """
    p = lambda r: T_0 / (1.0 + r / a) * (c + r / a_c) / (1.0 + r / a_c)
    return RadialProfile(p, name=inspect.stack()[0][3])


def baseline_entropy_profile(K_0, K_200, r_200, alpha):
    """
    The baseline entropy profile for galaxy clusters [VoKB05]_.

    Parameters
    ----------
    K_0 : float
        The central entropy floor in keV*cm**2.
    K_200 : float
        The entropy at the radius r_200 in keV*cm**2.
    r_200 : float
        The virial radius in kpc.
    alpha : float
        The logarithmic slope of the profile.

    Returns
    -------
    RadialProfile
        The corresponding radial profile object.

    References
    ----------
    .. [VoKB05] (Voit, G.M.,Kay, S.T., & Bryan, G.L. 2005, MNRAS, 364, 909).

    """
    p = lambda r: K_0 + K_200 * (r / r_200) ** alpha
    return RadialProfile(p, name=inspect.stack()[0][3])


def broken_entropy_profile(r_s, K_scale, alpha, K_0=0.0):
    """Generate a broken entropy profile"""

    def _entr(r):
        x = r / r_s
        ret = (x**alpha) * (1.0 + x**5) ** (0.2 * (1.1 - alpha))
        return K_scale * (K_0 + ret)

    return RadialProfile(_entr, name=inspect.stack()[0][3])


def walker_entropy_profile(r_200, A, B, K_scale, alpha=1.1):
    """Generate a walker entropy profile."""

    def _entr(r):
        x = r / r_200
        return K_scale * (A * x**alpha) * np.exp(-((x / B) ** 2))

    return RadialProfile(_entr, name=inspect.stack()[0][3])


def rescale_profile_by_mass(profile, mass, radius):
    """
    Rescale a density ``profile`` by a total ``mass``
    within some ``radius``.

    Parameters
    ----------
    profile : ``RadialProfile`` object
        The profile object to rescale.
    mass : float
        The input mass in Msun.
    radius : float
        The input radius that the input ``mass`` corresponds to in kpc.

    """
    from scipy.integrate import quad

    mass_int = lambda r: profile(r) * r * r
    rescale = mass / (4.0 * np.pi * quad(mass_int, 0.0, radius)[0])
    return rescale * profile


def find_overdensity_radius(m, delta, z=0.0, cosmo=None):
    """
    Given a mass value and an overdensity, find the radius
    that corresponds to that enclosed mass.

    Parameters
    ----------
    m : float
        The enclosed mass.
    delta : float
        The overdensity to compute the radius for.
    z : float, optional
        The redshift of the halo formation. Default: 0.0
    cosmo : yt ``Cosmology`` object
        The cosmology to be used when computing the critical
        density. If not supplied, a default one from yt will
        be used.

    """
    from yt.utilities.cosmology import Cosmology

    if cosmo is None:
        cosmo = Cosmology()
    rho_crit = cosmo.critical_density(z).to_value("Msun/kpc**3")
    return (3.0 * m / (4.0 * np.pi * delta * rho_crit)) ** (1.0 / 3.0)


def find_radius_mass(m_r, delta, z=0.0, cosmo=None):
    """
    Given a mass profile and an overdensity, find the radius
    and mass (e.g. M200, r200)

    Parameters
    ----------
    m_r : RadialProfile
        The mass profile.
    delta : float
        The overdensity to compute the mass and radius for.
    z : float, optional
        The redshift of the halo formation. Default: 0.0
    cosmo : yt ``Cosmology`` object
        The cosmology to be used when computing the critical
        density. If not supplied, a default one from yt will
        be used.

    """
    from scipy.optimize import bisect
    from yt.utilities.cosmology import Cosmology

    if cosmo is None:
        cosmo = Cosmology()
    rho_crit = cosmo.critical_density(z).to_value("Msun/kpc**3")
    f = lambda r: 3.0 * m_r(r) / (4.0 * np.pi * r**3) - delta * rho_crit
    r_delta = bisect(f, 0.01, 10000.0)
    return r_delta, m_r(r_delta)
