"""
Test Suite Utility for Generating Cluster Models.

This module provides utility functions to generate test models for cluster simulations.
It utilizes various radial profile functions to compute density and mass profiles
based on specified parameters such as redshift, mass, and concentration. The resulting
model is configured with density profiles for dark matter, gas, and stars, and includes
a magnetic field setup.

Functions
---------
generate_model() -> ClusterModel
    Generates a `ClusterModel` using specified cosmological and density parameters.

Examples
--------
To generate a test cluster model for simulations:

    >>> model = generate_model()
    >>> print(model)

"""


from cluster_generator.model import ClusterModel
from cluster_generator.radial_profiles import (
    find_overdensity_radius,
    find_radius_mass,
    rescale_profile_by_mass,
    snfw_density_profile,
    snfw_mass_profile,
    snfw_total_mass,
    vikhlinin_density_profile,
)


def generate_model() -> ClusterModel:
    """
    Generate a test cluster model with specific cosmological and density parameters.

    This function creates a `ClusterModel` instance using predefined parameters
    for redshift, mass, concentration, and other properties. It computes the
    density and mass profiles for dark matter, gas, and stars, and sets up a
    magnetic field using a beta profile.

    Returns
    -------
    ClusterModel
        The generated cluster model configured with density profiles and
        magnetic field settings.

    Examples
    --------
    >>> model = generate_model()
    >>> print(model)

    Notes
    -----
    - The function assumes a specific cosmological context and uses a
      concentration parameter for the NFW profile.
    - Gas density is rescaled based on the computed M500 value and a
      specified gas fraction.

    Raises
    ------
    ValueError
        If any of the profile computations fail or result in invalid values.
    """
    # Define cosmological and profile parameters
    z = 0.1  # Redshift
    M200 = 1.5e15  # Mass enclosed within r200 (Msun)
    conc = 4.0  # Concentration parameter for NFW profile
    f_g = 0.12  # Gas fraction

    # Compute overdensity radius and scale radius for NFW profile
    r200 = find_overdensity_radius(M200, 200.0, z=z)
    a = r200 / conc

    # Calculate total mass for super-NFW profile and corresponding density and mass profiles
    M = snfw_total_mass(M200, r200, a)
    rhot = snfw_density_profile(M, a)
    Mt = snfw_mass_profile(M, a)

    # Find radius and mass for a different overdensity (500 times critical density)
    r500, M500 = find_radius_mass(Mt, z=z, delta=500.0)

    # Define gas density profile and rescale to match gas fraction of total mass within r500
    rhog = vikhlinin_density_profile(1.0, 100.0, r200, 1.0, 0.67, 3)
    rhog = rescale_profile_by_mass(rhog, f_g * M500, r500)

    # Define stellar density profile as a fraction of the total dark matter density
    rhos = 0.02 * rhot

    # Define minimum and maximum radii for the model
    rmin = 0.1
    rmax = 10000.0

    # Generate the cluster model with the computed density profiles
    m = ClusterModel.from_dens_and_tden(rmin, rmax, rhog, rhot, stellar_density=rhos)

    # Set up a magnetic field profile based on beta parameter
    m.set_magnetic_field_from_beta(100.0, gaussian=True)

    return m
