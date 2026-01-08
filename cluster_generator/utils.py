"""
Utility functions for basic functionality of the py:module:`cluster_generator` package.
"""

import logging
import os
import sys

import numpy as np
import pathlib as pt
import yaml
from more_itertools import always_iterable
from numpy.random import RandomState
from scipy.integrate import quad
from unyt import kpc, physical_constants as pc, unyt_array, unyt_quantity

try:
    from typing import Self  # noqa
except ImportError:
    from typing_extensions import Self  # noqa

# -- configuration directory -- #
_config_directory = os.path.join(pt.Path(__file__).parents[0], "bin", "config.yaml")


# defining the custom yaml loader for unit-ed objects
def _yaml_unit_constructor(loader: yaml.FullLoader, node: yaml.nodes.MappingNode):
    kw = loader.construct_mapping(node)
    i_s = kw["input_scalar"]
    del kw["input_scalar"]
    return unyt_array(i_s, **kw)


def _yaml_lambda_loader(loader: yaml.FullLoader, node: yaml.nodes.ScalarNode):
    return eval(loader.construct_scalar(node))


def _get_loader():
    loader = yaml.FullLoader
    loader.add_constructor("!unyt", _yaml_unit_constructor)
    loader.add_constructor("!lambda", _yaml_lambda_loader)
    return loader


try:
    with open(_config_directory, "r+") as config_file:
        cgparams = yaml.load(config_file, _get_loader())

except FileNotFoundError as er:
    raise FileNotFoundError(
        f"Couldn't find the configuration file! Is it at {_config_directory}? Error = {er.__repr__()}"
    ) from er
except yaml.YAMLError as er:
    raise yaml.YAMLError(f"The configuration file is corrupted! Error = {er.__repr__()}") from er


stream = sys.stdout if cgparams["system"]["logging"]["main"]["stream"] in ["STDOUT", "stdout"] else sys.stderr
cgLogger = logging.getLogger("cluster_generator")

cg_sh = logging.StreamHandler(stream=stream)

# create formatter and add it to the handlers
formatter = logging.Formatter(cgparams["system"]["logging"]["main"]["format"])
cg_sh.setFormatter(formatter)
# add the handler to the logger
cgLogger.addHandler(cg_sh)
cgLogger.setLevel(cgparams["system"]["logging"]["main"]["level"])
cgLogger.propagate = False

mylog = cgLogger

# -- Setting up the developer debugger -- #
devLogger = logging.getLogger("development_logger")

if cgparams["system"]["logging"]["developer"]["enabled"]:  # --> We do want to use the development logger.
    # -- checking if the user has specified a directory -- #
    if cgparams["system"]["logging"]["developer"]["output_directory"] is not None:
        from datetime import datetime

        dv_fh = logging.FileHandler(
            os.path.join(
                cgparams["system"]["logging"]["developer"]["output_directory"],
                f"{datetime.now().strftime('%m-%d-%y_%H-%M-%S')}.log",
            )
        )

        # adding the formatter
        dv_formatter = logging.Formatter(cgparams["system"]["logging"]["main"]["format"])

        dv_fh.setFormatter(dv_formatter)
        devLogger.addHandler(dv_fh)
        devLogger.setLevel("DEBUG")
        devLogger.propagate = False

    else:
        mylog.warning(
            "User enabled development logger but did not specify output directory. "
            "Dev logger will not be used."
        )
else:
    devLogger.propagate = False
    devLogger.disabled = True


mp = (pc.mp).to("Msun")
G = (pc.G).to("kpc**3/Msun/Myr**2")
kboltz = (pc.kboltz).to("Msun*kpc**2/Myr**2/K")
kpc_to_cm = (1.0 * kpc).to_value("cm")

X_H = cgparams["physics"]["hydrogen_abundance"]
mu = 1.0 / (2.0 * X_H + 0.75 * (1.0 - X_H))
mue = 1.0 / (X_H + 0.5 * (1.0 - X_H))


# -- Utility functions -- #
def _truncator_function(a, r, x):
    return 1 / (1 + (x / r) ** a)


def integrate_mass(profile, rr):
    def mass_int(r):
        return profile(r) * r * r

    mass = np.zeros(rr.shape)
    for i, r in enumerate(rr):
        mass[i] = 4.0 * np.pi * quad(mass_int, 0, r)[0]
    return mass


def integrate(profile, rr):
    ret = np.zeros(rr.shape)
    rmax = rr[-1]
    for i, r in enumerate(rr):
        ret[i] = quad(profile, r, rmax)[0]
    return ret


def integrate_toinf(profile, rr):
    ret = np.zeros(rr.shape)
    rmax = rr[-1]
    for i, r in enumerate(rr):
        ret[i] = quad(profile, r, rmax)[0]
    ret[:] += quad(profile, rmax, np.inf, limit=100)[0]
    return ret


def generate_particle_radii(r, m, num_particles, r_max=None, prng=None):
    prng = parse_prng(prng)
    if r_max is None:
        ridx = r.size
    else:
        ridx = np.searchsorted(r, r_max)
    mtot = m[ridx - 1]
    u = prng.uniform(size=num_particles)
    P_r = np.insert(m[:ridx], 0, 0.0)
    P_r /= P_r[-1]
    r = np.insert(r[:ridx], 0, 0.0)
    radius = np.interp(u, P_r, r, left=0.0, right=1.0)
    return radius, mtot


def ensure_ytquantity(x, default_units):
    if isinstance(x, unyt_quantity):
        return unyt_quantity(x.v, x.units).in_units(default_units)
    elif isinstance(x, tuple):
        return unyt_quantity(x[0], x[1]).in_units(default_units)
    else:
        return unyt_quantity(x, default_units)


def ensure_ytarray(arr, units):
    if not isinstance(arr, unyt_array):
        arr = unyt_array(arr, units)
    return arr.to(units)


def parse_prng(prng):
    if isinstance(prng, RandomState):
        return prng
    else:
        return RandomState(prng)


def ensure_list(x):
    return list(always_iterable(x))


field_label_map = {
    "density": "$\\rho_g$ (g cm$^{-3}$)",
    "temperature": "kT (keV)",
    "pressure": "P (erg cm$^{-3}$)",
    "entropy": "S (keV cm$^{2}$)",
    "dark_matter_density": "$\\rho_{\\rm DM}$ (g cm$^{-3}$)",
    "electron_number_density": "n$_e$ (cm$^{-3}$)",
    "stellar_mass": "M$_*$ (M$_\\odot$)",
    "stellar_density": "$\\rho_*$ (g cm$^{-3}$)",
    "dark_matter_mass": "$M_{\\rm DM}$ (M$_\\odot$)",
    "gas_mass": "M$_g$ (M$_\\odot$)",
    "total_mass": "M$_{\\rm tot}$ (M$_\\odot$)",
    "gas_fraction": "f$_{\\rm gas}$",
    "magnetic_field_strength": "B (G)",
    "gravitational_potential": "$\\Phi$ (kpc$^2$ Myr$^{-2}$)",
    "gravitational_field": "g (kpc Myr$^{-2}$)",
}
