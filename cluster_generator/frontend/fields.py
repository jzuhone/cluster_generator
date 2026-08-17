"""
Field aliases for :class:`ClusterGeneratorDataset`.

Adds the ``gas``/``dark_matter``/``stellar`` velocity fields (derived from
the on-disk momentum density) and a ``gas`` temperature field on top of the
raw ``boxlib``-typed fields written by
:class:`cluster_generator.datasets.AMRClusterDataset`, the same
derived fields the old (pre-pyAMReX) custom frontend provided.
"""

from yt.frontends.amrex.api import BoxlibFieldInfo
from yt.utilities.physical_constants import kboltz, mh

rho_units = "code_mass / code_length**3"
mom_units = "code_mass / (code_time * code_length**2)"
pres_units = "code_mass / (code_length * code_time**2)"


def _velocity_field(axis):
    def _velocity(field, data):
        return data["boxlib", f"momentum_density_{axis}"] / data["boxlib", "density"]

    return _velocity


class ClusterGeneratorFieldInfo(BoxlibFieldInfo):
    known_other_fields = (
        ("density", (rho_units, ["density"], None)),
        ("dark_matter_density", (rho_units, [], None)),
        ("stellar_density", (rho_units, [], None)),
        ("pressure", (pres_units, ["pressure"], None)),
        ("momentum_density_x", (mom_units, [], None)),
        ("momentum_density_y", (mom_units, [], None)),
        ("momentum_density_z", (mom_units, [], None)),
        ("magnetic_pressure", (pres_units, [], None)),
    )

    def setup_fluid_fields(self):
        unit_system = self.ds.unit_system

        for axis in self.ds.coordinates.axis_order:
            vel_field = ("gas", f"velocity_{axis}")
            self.add_field(
                vel_field,
                sampling_type="cell",
                function=_velocity_field(axis),
                units=unit_system["velocity"],
            )
            for ptype in ("dark_matter", "stellar"):
                self.alias((ptype, f"velocity_{axis}"), vel_field, units=unit_system["velocity"])

        def _temperature(field, data):
            return (data["gas", "pressure"] / data["gas", "density"]) * data.ds.mu * mh / kboltz

        self.add_field(
            ("gas", "temperature"),
            sampling_type="cell",
            function=_temperature,
            units=unit_system["temperature"],
        )

        super().setup_fluid_fields()
