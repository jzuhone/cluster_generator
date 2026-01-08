from yt.fields.field_info_container import FieldInfoContainer
from yt.utilities.physical_constants import kboltz, mh

# We need to specify which fields we might have in our dataset.  The field info
# container subclass here will define which fields it knows about.  There are
# optionally methods on it that get called which can be subclassed.

b_units = "code_magnetic"
pres_units = "code_mass/(code_length*code_time**2)"
en_units = "code_mass * (code_length/code_time)**2"
rho_units = "code_mass / code_length**3"
mom_units = "code_mass/(code_length**2 * code_time)"


def velocity_field(j):
    def _velocity(field, data):
        return data["cluster_generator", f"momentum_density_{j}"] / data["cluster_generator", "density"]

    return _velocity


class ClusterGeneratorFieldInfo(FieldInfoContainer):
    # Tells about base units.
    known_other_fields = (
        ("density", (rho_units, ["density"], None)),
        ("momentum_density_x", (mom_units, [], None)),
        ("momentum_density_y", (mom_units, [], None)),
        ("momentum_density_z", (mom_units, [], None)),
        ("pressure", (pres_units, ["pressure"], None)),
        ("stellar_density", (rho_units, [], None)),
        ("dark_matter_density", (rho_units, [], None)),
    )

    known_particle_fields = (
        # Identical form to above
        # ( "name", ("units", ["fields", "to", "alias"], # "display_name")),
    )

    def __init__(self, ds, field_list):
        super().__init__(ds, field_list)
        # If you want, you can check self.field_list

    def setup_fluid_fields(self):
        unit_system = self.ds.unit_system

        # -- Adding velocity fields -- #
        # Necessary because we only store the momentum density, so it needs to be
        # converted back into a viable velocity field.
        for comp in self.ds.coordinates.axis_order:
            vel_field = ("cluster_generator", f"velocity_{comp}")
            mom_field = ("cluster_generator", f"momentum_density_{comp}")

            # Add the momentum field as an output field.
            self.add_output_field(
                mom_field,
                sampling_type="cell",
                units="code_mass/code_time/code_length**2",
            )
            # Add the velocity field by dividing out the density.
            self.add_field(
                vel_field,
                sampling_type="cell",
                function=velocity_field(comp),
                units=unit_system["velocity"],
            )
            self.alias(
                ("gas", f"momentum_density_{comp}"),
                mom_field,
                units="code_mass/code_time/code_length**2",
            )
            self.alias(
                ("gas", f"velocity_{comp}"),
                vel_field,
                units=unit_system["velocity"],
            )
            self.alias(
                ("dark_matter", f"velocity_{comp}"),
                vel_field,
                units=unit_system["velocity"],
            )
            self.alias(
                ("stellar", f"velocity_{comp}"),
                vel_field,
                units=unit_system["velocity"],
            )

        # -- Adding Temperature Fields -- #
        # we assume an ideal gas EOS, so T = m_p*mu*P/(rho*k)
        self.alias(
            ("gas", "pressure"),
            ("cluster_generator", "pressure"),
            units=unit_system["pressure"],
        )

        def _specific_thermal_energy(field, data):
            return (3 / 2) * (data["cluster_generator", "pressure"] / data["cluster_generator", "rho"])

        self.add_field(
            ("gas", "specific_thermal_energy"),
            sampling_type="cell",
            function=_specific_thermal_energy,
            units=unit_system["specific_energy"],
        )

        # Add temperature field
        def _temperature(field, data):
            return (data["gas", "pressure"] / data["gas", "density"]) * data.ds.mu * mh / kboltz

        self.add_field(
            ("gas", "temperature"),
            sampling_type="cell",
            function=_temperature,
            units=unit_system["temperature"],
        )

    def setup_particle_fields(self, ptype):
        super().setup_particle_fields(ptype)
        # This will get called for every particle type.
