"""
yt Dataset subclass for cluster_generator's AMReX plotfile output.

The plotfiles written by
:class:`cluster_generator.datasets.AMRClusterDataset` are real AMReX
plotfiles, so all of the grid indexing and I/O is handled natively by yt's
``amrex``/``boxlib`` frontend (:class:`~yt.frontends.amrex.data_structures.AMReXDataset`);
this subclass only adds cluster_generator's field aliases (see
:mod:`.fields`) and physical unit conventions (kpc, Msun, Myr).
"""

from yt.frontends.amrex.api import AMReXDataset

from .fields import ClusterGeneratorFieldInfo


class ClusterGeneratorDataset(AMReXDataset):
    _field_info_class = ClusterGeneratorFieldInfo
    _subtype_keyword = "cluster_generator"
    # "dark_matter"/"stellar" aliases (see .fields) need these registered as
    # fluid types up front -- BoxlibDataset.__init__ extends this tuple
    # before any field detection runs.
    fluid_types = AMReXDataset.fluid_types + ("dark_matter", "stellar")

    def _set_code_unit_attributes(self):
        self.length_unit = self.quan(1.0, "kpc")
        self.mass_unit = self.quan(1.0, "Msun")
        self.time_unit = self.quan(1.0, "Myr")
        self.velocity_unit = self.quan(1.0, "kpc/Myr")
        self.magnetic_unit = self.quan(1.0, "gauss")

    def _parse_parameter_file(self):
        super()._parse_parameter_file()
        self.mu = self.parameters.get("mu", 0.6)
