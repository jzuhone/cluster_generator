"""
Analytic unit tests for cluster_generator.substructure.

Three cases that have closed-form answers:

1. Pressure-matched truncation in a uniform-pressure host.
2. tanh blend: correct limits and monotonic weight.
3. Overlap arbitration: both rules select the correct subhalo.
"""

from collections import OrderedDict

import numpy as np
import pytest
from unyt import unyt_array

from cluster_generator.model import ClusterModel
from cluster_generator.substructure import (
    CompositeCluster,
    find_pressure_truncation_radius,
)

# Fine radius grid used by all mock models (1 to 10000 kpc)
RR = np.logspace(0, 4, 2000)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock(rr, density, pressure, total_mass=None, gravitational_potential=None):
    """Build a minimal ClusterModel from raw arrays (in native units)."""
    fields = OrderedDict()
    fields["radius"] = unyt_array(rr, "kpc")
    fields["density"] = unyt_array(density, "Msun/kpc**3")
    fields["pressure"] = unyt_array(pressure, "Msun/kpc/Myr**2")
    if total_mass is not None:
        fields["total_mass"] = unyt_array(total_mass, "Msun")
    if gravitational_potential is not None:
        fields["gravitational_potential"] = unyt_array(gravitational_potential, "kpc**2/Myr**2")
    return ClusterModel.from_arrays(fields)


def _nfw_potential(rr, phi_s, r_s):
    """Simple analytic NFW-like potential: Φ(r) = -phi_s / (1 + r/r_s)."""
    return -phi_s / (1.0 + rr / r_s)


# ---------------------------------------------------------------------------
# 1. Pressure-matched truncation in a uniform-pressure host
# ---------------------------------------------------------------------------


class TestPressureTruncationRadius:
    """
    Host: P_host = P0 (constant).
    Subhalo: P_sub(r) = P_center * exp(-r / r_s).

    Analytic truncation radius: r_trunc = r_s * ln(P_center / P0).
    """

    P0 = 1.0e-4  # Msun/kpc/Myr^2
    P_center = 1.0e-3  # 10x host pressure at subhalo center
    r_s = 200.0  # kpc
    pos = np.array([2000.0, 0.0, 0.0])  # subhalo center in host frame

    @pytest.fixture
    def host(self):
        return _mock(RR, np.full_like(RR, 1.0e-3), np.full_like(RR, self.P0))

    @pytest.fixture
    def subhalo(self):
        P = self.P_center * np.exp(-RR / self.r_s)
        rho = 1.0e-2 * np.exp(-RR / self.r_s)
        return _mock(RR, rho, P)

    def test_truncation_matches_analytic(self, host, subhalo):
        r_trunc = find_pressure_truncation_radius(subhalo, host, self.pos)
        r_analytic = self.r_s * np.log(self.P_center / self.P0)
        assert abs(float(r_trunc.v) - r_analytic) / r_analytic < 1.0e-3

    def test_pressure_continuous_at_truncation(self, host, subhalo):
        r_trunc = find_pressure_truncation_radius(subhalo, host, self.pos)
        P_sub_at_trunc = float(
            subhalo.find_field_at_radius("pressure", r_trunc).in_units("Msun/kpc/Myr**2").v
        )
        assert abs(P_sub_at_trunc - self.P0) / self.P0 < 1.0e-3

    def test_weak_subhalo_raises(self, host):
        """Central pressure at or below host should raise ValueError."""
        rho = np.full_like(RR, 1.0e-3)
        P_low = np.full_like(RR, self.P0 * 0.5)
        weak_sub = _mock(RR, rho, P_low)
        with pytest.raises(ValueError, match="central pressure"):
            find_pressure_truncation_radius(weak_sub, host, self.pos)


# ---------------------------------------------------------------------------
# 2. tanh blend: limits and monotone weight
# ---------------------------------------------------------------------------


class TestTanhBlend:
    """
    Single subhalo at (1000, 0, 0) kpc with r_trunc = 300 kpc.
    Host pressure is flat; subhalo pressure is an exponential.
    """

    r_trunc = 300.0  # kpc
    blend_width = 20.0  # kpc — narrow so limits are easy to hit
    P_host = 1.0e-4
    rho_host = 1.0e-3
    P_center = 1.0e-3
    rho_center = 1.0e-2
    r_s_sub = 200.0  # kpc
    sub_pos = np.array([1000.0, 0.0, 0.0])

    @pytest.fixture
    def composite(self):
        host = _mock(
            RR,
            np.full_like(RR, self.rho_host),
            np.full_like(RR, self.P_host),
        )
        P_sub = self.P_center * np.exp(-RR / self.r_s_sub)
        rho_sub = self.rho_center * np.exp(-RR / self.r_s_sub)
        subhalo = _mock(RR, rho_sub, P_sub)
        return CompositeCluster(
            host=host,
            host_center=[0.0, 0.0, 0.0],
            subhalos=[subhalo],
            subhalo_positions=[self.sub_pos],
            truncation_radii=[self.r_trunc],
            blend_width=self.blend_width,
            overlap_rule="nearest_normalized",
        )

    def test_deep_inside_approximates_subhalo(self, composite):
        # r_sub = 1 kpc, weight ≈ 1 for r_trunc / blend_width = 15
        r_from_sub = 1.0
        x = self.sub_pos[0] + r_from_sub
        result = composite(x, 0.0, 0.0)
        expected_P = self.P_center * np.exp(-r_from_sub / self.r_s_sub)
        assert abs(float(result["pressure"].v[0]) - expected_P) / expected_P < 0.01

    def test_far_outside_approximates_host(self, composite):
        # r_sub = r_trunc + 8*blend_width: tanh weight ≈ exp(-8) ≈ 3e-4
        r_from_sub = self.r_trunc + 8.0 * self.blend_width
        x = self.sub_pos[0] + r_from_sub
        result = composite(x, 0.0, 0.0)
        assert abs(float(result["pressure"].v[0]) - self.P_host) / self.P_host < 0.01

    def test_weight_is_monotone_decreasing(self, composite):
        # Along the x-axis from sub center outward, the tanh weight in [0,1]
        # must decrease monotonically.  We verify this via the pressure: inside
        # r_trunc, P_blend = (1-w)*P_host + w*P_sub; since both P_sub and w
        # decrease with r_sub, and P_sub > P_host inside r_trunc, P_blend also
        # decreases.  We sample only the interval [0, r_trunc].
        n = 300
        r_vals = np.linspace(0.5, self.r_trunc - 5.0, n)
        x_vals = self.sub_pos[0] + r_vals
        result = composite(x_vals, np.zeros(n), np.zeros(n))
        P = result["pressure"].v
        diffs = np.diff(P)
        # Allow a tiny numerical tolerance
        assert np.all(diffs <= 1.0e-12), "Pressure not monotonically decreasing inside truncation sphere"

    def test_subhalo_id_inside(self, composite):
        result = composite(self.sub_pos[0] + 1.0, 0.0, 0.0)
        assert result["subhalo_id"][0] == 0

    def test_subhalo_id_outside(self, composite):
        r_far = self.r_trunc + 8.0 * self.blend_width
        result = composite(self.sub_pos[0] + r_far, 0.0, 0.0)
        assert result["subhalo_id"][0] == -1


# ---------------------------------------------------------------------------
# 3. Overlap arbitration: two subhalos, correct winner under both rules
# ---------------------------------------------------------------------------


class TestOverlapArbitration:
    """
    Two subhalos at x = -100 and +200 kpc, same r_trunc = 300 kpc.
    Query at origin (0, 0, 0):
      r1 = 100 kpc,  r1/r_trunc = 1/3
      r2 = 200 kpc,  r2/r_trunc = 2/3
    Sub1 is closer in normalized distance and has a deeper potential at
    that distance, so it should win under both rules.
    """

    r_trunc = 300.0
    blend_width = 30.0
    P_host = 1.0e-4
    rho_host = 1.0e-3
    phi_s = 1.0e4  # kpc^2/Myr^2 — scale depth for mock potential
    r_s_phi = 150.0  # kpc — potential scale radius

    @pytest.fixture
    def two_subhalo_composite(self):
        host = _mock(
            RR,
            np.full_like(RR, self.rho_host),
            np.full_like(RR, self.P_host),
        )

        def make_sub():
            P = 5.0e-4 * np.exp(-RR / 200.0)  # higher than host near center
            rho = 1.0e-2 * np.exp(-RR / 200.0)
            phi = _nfw_potential(RR, self.phi_s, self.r_s_phi)
            return _mock(RR, rho, P, gravitational_potential=phi)

        sub1 = make_sub()
        sub2 = make_sub()
        pos1 = [-100.0, 0.0, 0.0]
        pos2 = [200.0, 0.0, 0.0]
        return sub1, sub2, pos1, pos2, host

    def test_nearest_normalized_picks_closer(self, two_subhalo_composite):
        sub1, sub2, pos1, pos2, host = two_subhalo_composite
        cc = CompositeCluster(
            host=host,
            host_center=[0.0, 0.0, 0.0],
            subhalos=[sub1, sub2],
            subhalo_positions=[pos1, pos2],
            truncation_radii=[self.r_trunc, self.r_trunc],
            blend_width=self.blend_width,
            overlap_rule="nearest_normalized",
        )
        result = cc(0.0, 0.0, 0.0)
        assert result["subhalo_id"][0] == 0, "Sub1 (closer in r/r_trunc) should win"

    def test_deepest_potential_picks_deeper(self, two_subhalo_composite):
        sub1, sub2, pos1, pos2, host = two_subhalo_composite
        cc = CompositeCluster(
            host=host,
            host_center=[0.0, 0.0, 0.0],
            subhalos=[sub1, sub2],
            subhalo_positions=[pos1, pos2],
            truncation_radii=[self.r_trunc, self.r_trunc],
            blend_width=self.blend_width,
            overlap_rule="deepest_potential",
        )
        result = cc(0.0, 0.0, 0.0)
        # Sub1 at 100 kpc: Φ = -phi_s/(1+100/r_s) < Sub2 at 200 kpc: Φ = -phi_s/(1+200/r_s)
        assert result["subhalo_id"][0] == 0, "Sub1 (deeper potential) should win"

    def test_invalid_overlap_rule_raises(self, two_subhalo_composite):
        sub1, sub2, pos1, pos2, host = two_subhalo_composite
        with pytest.raises(ValueError, match="overlap_rule"):
            CompositeCluster(
                host=host,
                host_center=[0.0, 0.0, 0.0],
                subhalos=[sub1, sub2],
                subhalo_positions=[pos1, pos2],
                truncation_radii=[self.r_trunc, self.r_trunc],
                overlap_rule="bad_rule",
            )
