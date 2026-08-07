"""
Unit tests for the AREPO mesh/volume/mass helpers in
:mod:`cluster_generator.codes`.

The mesh relaxation and cell-integrated mass use a streaming grid (O(N)
memory, scales to tens of millions of cells); the exact Voronoi *volumes* used
for mass assignment come from voro++ (``pyvoro2``).  Tests that need voro++ are
skipped when it is not installed.  Everything runs on a synthetic density field
and needs none of the answer-testing fixtures.
"""

import numpy as np
import pytest

from cluster_generator.codes import (
    _compute_arepo_masses,
    _exact_voronoi_volumes,
    _lloyd_relax,
    _stream_grid_moments,
)

BOXSIZE = 100.0


def _peaked_density(p):
    """Steep, centrally-peaked density in Msun/kpc**3 (never zero)."""
    r = np.sqrt(((np.asarray(p) - BOXSIZE / 2) ** 2).sum(axis=1))
    return 1.0e6 / (1.0 + (r / 8.0) ** 2) + 1.0e2


def _sample_positions(n=2000, seed=0):
    """Positions concentrated toward the box centre, all inside the box."""
    rng = np.random.default_rng(seed)
    chunks = []
    while sum(len(c) for c in chunks) < n:
        p = rng.uniform(0, BOXSIZE, size=(n, 3))
        r2 = ((p - BOXSIZE / 2) ** 2).sum(axis=1)
        chunks.append(p[rng.uniform(size=n) < 1.0 / (1.0 + r2 / 200.0)])
    return np.vstack(chunks)[:n]


def _have_voro():
    try:
        __import__("pyvoro2")
        return True
    except ImportError:
        return False


requires_voro = pytest.mark.skipif(not _have_voro(), reason="pyvoro2 not installed")


def test_compute_arepo_masses_rejects_bad_mass_method():
    """A bad mass_method raises before any tessellation (no voro++ needed)."""
    pos = _sample_positions(n=50)
    with pytest.raises(ValueError):
        _compute_arepo_masses(pos, BOXSIZE, _peaked_density, _peaked_density(pos), mass_method="bogus")


def test_stream_grid_moments_partition_and_chunk_invariance():
    """
    Grid-count volumes tile the box, weighted centroids are interior, and the
    result is independent of the streaming chunk size (memory-scaling knob).
    """
    pos = _sample_positions(n=1500, seed=2)
    count, wsum, wpos, dx = _stream_grid_moments(pos, BOXSIZE, 8, _peaked_density, chunk_size=1_000_000)
    vol_grid = count * dx**3
    np.testing.assert_allclose(vol_grid.sum(), BOXSIZE**3, rtol=1e-6)

    owned = wsum > 0
    centroids = np.where(owned[:, None], wpos / np.where(owned, wsum, 1.0)[:, None], pos)
    assert (centroids >= 0).all() and (centroids <= BOXSIZE).all()

    # Chunking must not change the answer.
    c2, w2, wp2, dx2 = _stream_grid_moments(pos, BOXSIZE, 8, _peaked_density, chunk_size=97)
    assert dx == dx2
    np.testing.assert_allclose(count, c2)
    np.testing.assert_allclose(wsum, w2)
    np.testing.assert_allclose(wpos, wp2)


def test_lloyd_relax_regularizes_mesh():
    """
    The streaming relaxation must reduce the spread of cell volumes (a more
    regular mesh) and keep every point inside the box.  Needs no voro++.
    """

    def vol_cov(p):
        count, _, _, dx = _stream_grid_moments(p, BOXSIZE, 8, None, chunk_size=1_000_000)
        v = count * dx**3
        return v.std() / v.mean()

    pos = _sample_positions(n=1500, seed=3)
    cov0 = vol_cov(pos)
    relaxed = _lloyd_relax(pos, BOXSIZE, num_iterations=8, step_damping=1.0, tol=None)
    assert vol_cov(relaxed) < cov0
    assert (relaxed >= 0).all() and (relaxed <= BOXSIZE).all()
    assert relaxed.shape == pos.shape


@requires_voro
def test_exact_voronoi_volumes_partition_box():
    """Exact bounded Voronoi volumes (cell_measures) tile the box."""
    pos = _sample_positions()
    vols = _exact_voronoi_volumes(pos, BOXSIZE)
    assert (vols > 0).all()
    np.testing.assert_allclose(vols.sum(), BOXSIZE**3, rtol=1e-6)


@requires_voro
def test_compute_arepo_masses_point_is_rho_times_exact_volume():
    pos = _sample_positions(n=1500, seed=5)
    m = _compute_arepo_masses(pos, BOXSIZE, _peaked_density, _peaked_density(pos), "point")
    np.testing.assert_allclose(m, _peaked_density(pos) * _exact_voronoi_volumes(pos, BOXSIZE))


@requires_voro
def test_integrated_beats_point_reconstruction():
    """
    AREPO reconstructs density = mass / V_exact.  In a steep gradient the
    cell-integrated mass ('integrated') reproduces the true cell-averaged
    density better than the point value ('point').
    """
    from scipy.spatial import cKDTree

    pos = _sample_positions(n=1500, seed=6)
    v_exact = _exact_voronoi_volumes(pos, BOXSIZE)
    m_int = _compute_arepo_masses(pos, BOXSIZE, _peaked_density, _peaked_density(pos), "integrated")
    rho_integrated = m_int / v_exact
    rho_point = _peaked_density(pos)

    rng = np.random.default_rng(9)
    samples = rng.uniform(0, BOXSIZE, size=(2_000_000, 3))
    owner = cKDTree(pos).query(samples, workers=-1)[1]
    counts = np.bincount(owner, minlength=len(pos))
    sums = np.bincount(owner, weights=_peaked_density(samples), minlength=len(pos))
    well = counts >= 200
    rho_true = sums[well] / counts[well]

    err_int = np.median(np.abs(rho_integrated[well] / rho_true - 1.0))
    err_pt = np.median(np.abs(rho_point[well] / rho_true - 1.0))
    assert err_int < err_pt, f"integrated ({err_int:.3f}) not better than point ({err_pt:.3f})"

    assert np.isfinite(m_int).all() and (m_int > 0).all()


@pytest.mark.slow
def test_setup_gizmo_ics(tmp_path, temp_dir):
    """
    GIZMO ICs must be equal-mass (meshless), carry strictly positive internal
    energy everywhere (incl. the far background — the ext=3 resample fix), and
    carry no Voronoi volumes / AREPO config.
    """
    import h5py
    from numpy.random import RandomState

    from cluster_generator.codes import setup_gizmo_ics
    from cluster_generator.ics import ClusterICs
    from cluster_generator.tests.utils import get_base_model_path

    model_path = get_base_model_path(temp_dir)
    boxsize = 40000.0  # must contain the cluster (r_max = 20000 kpc)
    ics = ClusterICs(
        "single",
        1,
        model_path,
        [boxsize / 2] * 3,
        [0.0, 0.0, 0.0],
        num_particles=dict.fromkeys(["dm", "star", "gas"], 5000),
    )
    out = str(tmp_path / "gizmo.hdf5")
    setup_gizmo_ics(
        ics, boxsize, 24, 1.0e-30, out, overwrite=True, num_lloyd_iterations=2, prng=RandomState(25)
    )

    with h5py.File(out, "r") as f:
        g = f["PartType0"]
        m = g["Masses"][:]
        u = g["InternalEnergy"][:]
        assert np.allclose(m, m[0]), "GIZMO particles must be equal-mass"
        assert (u > 0).all(), "internal energy must be positive everywhere (ext=3 resample)"
        assert "Volume" not in g, "GIZMO IC must not carry Voronoi volumes"
        assert "Config" not in f, "GIZMO IC must not carry the AREPO VORONOI config"
