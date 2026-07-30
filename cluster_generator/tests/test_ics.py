"""
Testing module for initial conditions objects.
"""

import pytest

from cluster_generator.ics import ClusterICs, compute_centers_for_binary
from cluster_generator.tests.utils import (
    get_base_model_path,
    particle_answer_testing,
    prng,
)


@pytest.mark.slow
def test_single_ics(answer_dir: str, answer_store: bool, temp_dir: str):
    """
    Generate a 1-cluster IC object and create its particles. Then check against answers.
    """
    # -- Making sure that there is a model -- #
    # The model should be at temp_dir/base_model.h5. If not, we may need to generate a new one.
    base_model_path = get_base_model_path(temp_dir)

    # Configure and generate the ICs
    num_particles = dict.fromkeys(["dm", "star", "gas"], 100000)
    ics = ClusterICs(
        "single",
        1,
        base_model_path,
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        num_particles=num_particles,
    )

    # Generate the particles
    parts = ics.setup_particle_ics(prng=prng)

    # Perform the answer tests.
    particle_answer_testing(parts, "particles_single.h5", answer_store, answer_dir)


@pytest.mark.slow
def test_double_ics(answer_dir: str, answer_store: bool, temp_dir: str):
    """
    Generate a 2-cluster IC object and create its particles. Then check against answers.
    """
    # -- Making sure that there is a model -- #
    # The model should be at temp_dir/base_model.h5. If not, we may need to generate a new one.
    base_model_path = get_base_model_path(temp_dir)

    # Configure and generate the ICs
    num_particles = dict.fromkeys(["dm", "star", "gas"], 200000)
    center1, center2 = compute_centers_for_binary([0.0, 0.0, 0.0], 3000.0, 500.0)
    velocity1 = [500.0, 0.0, 0.0]
    velocity2 = [-500.0, 0.0, 0.0]
    ics = ClusterICs(
        "double",
        2,
        [base_model_path, base_model_path],
        [center1, center2],
        [velocity1, velocity2],
        num_particles=num_particles,
    )

    # Generate the particles
    parts = ics.setup_particle_ics(prng=prng)

    # Perform the answer tests.
    particle_answer_testing(parts, "particles_double.h5", answer_store, answer_dir)
