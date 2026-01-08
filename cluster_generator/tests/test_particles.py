"""
Pytest suite for unit-testing the particle generation system.
"""

import pytest

from cluster_generator.tests.utils import get_base_model, particle_answer_testing, prng


@pytest.mark.slow
def test_particles(answer_store: bool, answer_dir: str, temp_dir: str):
    """
    Test model particle generation.
    """
    # fetching the base model
    m = get_base_model(temp_dir)

    # creating the particles
    dp = m.generate_dm_particles(100000, prng=prng)
    sp = m.generate_star_particles(100000, prng=prng)
    hp = m.generate_gas_particles(100000, prng=prng)
    parts = hp + dp + sp
    particle_answer_testing(parts, "model_particles.h5", answer_store, answer_dir)
