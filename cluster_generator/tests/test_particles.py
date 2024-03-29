from numpy.random import RandomState

from cluster_generator.tests.utils import generate_model, particle_answer_testing

prng = RandomState(25)


def test_particles(answer_store, answer_dir):
    m = generate_model()
    dp = m.generate_dm_particles(100000, prng=prng)
    sp = m.generate_star_particles(100000, prng=prng)
    hp = m.generate_gas_particles(100000, prng=prng)
    parts = hp + dp + sp
    particle_answer_testing(parts, "particles.h5", answer_store, answer_dir)
