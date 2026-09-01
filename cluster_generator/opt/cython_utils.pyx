#cython: language_level=3, boundscheck=False
"""
Cythonized utilities for equilibrium models
"""

#-----------------------------------------------------------------------------
# Copyright (c) 2013, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

import numpy as np

cimport cython
cimport numpy as np

np.import_array() # --> fix numpy error at runtime for not having it. Why do we need this?

from tqdm.auto import tqdm


cdef extern from "math.h":
    double sqrt(double x) nogil
    double log10(double x) nogil
    double fmod(double numer, double denom) nogil
    double sin(double x) nogil

cdef extern from "stdlib.h":
    double drand48() nogil
    void srand48(long int seedval) nogil

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

CTYPE = np.complex128
ctypedef np.complex128_t CTYPE_t

# Maximum supported spline degree for the fixed-size de Boor work buffer
# below. Splines used throughout this package are cubic (k=3), so this
# leaves a generous margin.
DEF MAX_SPLINE_DEGREE = 15

@cython.cdivision(True)
cdef inline DTYPE_t bspline_eval(double *t, double *c, int n_knots, int k,
                                  DTYPE_t x) nogil:
    """
    Evaluate, at the scalar point `x`, the B-spline of degree `k` defined
    by the full FITPACK-style knot array `t` (length `n_knots`) and
    coefficient array `c`, using de Boor's algorithm.

    This reproduces the zero-order, extrapolating evaluation that
    ``scipy.interpolate.BSpline(t, c, k, extrapolate=True)`` gives, but
    depends only on the mathematical definition of a B-spline rather than
    on any private scipy/FITPACK interface, so it cannot be broken by
    scipy internals changing across versions.
    """
    cdef int hi = n_knots - k - 2
    cdef int i = k
    cdef int j, r
    cdef DTYPE_t d[MAX_SPLINE_DEGREE + 1]
    cdef DTYPE_t alpha

    while i < hi and x >= t[i + 1]:
        i += 1

    for j in range(k + 1):
        d[j] = c[i - k + j]

    for r in range(1, k + 1):
        for j in range(k, r - 1, -1):
            alpha = (x - t[j + i - k]) / (t[j + i - r + 1] - t[j + i - k])
            d[j] = (1.0 - alpha) * d[j - 1] + alpha * d[j]

    return d[k]


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def generate_velocities(np.ndarray[DTYPE_t, ndim=1] psi,
                        np.ndarray[DTYPE_t, ndim=1] vesc,
                        np.ndarray[DTYPE_t, ndim=1] fv2esc,
                        np.ndarray[DTYPE_t, ndim=1] t,
                        np.ndarray[DTYPE_t, ndim=1] c,
                        int k,
                        int pbar_status):
    cdef DTYPE_t v2, e, f
    cdef np.uint8_t not_done
    cdef unsigned int i
    cdef int num_particles, n_knots
    cdef long int seedval
    cdef np.ndarray[np.float64_t, ndim=1] velocity
    cdef double *t_ptr = &t[0]
    cdef double *c_ptr = &c[0]

    if k > MAX_SPLINE_DEGREE:
        raise ValueError(f"Spline degree k={k} exceeds the maximum supported degree "
                          f"({MAX_SPLINE_DEGREE}).")

    seedval = -100
    srand48(seedval)
    n_knots = t.shape[0]
    num_particles = psi.shape[0]
    velocity = np.zeros(num_particles, dtype='float64')
    pbar = tqdm(leave=True, total=num_particles,
                desc="Generating particle velocities ",
                disable=(pbar_status==1))
    for i in range(num_particles):
        not_done = 1
        while not_done:
            v2 = drand48()*vesc[i]
            v2 *= v2
            e = psi[i]-0.5*v2
            f = bspline_eval(t_ptr, c_ptr, n_knots, k, e)
            not_done = f*v2 < drand48()*fv2esc[i]
        velocity[i] = sqrt(v2)
        pbar.update()
    pbar.close()
    return velocity


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def div_clean(np.ndarray[CTYPE_t, ndim=3] gx,
                     np.ndarray[CTYPE_t, ndim=3] gy,
                     np.ndarray[CTYPE_t, ndim=3] gz,
                     np.ndarray[DTYPE_t, ndim=1] kx,
                     np.ndarray[DTYPE_t, ndim=1] ky,
                     np.ndarray[DTYPE_t, ndim=1] kz,
                     np.ndarray[DTYPE_t, ndim=1] deltas):

    cdef int i, j, k
    cdef int nx, ny, nz
    cdef DTYPE_t kxd, kyd, kzd, kkd
    cdef CTYPE_t ggx, ggy, ggz, kg

    nx = gx.shape[0]
    ny = gx.shape[1]
    nz = gx.shape[2]

    # These k's are different because we are
    # using the finite difference form of the
    # divergence operator.
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                ggx = gx[i,j,k]
                ggy = gy[i,j,k]
                ggz = gz[i,j,k]
                kxd = sin(kx[i] * deltas[0]) / deltas[0]
                kyd = sin(ky[j] * deltas[1]) / deltas[1]
                kzd = sin(kz[k] * deltas[2]) / deltas[2]
                kkd = sqrt(kxd*kxd + kyd*kyd + kzd*kzd)
                if kkd > 0:
                    kxd /= kkd
                    kyd /= kkd
                    kzd /= kkd
                kg = kxd * ggx + kyd * ggy + kzd * ggz
                gx[i,j,k] = ggx - kxd * kg
                gy[i,j,k] = ggy - kyd * kg
                gz[i,j,k] = ggz - kzd * kg
