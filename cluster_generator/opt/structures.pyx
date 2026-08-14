#----------------------------------------------------------------------------------#
# Cython utility functions for data structure processes.
#
# Written by: Eliza Diggins
#----------------------------------------------------------------------------------#
# - structures.pyx provides various cython functions used for converting between
#   data structures and for building equivalent data types for use in external libraries.
#
#
import numpy as np

cimport cython
from cpython cimport array

import array

cimport numpy as np

np.import_array() # --> fix numpy error at runtime for not having it. Why do we need this?
try:
    # See https://github.com/scipy/scipy/issues/16729 and related issues. This may need continual updating as
    # Scipy continues to update its conventions for interfacing with Dierckx FITPACK.
    from scipy.interpolate import _dfitpack  # noqa
except ImportError:
    from scipy.interpolate import dfitpack as _dfitpack #noqa

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

ITYPE32 = np.uint32
ctypedef np.uint32_t ITYPE32_t

CTYPE = np.complex128
ctypedef np.complex128_t CTYPE_t


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def construct_chunks(
    np.ndarray[ITYPE32_t, ndim=1] domain_dimensions,
    unsigned long chunksize
):
    """
    Construct a chunk-map of a uniform grid with a given chunksize and domain dimensions.

    Parameters
    ----------
    domain_dimensions: np.ndarray[ITYPE32_t, ndim=1]
        The ``domain_dimensions`` are the number of grid cells along each of the relevant axes. These should be
        integer type (unsigned int 32-bit) and the array should have shape ``(3,)``. For example, ``[512,512,512]`` corresponds
        to a uniform grid with 512 grid cells along each axis.
    chunksize: unsigned long
        The size of each of the chunks in the chunkmap. The chunksize must evenly divide the total domain dimension along a
        given axis. None of the resulting chunks in the chunk map can be larger than ``chunksize**3``.

    Returns
    -------
    chunkmap:
        The output chunkmap. This is a ``(3,2,N)`` array corresponding to ``N`` total chunks in the dataset. for each chunk ``i``,
        ``chunkmap[:,:,i]`` is a ``(3,2)`` array for which each row represents an axis and each column represents the minimum / maximum
        value along that axis respectively. The chunkmap is returned in pixel units (same as ``domain_dimensions``.

    """
    #// Compute the number of necessary chunks.
    cdef unsigned long i,j,k,_i
    cdef np.ndarray[ITYPE32_t, ndim=1] nchunks = np.zeros(domain_dimensions.size,dtype="uint32")

    # Determine the number of required chunks along each axis.
    for _i in range(domain_dimensions.size):
        if domain_dimensions[_i]%chunksize == 0:
            # The chunksize evenly divides the domain.
            nchunks[_i] = domain_dimensions[_i]//chunksize
        else:
            # The chunksize doesn't evenly divide the domain.
            nchunks[_i] = domain_dimensions[_i]//chunksize + 1


    cdef np.ndarray[ITYPE32_t,ndim=5] chunks = np.zeros((3,2,nchunks[0],nchunks[1],nchunks[2]),dtype="uint32")
    cdef np.ndarray[ITYPE32_t,ndim=1] p = np.zeros(3,dtype="uint32")


    # Populate the chunkmap.
    for i in range(nchunks[0]):
        p[1:] = 0
        for j in range(nchunks[1]):
            p[2:] = 0
            for k in range(nchunks[2]):
                # figure out the i,j,k - th value
                chunks[:,0,i,j,k] = p
                chunks[:,1,i,j,k] = p+chunksize
                p[2] += chunksize
            chunks[2,1,i,j,nchunks[2]-1] = domain_dimensions[2]
            p[1] += chunksize
        chunks[1, 1, i,  nchunks[1] - 1,:] = domain_dimensions[1]
        p[0] += chunksize
    chunks[0, 1, nchunks[0] - 1,:,:] = domain_dimensions[0]

    return chunks.reshape((3,2,nchunks[0]*nchunks[1]*nchunks[2]))

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def dump_field_to_hdf5(
    buffer_object,
    np.ndarray[DTYPE_t, ndim=2] bbox,
    np.ndarray[ITYPE32_t, ndim=1] domain_dimensions,
    np.ndarray[ITYPE32_t, ndim=3] chunkmap,
    np.ndarray[DTYPE_t, ndim=1] t,
    np.ndarray[DTYPE_t, ndim=1] c,
    int k,
    str fieldname
):
    """
    Dump interpolated field data to HDF5 file.

    Parameters
    ----------
    buffer_object: HDF5 dataset
        The ``buffer_object`` is the HDF5 connection (via the C-level API) to which we write the relevant data during this
        process.
    bbox: np.ndarray[DTYPE_t, ndim=2]
        This is the ``(3,2)`` bounding box for the domain where values are assumed to be in kpc.
    domain_dimensions: np.ndarray[DTYPE_t,ndim=1]
        The domain dimensions ``(3,1)`` of the uniform grid.
    chunkmap: (3,2,N) array
        The map of chunk coordinates. There are N total chunks in the domain. For each chunk, the second axis corresponds
        to the lower and upper corner of the chunk and the first axis to the x,y,z position of each corner.
    t: Interpolation parameters.
    c: Interpolation parameters.
    k: Interpolation parameters.
    fieldname: str
        The name of the field (used only in the progress bar).
    """

    cdef np.ndarray[DTYPE_t,ndim=2] dp = ((bbox[:, 1] - bbox[:, 0]) / domain_dimensions).reshape((3,1))
    #: dp is the separation size between bin edges along each dimension/
    cdef np.ndarray[DTYPE_t,ndim=2] sbbox
    #: sbbox is the chunk-specific bounding box. Built by taking the BBOX and adding dp * the chunk position coordinates.
    cdef np.ndarray[ITYPE32_t,ndim=1] s
    #: Proxy for the chunk size.
    cdef np.ndarray[DTYPE_t,ndim=3] _x,_y,_z
    cdef np.ndarray[DTYPE_t,ndim=3] _r

    # Create the progress bar to iterate over the relevant chunks.
    pbar = tqdm(
        desc=f"Chunked Interpolation: {fieldname}",
        leave=False,
        total=chunkmap.shape[2],
    )
    s = chunkmap[:, 1, 0] - chunkmap[:, 0, 0]
    # s is the separation size. ! We can do it this way because we forced chunksize to evenly divide the domain.

    # --- Performing iteration over chunks --- #

    for chunk_id in range(chunkmap.shape[2]): # --> Iterate over all of the chunks.
        sbbox =  bbox[:, 0].reshape(3, 1) + chunkmap[:,:,chunk_id]*dp
        _x, _y, _z = np.mgrid[
                     sbbox[0, 0]:sbbox[0, 1]:s[0] * 1j,
                     sbbox[1, 0]:sbbox[1, 1]:s[1] * 1j,
                     sbbox[2, 0]:sbbox[2, 1]:s[2] * 1j
                     ]

        _r = np.sqrt(_x ** 2 + _y ** 2 + _z ** 2)
        buffer_object[chunk_id,:,:,:] += _dfitpack.splev(t,c,k,_r,0)[0].reshape(s)

        pbar.update()
    pbar.close()
