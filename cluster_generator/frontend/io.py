from itertools import groupby

import numpy as np
from yt.utilities.io_handler import BaseIOHandler


# group grids with consecutive indices together to improve the I/O performance
# --> grids are assumed to be sorted into ascending numerical order already
def grid_sequences(grids):
    for _k, g in groupby(enumerate(grids), lambda i_x: i_x[0] - i_x[1].id):
        seq = [v[1] for v in g]
        yield seq


class ClusterGeneratorIOHandler(BaseIOHandler):
    _particle_reader = False
    _dataset_type = "cluster_generator"

    def __init__(self, ds):
        super().__init__(ds)
        self._handle = ds._handle

    def _read_particle_coords(self, chunks, ptf):
        # There are no particles in our datasets.
        pass

    def _read_particle_fields(self, chunks, ptf, selector):
        # This gets called after the arrays have been allocated.  It needs to
        # yield ((ptype, field), data) where data is the masked results of
        # reading ptype, field and applying the selector to the data read in.
        # Selector objects have a .select_points(x,y,z) that returns a mask, so
        # you need to do your masking here.
        pass

    def _read_fluid_selection(self, chunks, selector, fields, size):
        # This needs to allocate a set of arrays inside a dictionary, where the
        # keys are the (ftype, fname) tuples and the values are arrays that
        # have been masked using whatever selector method is appropriate.  The
        # dict gets returned at the end and it should be flat, with selected
        # data.  Note that if you're reading grid data, you might need to
        # special-case a grid selector object.
        # Also note that "chunks" is a generator for multiple chunks, each of
        # which contains a list of grids. The returned numpy arrays should be
        # in 64-bit float and contiguous along the z direction. Therefore, for
        # a C-like input array with the dimension [x][y][z] or a
        # Fortran-like input array with the dimension (z,y,x), a matrix
        # transpose is required (e.g., using np_array.transpose() or
        # np_array.swapaxes(0,2)).

        # This method is not abstract, and has a default implementation
        # in the base class.However, the default implementation requires that the method
        # io_iter be defined
        chunks = list(chunks)  # force the generator to yield as a list

        # -- Check fields for validity -- #
        if any((ftype != self._dataset_type for ftype, _ in fields)):
            raise NotImplementedError(
                f"One of the field types is not valid for dataset type of {self._dataset_type}."
            )

        # -- Setup field iteration -- #
        returned_fields = {}
        for field in fields:
            returned_fields[field] = np.empty(size, dtype="=f8")  # allocate empty grid with correct sizes.

        # -- Iterating over the fields -- #
        for field in fields:
            ftype, fname = field

            ds = self._handle[f"grid/{fname}"]

            ind = 0
            for chunk in chunks:
                for gs in grid_sequences(chunk.objs):
                    start = gs[0].id - gs[0]._id_offset
                    end = gs[-1].id - gs[-1]._id_offset + 1
                    data = ds[start:end, :, :, :].transpose()
                    for i, g in enumerate(gs):
                        ind += g.select(selector, data[..., i], returned_fields[field], ind)
        return returned_fields

    def _read_chunk_data(self, chunk, fields):
        # This reads the data from a single chunk without doing any selection,
        # and is only used for caching data that might be used by multiple
        # different selectors later. For instance, this can speed up ghost zone
        # computation.
        returned_fields = {}
        if len(chunk.objs) == 0:
            return returned_fields

        for g in chunk.objs:
            returned_fields[g.id] = {}

        for field in fields:
            ftype, fname = field
            ds = self._handle[f"grid/{fname}"]

            for gs in grid_sequences(chunk.objs):
                start = gs[0].id - gs[0]._id_offset
                end = gs[-1].id - gs[-1]._id_offset + 1
                buf = ds[start:end, :, :, :].transpose()

                for i, g in enumerate(gs):
                    returned_fields[g.id][field] = buf[..., i]

        return returned_fields
