import cython
import numpy
from math import sqrt

# @cython.boundscheck(False)
# def get_distance_matrix(coord, sparse_contacts, cutoff):
#     rows = len(coord)
#     coord_view = coord[:, ::1]

#     rr = [0.0,0.0,0.0]
#     for i in range(rows):
#         for j in range(i+1, rows):
#             rr[0] = coord_view[i, 0] - coord_view[j, 0]
#             rr[1] = coord_view[i, 1] - coord_view[j, 1]
#             rr[2] = coord_view[i, 2] - coord_view[j, 2]
#             dist = rr[0]*rr[0] + rr[1]*rr[1] + rr[2]*rr[2]
#             if dist < cutoff:
#                 sparse_contacts[i, j] = dist
#                 sparse_contacts[j, i] = dist


@cython.boundscheck(False)
def get_distance_matrix(coord, sparse_contacts, cutoff):
    cdef int rows = len(coord)
    cdef float[:, ::1] coord_view = coord

    cdef int i, j
    cdef double[3] rr
    cdef double dist
    for i in range(rows):
        for j in range(i+1, rows):
            rr[0] = coord_view[i, 0] - coord_view[j, 0]
            rr[1] = coord_view[i, 1] - coord_view[j, 1]
            rr[2] = coord_view[i, 2] - coord_view[j, 2]
            dist = sqrt(rr[0]*rr[0] + rr[1]*rr[1] + rr[2]*rr[2])
            if dist < cutoff:
                sparse_contacts[i, j] = dist
                sparse_contacts[j, i] = dist