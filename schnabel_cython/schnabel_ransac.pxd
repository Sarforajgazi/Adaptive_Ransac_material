# schnabel_ransac.pxd
# Cython declaration file — tells Cython about the C++ bridge types and functions.

from libc.stddef cimport size_t

cdef extern from "bridge.h":
    # Result struct for each detected shape
    cdef struct DetectedShape:
        int type
        size_t num_points
        float params[16]
        size_t num_params
        char description[256]

    # Main detection function — declared 'nogil' so we can release
    # Python's GIL before calling, allowing C++ OpenMP threads to run.
    size_t detect_shapes(
        const float* points,
        size_t num_points,
        float epsilon,
        float normal_thresh,
        unsigned int min_support,
        float bitmap_epsilon,
        float probability,
        int shape_mask,
        float normal_radius,
        unsigned int normal_knn,
        DetectedShape* out_shapes,
        size_t max_shapes,
        int* out_point_shape_index
    ) nogil
