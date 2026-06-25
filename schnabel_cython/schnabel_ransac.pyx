# schnabel_ransac.pyx
# cython: language_level=3
# distutils: language = c++
"""
Python wrapper for Schnabel's Efficient RANSAC shape detection.

Usage
-----
    import numpy as np
    import schnabel_ransac

    points = np.random.randn(10000, 3).astype(np.float32)
    shapes, n_remaining = schnabel_ransac.detect(points, epsilon=0.01)

    for s in shapes:
        print(s["type"], s["n_points"], s["description"])
"""

import numpy as np
cimport numpy as cnp
from libc.stdlib cimport malloc, free
from libc.string cimport memset, memcpy

from schnabel_ransac cimport DetectedShape, detect_shapes

cnp.import_array()

# Shape type ID → name mapping (matches PrimitiveShape::Identifier())
SHAPE_NAMES = {0: "plane", 1: "sphere", 2: "cylinder", 3: "cone", 4: "torus"}

# Shape name → bitmask mapping
_SHAPE_BITS = {
    "plane": 1, "sphere": 2, "cylinder": 4, "cone": 8, "torus": 16,
}


def detect(
    points_input,
    float epsilon=0.01,
    float normal_thresh=0.9,
    unsigned int min_support=500,
    float bitmap_epsilon=-1.0,
    float probability=0.001,
    list shapes=None,
    bint relative_epsilon=True,
    float normal_radius=3.0,
    unsigned int normal_knn=20,
    size_t max_shapes=100,
):
    """
    Detect primitive shapes in a 3D point cloud using Efficient RANSAC.

    Parameters
    ----------
    points_input : array-like, shape (N, 3)
        The 3D point cloud. Will be converted to float32 C-contiguous.

    epsilon : float, default 0.01
        Distance threshold.
        - If ``relative_epsilon=True`` (default): fraction of the bounding-box
          diagonal (e.g. 0.01 = 1% of bbox size).
        - If ``relative_epsilon=False``: absolute distance in data units.
        **NOTE**: Internally the Schnabel algorithm multiplies this by 3 for
        global scoring!

    normal_thresh : float, default 0.9
        Cosine of the maximum allowed angle between a point's surface normal
        and the candidate shape's normal.  0.9 ≈ 26°, 0.95 ≈ 18°.

    min_support : int, default 500
        Minimum number of inlier points for a shape to be accepted.

    bitmap_epsilon : float, default -1 (auto)
        Bitmap cell size for connected-component filtering.  If negative,
        set to ``2 * epsilon`` automatically.

    probability : float, default 0.001
        Miss probability for the adaptive stopping criterion.  Lower = more
        thorough search.  The algorithm stops when the probability of missing
        a better shape drops below this value.

    shapes : list of str, optional
        Which shape types to detect.  Default: ``["plane"]``.
        Options: ``"plane"``, ``"sphere"``, ``"cylinder"``, ``"cone"``,
        ``"torus"``.

    relative_epsilon : bool, default True
        If True, ``epsilon`` and ``bitmap_epsilon`` are interpreted as
        fractions of the bounding-box diagonal.  If False, they are
        absolute values.

    normal_radius : float, default 3.0
        Neighbourhood radius for PCA-based surface normal estimation.

    normal_knn : int, default 20
        Number of nearest neighbours for normal estimation.

    max_shapes : int, default 100
        Maximum number of shapes to return.

    Returns
    -------
    detected : list of dict
        Each dict contains:

        - ``"type"`` : str — shape type name (e.g. ``"plane"``)
        - ``"type_id"`` : int — shape type identifier (0–4)
        - ``"n_points"`` : int — number of inlier points
        - ``"description"`` : str — human-readable description
        - ``"params"`` : np.ndarray (float32) — serialized shape parameters
        - ``"inlier_mask"`` : np.ndarray (bool, shape N) — True for inliers

    n_remaining : int
        Number of points not assigned to any shape.
    """

    # ---- Convert input to contiguous float32 ----
    cdef cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] points
    points = np.ascontiguousarray(points_input, dtype=np.float32)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            f"points must have shape (N, 3), got {tuple(points_input.shape)}"
        )

    cdef size_t n_points = <size_t>points.shape[0]
    if n_points == 0:
        return [], 0

    # ---- Build shape bitmask ----
    if shapes is None:
        shapes = ["plane"]

    cdef int shape_mask = 0
    for s in shapes:
        key = s.strip().lower()
        if key not in _SHAPE_BITS:
            raise ValueError(
                f"Unknown shape type '{s}'. "
                f"Choose from: {list(_SHAPE_BITS.keys())}"
            )
        shape_mask |= _SHAPE_BITS[key]

    # ---- Resolve relative epsilon ----
    cdef float abs_epsilon = epsilon
    cdef float abs_bitmap_epsilon = bitmap_epsilon
    cdef float scale

    if relative_epsilon:
        bbox_min = np.min(points, axis=0)
        bbox_max = np.max(points, axis=0)
        diff = bbox_max - bbox_min
        scale = float(np.max(diff))
        if scale < 1e-10:
            scale = 1.0
        abs_epsilon = epsilon * scale

    if abs_bitmap_epsilon < 0:
        abs_bitmap_epsilon = 2.0 * abs_epsilon
    elif relative_epsilon and bitmap_epsilon > 0:
        abs_bitmap_epsilon = bitmap_epsilon * scale

    # ---- Allocate output buffers ----
    cdef DetectedShape* out_buf = NULL
    cdef int* point_idx = NULL
    cdef float* pts_ptr = <float*>points.data
    cdef size_t n_detected
    cdef cnp.ndarray[cnp.int32_t, ndim=1] point_assignments
    cdef size_t n_remaining = n_points

    try:
        out_buf = <DetectedShape*>malloc(max_shapes * sizeof(DetectedShape))
        point_idx = <int*>malloc(n_points * sizeof(int))

        if out_buf == NULL or point_idx == NULL:
            raise MemoryError("Failed to allocate output buffers")

        memset(out_buf, 0, max_shapes * sizeof(DetectedShape))

        # ---- Call C++ with GIL released (enables OpenMP inside C++) ----
        with nogil:
            n_detected = detect_shapes(
                pts_ptr, n_points,
                abs_epsilon, normal_thresh, min_support,
                abs_bitmap_epsilon, probability,
                shape_mask,
                normal_radius, normal_knn,
                out_buf, max_shapes,
                point_idx,
            )

        # ---- Convert C results → Python ----

        # Build per-point assignment array (numpy int32)
        point_assignments = np.empty(n_points, dtype=np.int32)
        memcpy(<void*>point_assignments.data, <void*>point_idx,
               n_points * sizeof(int))

        result = []

        for i in range(n_detected):
            # Per-shape inlier mask
            inlier_mask = (point_assignments == <int>i)

            # Extract float parameters
            n_params = out_buf[i].num_params
            params = np.empty(n_params, dtype=np.float32)
            for k in range(n_params):
                params[k] = out_buf[i].params[k]

            shape_dict = {
                "type": SHAPE_NAMES.get(out_buf[i].type, "unknown"),
                "type_id": int(out_buf[i].type),
                "n_points": int(out_buf[i].num_points),
                "description": out_buf[i].description.decode(
                    "utf-8", errors="replace"
                ),
                "params": params,
                "inlier_mask": inlier_mask,
            }
            result.append(shape_dict)
            n_remaining -= out_buf[i].num_points

        return result, int(n_remaining)

    finally:
        # Cleanup — runs even if an exception was raised above
        if out_buf != NULL:
            free(out_buf)
        if point_idx != NULL:
            free(point_idx)
