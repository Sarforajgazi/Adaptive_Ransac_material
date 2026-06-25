/*
 * bridge.h — Thin C++ bridge between Schnabel's Efficient RANSAC and Cython.
 *
 * Wraps the complex template-heavy Schnabel API into simple flat-array I/O
 * that Cython can call directly.
 */
#ifndef SCHNABEL_BRIDGE_H
#define SCHNABEL_BRIDGE_H

#include <cstddef>

/* Result struct for each detected shape. */
struct DetectedShape {
    int type;               /* 0=plane, 1=sphere, 2=cylinder, 3=cone, 4=torus */
    size_t num_points;      /* number of inlier points */
    float params[16];       /* serialized shape parameters (see SerializedFloatSize) */
    size_t num_params;      /* how many entries in params[] are valid */
    char description[256];  /* human-readable description string */
};

/*
 * Detect primitive shapes in a 3D point cloud.
 *
 * Parameters
 * ----------
 * points          : flat float array [x0,y0,z0, x1,y1,z1, ...], size = num_points*3
 * num_points      : number of 3D points
 * epsilon         : distance threshold (absolute). NOTE: internally multiplied by 3
 *                   for global scoring in the Schnabel algorithm!
 * normal_thresh   : cos(max_normal_angle), e.g. 0.9 ≈ 26°
 * min_support     : minimum inlier count to accept a shape
 * bitmap_epsilon  : bitmap cell size for connected-component analysis
 * probability     : miss probability for adaptive stopping (e.g. 0.001)
 * shape_mask      : bitmask selecting shapes to detect:
 *                   bit0(1)=plane, bit1(2)=sphere, bit2(4)=cylinder,
 *                   bit3(8)=cone, bit4(16)=torus
 * normal_radius   : neighbourhood radius for PCA-based normal estimation
 * normal_knn      : k for k-nearest-neighbours in normal estimation
 * out_shapes      : pre-allocated array of DetectedShape[max_shapes]
 * max_shapes      : capacity of out_shapes
 * out_point_shape_index : pre-allocated int[num_points], filled with per-point
 *                   shape index (0-based) or -1 if unassigned. May be NULL.
 *
 * Returns
 * -------
 * Number of shapes detected (≤ max_shapes).
 */
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
);

#endif /* SCHNABEL_BRIDGE_H */
