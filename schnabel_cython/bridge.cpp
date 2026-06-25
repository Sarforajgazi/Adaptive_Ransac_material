/*
 * bridge.cpp — Implementation of the Schnabel RANSAC bridge.
 *
 * Workflow:
 *   1. Copy float array → PointCloud (with index tracking via POINTSWITHINDEX)
 *   2. Auto-compute bounding box from data
 *   3. Estimate surface normals via PCA + KD-Tree
 *   4. Configure RansacShapeDetector with user-supplied Options
 *   5. Add requested shape constructors
 *   6. Run Detect() — probabilistic stopping + octree sampling + OpenMP
 *   7. Extract results into flat DetectedShape structs
 *   8. Map rearranged points back to original indices via Point::index
 */

#include "bridge.h"

#include <PointCloud.h>
#include <RansacShapeDetector.h>
#include <PlanePrimitiveShapeConstructor.h>
#include <SpherePrimitiveShapeConstructor.h>
#include <CylinderPrimitiveShapeConstructor.h>
#include <ConePrimitiveShapeConstructor.h>
#include <TorusPrimitiveShapeConstructor.h>

#include <MiscLib/RefCountPtr.h>
#include <MiscLib/Vector.h>

#include <cstring>
#include <string>
#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

using MiscLib::RefCountPtr;

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
    int* out_point_shape_index)
{
    /* ------- Sanity checks ------- */
    if (num_points == 0 || !out_shapes || max_shapes == 0)
        return 0;

    /* ------- 1. Build PointCloud + track bounding box ------- */
    PointCloud pc;

    float mn_x =  std::numeric_limits<float>::max();
    float mn_y =  std::numeric_limits<float>::max();
    float mn_z =  std::numeric_limits<float>::max();
    float mx_x = -std::numeric_limits<float>::max();
    float mx_y = -std::numeric_limits<float>::max();
    float mx_z = -std::numeric_limits<float>::max();

    for (size_t i = 0; i < num_points; ++i) {
        float x = points[i * 3 + 0];
        float y = points[i * 3 + 1];
        float z = points[i * 3 + 2];

        Point p(Vec3f(x, y, z));
#ifdef POINTSWITHINDEX
        p.index = i;   /* track original position through rearrangements */
#endif
        pc.push_back(p);

        if (x < mn_x) mn_x = x;  if (x > mx_x) mx_x = x;
        if (y < mn_y) mn_y = y;  if (y > mx_y) mx_y = y;
        if (z < mn_z) mn_z = z;  if (z > mx_z) mx_z = z;
    }

    /* ------- 2. Set bounding box (small padding to avoid edge issues) ------- */
    float dx = mx_x - mn_x;
    float dy = mx_y - mn_y;
    float dz = mx_z - mn_z;
    float scale = std::max(dx, std::max(dy, dz));
    if (scale < 1e-10f) scale = 1.0f;  /* degenerate case: all points identical */
    float pad = 0.01f * scale;

    pc.setBBox(
        Vec3f(mn_x - pad, mn_y - pad, mn_z - pad),
        Vec3f(mx_x + pad, mx_y + pad, mx_z + pad)
    );

    /* ------- 3. Estimate surface normals (PCA via KD-Tree) ------- */
    pc.calcNormals(normal_radius, normal_knn);

    /* ------- 4. Configure RANSAC options ------- */
    RansacShapeDetector::Options opts;
    opts.m_epsilon       = epsilon;        /* distance threshold (×3 internally for global) */
    opts.m_normalThresh  = normal_thresh;  /* cos of max normal deviation */
    opts.m_minSupport    = min_support;    /* minimum inlier count */
    opts.m_bitmapEpsilon = bitmap_epsilon; /* bitmap cell size for connected components */
    opts.m_probability   = probability;    /* miss probability → adaptive stopping */

    RansacShapeDetector detector(opts);

    /* ------- 5. Add requested shape constructors ------- */
    if (shape_mask & 1)  detector.Add(new PlanePrimitiveShapeConstructor());
    if (shape_mask & 2)  detector.Add(new SpherePrimitiveShapeConstructor());
    if (shape_mask & 4)  detector.Add(new CylinderPrimitiveShapeConstructor());
    if (shape_mask & 8)  detector.Add(new ConePrimitiveShapeConstructor());
    if (shape_mask & 16) detector.Add(new TorusPrimitiveShapeConstructor());

    /* ------- 6. Run detection ------- */
    MiscLib::Vector< std::pair< RefCountPtr< PrimitiveShape >, size_t > > shapes;
    /* size_t remaining = */ detector.Detect(pc, 0, pc.size(), &shapes);

    /* ------- 7. Extract results ------- */
    size_t n_detected = shapes.size();
    if (n_detected > max_shapes)
        n_detected = max_shapes;

    /* Initialize per-point index to -1 (unassigned) */
    if (out_point_shape_index) {
        for (size_t i = 0; i < num_points; ++i)
            out_point_shape_index[i] = -1;
    }

    /*
     * After Detect(), points in pc are REARRANGED:
     *   shape[0] points → last shapes[0].second entries of pc
     *   shape[1] points → before that
     *   ...
     *   remaining unassigned → at the front
     *
     * With POINTSWITHINDEX, pc[i].index gives the ORIGINAL input index.
     */
    size_t end_idx = pc.size();

    for (size_t s = 0; s < n_detected; ++s) {
        size_t begin_idx = end_idx - shapes[s].second;

        /* Shape type (0=plane, 1=sphere, etc.) */
        out_shapes[s].type = (int)shapes[s].first->Identifier();
        out_shapes[s].num_points = shapes[s].second;

        /* Human-readable description */
        std::string desc;
        shapes[s].first->Description(&desc);
        std::strncpy(out_shapes[s].description, desc.c_str(), 255);
        out_shapes[s].description[255] = '\0';

        /* Serialized float parameters */
        size_t nparams = shapes[s].first->SerializedFloatSize();
        if (nparams > 16) nparams = 16;
        out_shapes[s].num_params = nparams;
        std::memset(out_shapes[s].params, 0, sizeof(out_shapes[s].params));
        shapes[s].first->Serialize(out_shapes[s].params);

        /* Map rearranged points back to original indices */
        if (out_point_shape_index) {
#ifdef POINTSWITHINDEX
            for (size_t i = begin_idx; i < end_idx; ++i) {
                size_t orig = pc[i].index;
                if (orig < num_points) {
                    out_point_shape_index[orig] = (int)s;
                }
            }
#endif
        }

        end_idx = begin_idx;
    }

    return n_detected;
}
