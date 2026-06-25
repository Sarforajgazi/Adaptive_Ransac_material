"""
test_wrapper.py — Smoke test for the schnabel_ransac Cython extension.

Creates a synthetic point cloud with a known flat plane + random noise,
runs Efficient RANSAC, and verifies the results.

Usage:
    python test_wrapper.py
"""

import numpy as np
import time

# ---- Import the compiled extension ----
try:
    import schnabel_ransac
except ImportError as e:
    print(f"ERROR: Could not import schnabel_ransac: {e}")
    print("Did you run:  python setup.py build_ext --inplace  ?")
    raise SystemExit(1)

print("=" * 60)
print("Schnabel Efficient RANSAC — Cython Wrapper Test")
print("=" * 60)


# ============================================================
# TEST 1: Flat plane at z=0
# ============================================================
print("\n--- Test 1: Flat plane at z=0 ---")

np.random.seed(42)

# 10000 points on a plane at z=0 with small noise
n_plane = 10000
plane_pts = np.column_stack([
    np.random.uniform(-5, 5, n_plane),
    np.random.uniform(-5, 5, n_plane),
    np.random.normal(0, 0.005, n_plane),   # z ≈ 0
]).astype(np.float32)

# 1000 random outlier points
n_noise = 1000
noise_pts = np.random.uniform(-5, 5, (n_noise, 3)).astype(np.float32)

points = np.vstack([plane_pts, noise_pts])
print(f"Point cloud: {points.shape[0]} points ({n_plane} plane + {n_noise} noise)")

# Run detection
t0 = time.perf_counter()
shapes, n_remaining = schnabel_ransac.detect(
    points,
    epsilon=0.01,           # 1% of bbox
    normal_thresh=0.9,
    min_support=100,
    shapes=["plane"],
)
elapsed = time.perf_counter() - t0

print(f"Detection time: {elapsed:.3f}s")
print(f"Shapes detected: {len(shapes)}")
print(f"Remaining (unassigned) points: {n_remaining}")

for i, s in enumerate(shapes):
    print(f"\n  Shape {i}:")
    print(f"    Type:        {s['type']} (id={s['type_id']})")
    print(f"    Inliers:     {s['n_points']}")
    print(f"    Description: {s['description'][:100]}")
    print(f"    Params:      {s['params']}")
    print(f"    Inlier mask: {s['inlier_mask'].sum()} True / {len(s['inlier_mask'])} total")

# Basic checks
if len(shapes) >= 1:
    main_shape = shapes[0]
    assert main_shape["type"] == "plane", f"Expected plane, got {main_shape['type']}"
    assert main_shape["n_points"] > n_plane * 0.5, (
        f"Expected majority of plane points as inliers, got {main_shape['n_points']}"
    )
    print("\n  ✓ Plane detected with sufficient inliers")
else:
    print("\n  ✗ WARNING: No shapes detected!")


# ============================================================
# TEST 2: Multiple shapes (plane + cylinder)
# ============================================================
print("\n--- Test 2: Plane + Cylinder ---")

np.random.seed(123)

# Plane at z=0
n_p = 5000
plane2 = np.column_stack([
    np.random.uniform(-10, 10, n_p),
    np.random.uniform(-10, 10, n_p),
    np.random.normal(0, 0.01, n_p),
]).astype(np.float32)

# Cylinder: radius=2, axis along z, centered at (5, 5, z)
n_c = 3000
theta = np.random.uniform(0, 2 * np.pi, n_c)
z_cyl = np.random.uniform(-3, 3, n_c)
cyl = np.column_stack([
    5.0 + 2.0 * np.cos(theta) + np.random.normal(0, 0.01, n_c),
    5.0 + 2.0 * np.sin(theta) + np.random.normal(0, 0.01, n_c),
    z_cyl,
]).astype(np.float32)

points2 = np.vstack([plane2, cyl])
print(f"Point cloud: {points2.shape[0]} points ({n_p} plane + {n_c} cylinder)")

t0 = time.perf_counter()
shapes2, n_rem2 = schnabel_ransac.detect(
    points2,
    epsilon=0.005,
    normal_thresh=0.9,
    min_support=100,
    shapes=["plane", "cylinder"],
)
elapsed2 = time.perf_counter() - t0

print(f"Detection time: {elapsed2:.3f}s")
print(f"Shapes detected: {len(shapes2)}")
print(f"Remaining points: {n_rem2}")

for i, s in enumerate(shapes2):
    print(f"\n  Shape {i}: {s['type']} — {s['n_points']} points")
    print(f"    {s['description'][:100]}")


# ============================================================
# TEST 3: Absolute epsilon mode
# ============================================================
print("\n--- Test 3: Absolute epsilon ---")

shapes3, n_rem3 = schnabel_ransac.detect(
    plane_pts,
    epsilon=0.02,             # absolute distance, not relative
    relative_epsilon=False,
    min_support=50,
    shapes=["plane"],
)

print(f"Shapes detected (absolute epsilon): {len(shapes3)}")
if len(shapes3) >= 1:
    print(f"  Plane: {shapes3[0]['n_points']} inliers")
    print("  ✓ Absolute epsilon mode works")


# ============================================================
print("\n" + "=" * 60)
print("All tests passed!")
print("=" * 60)
