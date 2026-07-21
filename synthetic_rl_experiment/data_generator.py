import os
import json
import numpy as np
import open3d as o3d


def save_scene(path_prefix, points, gt_mask, n_true, d_true, meta=None):
    """
    Persists a generated scene so it can be reloaded later instead of
    regenerated -- generate_scene() only ever returns in-memory arrays, so
    without this every scene is lost the moment the process exits.

    Writes two files:
      <path_prefix>.ply       -- the point cloud, colored by gt_mask
                                  (green=true inlier, gray=outlier/clutter)
      <path_prefix>_meta.json -- n_true, d_true, gt_mask, and generation
                                  params, for exact reproducibility and reuse
                                  in scripts that need the ground truth
                                  (evaluation, further RANSAC testing, etc).
    """
    os.makedirs(os.path.dirname(os.path.abspath(path_prefix)) or ".", exist_ok=True)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    colors = np.zeros((len(points), 3))
    colors[gt_mask] = [0.0, 0.8, 0.0]
    colors[~gt_mask] = [0.6, 0.6, 0.6]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    ply_path = path_prefix + ".ply"
    o3d.io.write_point_cloud(ply_path, pcd)

    meta_out = dict(meta or {})
    meta_out.update({
        "n_true": np.asarray(n_true).tolist(),
        "d_true": float(d_true),
        "gt_mask": np.asarray(gt_mask).astype(bool).tolist(),
        "num_points": int(len(points)),
    })
    meta_path = path_prefix + "_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta_out, f)

    return ply_path, meta_path


def load_scene(path_prefix):
    """Inverse of save_scene() -- returns (points, gt_mask, n_true, d_true, meta)."""
    pcd = o3d.io.read_point_cloud(path_prefix + ".ply")
    points = np.asarray(pcd.points).astype(np.float32)

    with open(path_prefix + "_meta.json") as f:
        meta = json.load(f)

    gt_mask = np.array(meta.pop("gt_mask"), dtype=bool)
    n_true = np.array(meta.pop("n_true"), dtype=np.float32)
    d_true = meta.pop("d_true")
    meta.pop("num_points", None)

    return points, gt_mask, n_true, d_true, meta


# Per-point source labels, opt-in via generate_scene(..., return_labels=True).
# Purely for visualization (e.g. visualize_synthetic_plane.py) -- not read
# anywhere in the RL loop, training, or evaluation, so adding this cannot
# affect the already-trained model or require a retrain.
SOURCE_FLAT_INLIER = 0
SOURCE_BUMP = 1
SOURCE_WAVE = 2
SOURCE_CYLINDER = 3
SOURCE_BOX = 4
SOURCE_WALL = 5
SOURCE_BACKGROUND = 6
SOURCE_LABEL_NAMES = {
    SOURCE_FLAT_INLIER: "flat_inlier",
    SOURCE_BUMP: "bump/crater",
    SOURCE_WAVE: "sinusoidal terrain",
    SOURCE_CYLINDER: "cylinder clutter",
    SOURCE_BOX: "box clutter",
    SOURCE_WALL: "intersecting plane/wall",
    SOURCE_BACKGROUND: "background noise",
}
# Displacement magnitude above which a bump/wave-touched inlier point is
# labeled as such -- both deformations are smooth fields technically
# touching every inlier a tiny amount, so a hard label needs a cutoff
# rather than any nonzero displacement.
_DEFORMATION_LABEL_THRESHOLD = 0.02


class SyntheticPlaneGenerator:
    """
    Generates synthetic point clouds containing a geometric plane with known ground truth
    (n_true, d_true) and a boolean mask of true inliers.
    """
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def _apply_noise(self, inliers, n_true, noise_sigma, noise_type="gaussian"):
        """
        Displace inlier points along the plane normal. All noise types
        perturb points up/down from the surface -- they differ only in the
        distribution shape, modeling different real-sensor error profiles.
        """
        num = len(inliers)

        if noise_type == "gaussian":
            magnitudes = self.rng.normal(0.0, noise_sigma, num)

        elif noise_type == "laplacian":
            # Heavy-tailed: occasional large displacements, e.g. multi-path
            # reflections in real LiDAR.
            magnitudes = self.rng.laplace(0.0, noise_sigma / np.sqrt(2), num)

        elif noise_type == "uniform":
            # Bounded noise -- e.g. quantization or structured sensor error.
            band = noise_sigma * np.sqrt(3)  # match variance to Gaussian
            magnitudes = self.rng.uniform(-band, band, num)

        elif noise_type == "spatially_varying":
            # Noise grows with distance from the sensor (placed at origin).
            distances = np.linalg.norm(inliers, axis=1)
            max_dist = np.max(distances) + 1e-8
            local_sigma = noise_sigma * (0.3 + 1.4 * (distances / max_dist) ** 2)
            magnitudes = self.rng.normal(0.0, 1.0, num) * local_sigma

        elif noise_type == "mixed":
            # 80% Gaussian + 20% heavy-tailed -- mostly clean with some bad returns.
            magnitudes = self.rng.normal(0.0, noise_sigma, num)
            heavy_mask = self.rng.random(num) < 0.2
            magnitudes[heavy_mask] = self.rng.laplace(0.0, noise_sigma * 2, int(heavy_mask.sum()))

        else:
            raise ValueError(f"Unknown noise_type: {noise_type}")

        return inliers + np.outer(magnitudes, n_true)

    def _add_surface_deformations(self, inliers, inliers_uv, n_true, num_bumps, box_size):
        """
        Add Gaussian bumps and craters to the plane surface -- terrain
        features (dirt roads, gravel, grass), not clutter. Points stay
        ground-truth inliers; RANSAC must still fit the best-approximating
        plane through this irregular terrain.

        Also returns the per-point cumulative |displacement|, purely for
        optional visualization labeling (which inlier points are "on a
        bump" vs. flat) -- not used anywhere in the RL loop or training.
        """
        total_abs_displacement = np.zeros(len(inliers))
        for _ in range(num_bumps):
            center_uv = self.rng.uniform(-box_size / 3, box_size / 3, 2)
            radius = self.rng.uniform(0.5, 2.5)
            # Comparable to the noise_sigma range (0.01-0.20), not a boulder --
            # at the old +-0.4 range, bumps were up to 40x noise_sigma at low
            # noise, setting a noise-independent eps floor that swamped the
            # noise-adaptivity signal entirely (confirmed: bump+noise residual
            # std was 3x pure-noise std at noise_sigma=0.01).
            height = self.rng.uniform(-0.10, 0.10)  # negative = crater, positive = bump

            dist_sq = ((inliers_uv[:, 0] - center_uv[0]) ** 2 +
                       (inliers_uv[:, 1] - center_uv[1]) ** 2)
            displacement = height * np.exp(-dist_sq / (2 * radius ** 2))

            inliers = inliers + np.outer(displacement, n_true)
            total_abs_displacement += np.abs(displacement)

        return inliers, total_abs_displacement

    def _add_sinusoidal_deformation(self, inliers, inliers_uv, n_true, num_waves, box_size):
        """
        Add sinusoidal (wavy/undulating) surface deformation -- rolling
        terrain, washboard gravel, sand ripples -- distinct from the
        localized Gaussian bumps/craters above (a smooth global wave vs. a
        local perturbation). Points stay ground-truth inliers.

        Also returns the per-point cumulative |displacement|, same purpose
        and caveat as `_add_surface_deformations` above.
        """
        total_abs_displacement = np.zeros(len(inliers))
        for _ in range(num_waves):
            theta = self.rng.uniform(0, 2 * np.pi)
            direction = np.array([np.cos(theta), np.sin(theta)])
            wavelength = self.rng.uniform(1.0, 4.0)
            # Same amplitude ceiling as the bump fix above -- keeps this
            # comparable to noise_sigma (0.01-0.20) rather than reintroducing
            # the noise-independent eps floor the old +-0.4m bump height once
            # caused (see the comment on `height` above and
            # SESSION_PROGRESS_LOG.md sec.11).
            amplitude = self.rng.uniform(0.02, 0.10)
            phase = self.rng.uniform(0, 2 * np.pi)

            projected = inliers_uv[:, 0] * direction[0] + inliers_uv[:, 1] * direction[1]
            displacement = amplitude * np.sin(2 * np.pi * projected / wavelength + phase)

            inliers = inliers + np.outer(displacement, n_true)
            total_abs_displacement += np.abs(displacement)

        return inliers, total_abs_displacement

    def _create_plane_frame(self, normal):
        """Orthonormal in-plane basis (u, v) for the given normal."""
        helper = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = np.cross(normal, helper)
        u /= np.linalg.norm(u)
        v = np.cross(normal, u)
        return u, v

    def _generate_cylinder(self, base_point, n_ground, num_points):
        """
        A vertical cylinder (tree trunk, pole) rising from the ground.
        Points are on the cylinder surface, NOT ground truth -- a
        false-positive trap for loose eps values.
        """
        radius = self.rng.uniform(0.1, 0.5)
        height = self.rng.uniform(1.0, 4.0)

        up = n_ground / (np.linalg.norm(n_ground) + 1e-8)
        side1, side2 = self._create_plane_frame(up)

        angles = self.rng.uniform(0, 2 * np.pi, num_points)
        heights = self.rng.uniform(0, height, num_points)

        pts = (base_point +
               heights[:, None] * up +
               radius * np.cos(angles)[:, None] * side1 +
               radius * np.sin(angles)[:, None] * side2)

        return pts.astype(np.float32)

    def _generate_box(self, base_point, n_ground, num_points):
        """
        A rectangular box (barrier, rock) sitting on the ground.
        Points are on the box surface, NOT ground truth.
        """
        width = self.rng.uniform(0.3, 1.5)
        depth = self.rng.uniform(0.3, 1.5)
        height = self.rng.uniform(0.3, 1.5)

        up = n_ground / (np.linalg.norm(n_ground) + 1e-8)
        side1, side2 = self._create_plane_frame(up)

        pts_per_face = max(1, num_points // 6)
        faces = []

        for sign in [-1, 1]:
            for axis, extent in [(side1, width / 2), (side2, depth / 2), (up, height)]:
                u_ax = side1 if not np.allclose(axis, side1) else side2
                v_ax = side2 if not np.allclose(axis, side2) else up
                u_ext = width / 2 if np.allclose(u_ax, side1) else depth / 2
                v_ext = depth / 2 if np.allclose(v_ax, side2) else height / 2

                u_coords = self.rng.uniform(-u_ext, u_ext, pts_per_face)
                v_coords = self.rng.uniform(-v_ext, v_ext, pts_per_face)
                face_pts = (base_point + sign * extent * axis +
                            u_coords[:, None] * u_ax + v_coords[:, None] * v_ax)
                faces.append(face_pts)

        return np.vstack(faces).astype(np.float32)[:num_points]

    def _generate_intersecting_plane(self, n_ground, d_ground, p0, u, v, box_size, num_points):
        """
        A second plane (wall, ramp, embankment) intersecting or nearly
        touching the ground plane. Points are on this second plane, NOT
        ground truth. The intersection angle is randomized:
          - near-tangent (5-15 deg): hardest case, wide overlap-within-eps zone
          - moderate (30-60 deg): typical wall/ramp
          - perpendicular (80-90 deg): clean wall, easy to separate
        """
        angle_deg = self.rng.choice([
            self.rng.uniform(5, 15),
            self.rng.uniform(30, 60),
            self.rng.uniform(80, 90),
        ])
        angle_rad = np.radians(angle_deg)

        rotate_axis = u * self.rng.normal() + v * self.rng.normal()
        rotate_axis /= np.linalg.norm(rotate_axis) + 1e-8

        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        n_wall = (n_ground * cos_a +
                  np.cross(rotate_axis, n_ground) * sin_a +
                  rotate_axis * np.dot(rotate_axis, n_ground) * (1 - cos_a))
        n_wall = n_wall / (np.linalg.norm(n_wall) + 1e-8)

        d_wall = d_ground + self.rng.uniform(-0.5, 0.5)
        u2, v2 = self._create_plane_frame(n_wall)
        p0_wall = d_wall * n_wall

        coeffs = self.rng.uniform(-box_size / 3, box_size / 3, (num_points, 2))
        wall_pts = p0_wall + coeffs[:, 0:1] * u2 + coeffs[:, 1:2] * v2

        wall_noise = self.rng.normal(0, 0.02, num_points)
        wall_pts = wall_pts + np.outer(wall_noise, n_wall)

        return wall_pts.astype(np.float32), float(angle_deg)

    def _sample_normal(self, orientation, slope_angle_deg):
        if orientation == "ground":
            # tilted "floor" plane, mostly facing up
            angle_rad = np.radians(slope_angle_deg)
            n = np.array([0.0, np.sin(angle_rad), np.cos(angle_rad)])
        elif orientation == "vertical":
            # wall-like plane, normal lies in the XY plane, random azimuth
            az = self.rng.uniform(0, 2 * np.pi)
            n = np.array([np.cos(az), np.sin(az), 0.0])
        elif orientation == "random":
            # fully arbitrary orientation, uniform on sphere
            n = self.rng.normal(size=3)
        else:
            raise ValueError(f"Unknown orientation: {orientation}")
        return n / np.linalg.norm(n)

    def generate_scene(self, num_points=10000, inlier_ratio=0.5, noise_sigma=0.01,
                       slope_angle_deg=0.0, box_size=10.0, orientation="ground",
                       noise_type="gaussian", num_bumps=0, num_cylinders=0, num_boxes=0,
                       add_intersecting_plane=False, num_sine_waves=0, return_labels=False):
        """
        Generate a synthetic scene.

        return_labels=True additionally returns a 5th value: a per-point
        int array (see SOURCE_* constants above) identifying what each
        point actually is -- flat plane, bump/crater, sinusoidal terrain,
        cylinder, box, intersecting wall, or background noise. Opt-in and
        defaults off so the return signature -- and every existing caller
        that unpacks exactly 4 values (synthetic_env.py, evaluate_synthetic.py,
        check_eps_signal.py, etc.) -- is completely unaffected. Purely for
        visualization; nothing in the RL loop reads this.
        """
        num_inliers = int(num_points * inlier_ratio)
        num_outliers = num_points - num_inliers

        # 1. True plane normal (now supports ground / vertical / random)
        n_true = self._sample_normal(orientation, slope_angle_deg).astype(np.float32)

        # 2. Build an orthonormal in-plane basis {u, v}
        u, v = self._create_plane_frame(n_true)

        d_true = self.rng.uniform(-box_size / 4, box_size / 4)
        p0 = d_true * n_true  # a point satisfying n_true . p0 = d_true

        # 3. Sample inliers directly in the plane's local (u, v) coordinates
        coeffs = self.rng.uniform(-box_size / 2, box_size / 2, (num_inliers, 2))
        inliers = p0 + coeffs[:, 0:1] * u + coeffs[:, 1:2] * v

        # 3b. Surface deformations (terrain bumps/craters) BEFORE noise -- still
        # ground-truth inliers, just non-planar terrain rather than clutter.
        bump_displacement = np.zeros(num_inliers)
        wave_displacement = np.zeros(num_inliers)
        if num_bumps > 0:
            inliers, bump_displacement = self._add_surface_deformations(inliers, coeffs, n_true, num_bumps, box_size)
        if num_sine_waves > 0:
            inliers, wave_displacement = self._add_sinusoidal_deformation(inliers, coeffs, n_true, num_sine_waves, box_size)

        # 4. Perturb each point along the normal (noise_type-dependent)
        inliers = self._apply_noise(inliers, n_true, noise_sigma, noise_type)
        inliers = inliers.astype(np.float32)

        if return_labels:
            inlier_labels = np.full(num_inliers, SOURCE_FLAT_INLIER, dtype=np.int64)
            inlier_labels[bump_displacement > _DEFORMATION_LABEL_THRESHOLD] = SOURCE_BUMP
            # Wave checked after bump so a point touched by both ends up
            # labeled by whichever is visually dominant is irrelevant here --
            # simple priority (wave wins ties) is enough for a display label.
            inlier_labels[wave_displacement > _DEFORMATION_LABEL_THRESHOLD] = SOURCE_WAVE

        # 5. Generate outlier points
        outliers = self.rng.uniform(-box_size, box_size, (num_outliers, 3)).astype(np.float32)
        if return_labels:
            outlier_labels = np.full(num_outliers, SOURCE_BACKGROUND, dtype=np.int64)

        # 5b. Structured clutter (cylinders/boxes) and an intersecting plane
        # (wall/ramp) replace some of the uniform outliers -- num_points stays
        # constant. Both are gt_mask=False: false-positive traps, not ground
        # truth. Written into disjoint slices of `outliers` via a running
        # offset -- writing both to the same starting indices would let the
        # second silently overwrite the first instead of raising any error.
        offset = 0

        clutter_points = []
        clutter_labels = []
        for _ in range(num_cylinders):
            base_uv = self.rng.uniform(-box_size / 3, box_size / 3, 2)
            base_3d = p0 + base_uv[0] * u + base_uv[1] * v
            pts = self._generate_cylinder(base_3d, n_true, num_points=int(self.rng.integers(50, 200)))
            clutter_points.append(pts)
            clutter_labels.append(np.full(len(pts), SOURCE_CYLINDER, dtype=np.int64))
        for _ in range(num_boxes):
            base_uv = self.rng.uniform(-box_size / 3, box_size / 3, 2)
            base_3d = p0 + base_uv[0] * u + base_uv[1] * v
            pts = self._generate_box(base_3d, n_true, num_points=int(self.rng.integers(50, 200)))
            clutter_points.append(pts)
            clutter_labels.append(np.full(len(pts), SOURCE_BOX, dtype=np.int64))
        if clutter_points:
            clutter = np.vstack(clutter_points)
            budget = min(len(clutter), num_outliers // 2, num_outliers - offset)
            outliers[offset:offset + budget] = clutter[:budget]
            if return_labels:
                clutter_lbl = np.concatenate(clutter_labels)
                outlier_labels[offset:offset + budget] = clutter_lbl[:budget]
            offset += budget

        if add_intersecting_plane:
            wall_pts, _intersection_angle = self._generate_intersecting_plane(
                n_true, d_true, p0, u, v, box_size, num_points=int(self.rng.integers(200, 800))
            )
            budget = min(len(wall_pts), num_outliers // 3, num_outliers - offset)
            outliers[offset:offset + budget] = wall_pts[:budget]
            if return_labels:
                outlier_labels[offset:offset + budget] = SOURCE_WALL
            offset += budget

        # 6. Combine and shuffle
        points = np.vstack([inliers, outliers])
        gt_mask = np.zeros(num_points, dtype=bool)
        gt_mask[:num_inliers] = True

        indices = np.arange(num_points)
        self.rng.shuffle(indices)

        points = points[indices]
        gt_mask = gt_mask[indices]

        if return_labels:
            source_labels = np.concatenate([inlier_labels, outlier_labels])[indices]
            return points, gt_mask, n_true, d_true, source_labels

        return points, gt_mask, n_true, d_true
