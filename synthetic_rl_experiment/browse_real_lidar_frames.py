"""
browse_real_lidar_frames.py

Interactive browser: cycles through random real LiDAR frames across every
scene in data/, running the synthetic-trained model's single-shot bypass
(see visualize_real_pointcloud.py's docstring for why single-shot) on each
one. Shows the ORIGINAL cloud (left, colored by height -- these per-frame
files have no RGB at all, confirmed via plyfile: only x/y/z properties, so
height is the most informative substitute available) and the PREDICTED
ground (right, green highlighted on the same cloud, offset sideways)
together in one window.

Uses O3DVisualizer (the newer gui-based renderer), not the classic
Visualizer, specifically because the classic one has no way to show text
in the window at all (confirmed: no title/text/label methods whatsoever).
O3DVisualizer has add_3d_label() and a settable .title, but no keyboard
callback hook (confirmed: no set_on_key) -- so navigation is a clickable
"Next Frame" menu action instead of a keypress.

No ground truth used here (only 6 of ~20 scenes have any) -- purely a
visual, exploratory tool.

Usage:
    python browse_real_lidar_frames.py --tag v2
"""
import os
import sys
import argparse
import pickle
import glob
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import matplotlib.cm as cm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "schnabel_cython"))

import schnabel_ransac
from ransac_env import load_ply_xyz
from features.scene_features import compute_scene_features
from synthetic_env import EPS_LEVELS, MIN_SUPPORT_LEVELS, NORM_THRESH_LEVELS

DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
HORIZONTAL_TOLERANCE_DEG = 30.0


def load_obs_normalizer(vecnormalize_path):
    if vecnormalize_path is None or not os.path.exists(vecnormalize_path):
        return lambda obs: obs
    with open(vecnormalize_path, "rb") as f:
        vec_normalize = pickle.load(f)
    obs_rms = vec_normalize.obs_rms
    clip_obs = vec_normalize.clip_obs
    epsilon = vec_normalize.epsilon
    return lambda obs: np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + epsilon),
                                -clip_obs, clip_obs).astype(np.float32)


def choose_params(model, normalize_obs, points):
    scene_feat = compute_scene_features(points)
    feedback_feat = np.zeros(10, dtype=np.float32)
    obs = np.concatenate([scene_feat, feedback_feat]).astype(np.float32)
    action, _ = model.predict(normalize_obs(obs), deterministic=True)
    eps = EPS_LEVELS[int(action[0])]
    min_supp = max(1, int(round(MIN_SUPPORT_LEVELS[int(action[1])] * len(points))))
    norm_th = NORM_THRESH_LEVELS[int(action[2])]
    return eps, min_supp, norm_th


def select_shape(shapes, points):
    """Largest support among roughly-horizontal candidates -- see
    SESSION_PROGRESS_LOG.md sec.27."""
    candidates = []
    for shape in shapes:
        mask = shape["inlier_mask"]
        plane_pts = points[mask]
        cov = np.cov(plane_pts.T)
        evals, evecs = np.linalg.eig(cov)
        normal = evecs[:, np.argmin(evals)]
        angle_from_vertical = np.degrees(np.arccos(np.clip(abs(normal[2]), 0.0, 1.0)))
        candidates.append((shape, angle_from_vertical, int(np.sum(mask))))
    horizontal = [c for c in candidates if c[1] <= HORIZONTAL_TOLERANCE_DEG]
    pool = horizontal if horizontal else candidates
    best_shape, best_angle, _ = max(pool, key=lambda c: c[2])
    return best_shape, best_angle, bool(horizontal)


def height_colormap(points):
    """These per-frame files have no RGB at all (confirmed via plyfile:
    only x/y/z properties) -- height is the most informative substitute
    for 'what does the original cloud actually look like' available."""
    z = points[:, 2]
    z_min, z_max = z.min(), z.max()
    normalized = (z - z_min) / (z_max - z_min + 1e-8)
    return cm.viridis(normalized)[:, :3]


def discover_frame_pool():
    pool = []
    for scene in sorted(os.listdir(DATA_ROOT)):
        lidar_dir = os.path.join(DATA_ROOT, scene, "Data_omni", "P0000", "lidar")
        if not os.path.isdir(lidar_dir):
            continue
        for ply_path in glob.glob(os.path.join(lidar_dir, "*.ply")):
            pool.append((scene, ply_path))
    return pool


class FrameBrowser:
    def __init__(self, model, normalize_obs, frame_pool):
        self.model = model
        self.normalize_obs = normalize_obs
        self.frame_pool = frame_pool
        self.rng = np.random.default_rng()
        self._has_geometry = False

        self.window = o3d.visualization.O3DVisualizer("Real LiDAR browser", 1600, 900)
        self.window.point_size = 2
        self.window.add_action("Next random frame", self._on_next)
        self.window.set_on_close(self._on_close)

        self._on_next(self.window)

    def _on_close(self):
        return True  # allow the window to close normally

    def _on_next(self, vis):
        scene, ply_path = self.frame_pool[self.rng.integers(len(self.frame_pool))]
        frame_name = os.path.basename(ply_path)
        points = load_ply_xyz(ply_path)  # voxel-downsampled -- keeps rendering/RANSAC fast

        eps, min_supp, norm_th = choose_params(self.model, self.normalize_obs, points)
        try:
            shapes, _ = schnabel_ransac.detect(
                points, shapes=["plane"], relative_epsilon=False,
                epsilon=eps, normal_thresh=norm_th, min_support=min_supp,
                probability=0.001, normal_knn=20, max_shapes=10,
            )
        except Exception as e:
            shapes = []
            print(f"  RANSAC error: {e}")

        colors_pred = np.full((len(points), 3), [0.75, 0.75, 0.75])
        if shapes:
            best_shape, angle, was_horizontal = select_shape(shapes, points)
            pred_mask = best_shape["inlier_mask"]
            colors_pred[pred_mask] = [0.0, 0.85, 0.0]
            status = (f"FOUND: {int(pred_mask.sum())} pts ({pred_mask.mean()*100:.1f}%), "
                      f"{angle:.1f} deg from vertical, horizontal_candidate={was_horizontal}")
        else:
            status = "NOT FOUND"

        colors_orig = height_colormap(points)

        # Side-by-side: predicted copy offset along the cloud's own widest
        # horizontal axis so the two never overlap regardless of scene shape.
        extent = points.max(axis=0) - points.min(axis=0)
        offset = np.zeros(3, dtype=np.float32)
        offset[0] = extent[0] * 1.15 if extent[0] >= extent[1] else 0.0
        offset[1] = extent[1] * 1.15 if extent[1] > extent[0] else 0.0

        pcd_original = o3d.geometry.PointCloud()
        pcd_original.points = o3d.utility.Vector3dVector(points)
        pcd_original.colors = o3d.utility.Vector3dVector(colors_orig)

        pcd_predicted = o3d.geometry.PointCloud()
        pcd_predicted.points = o3d.utility.Vector3dVector(points + offset)
        pcd_predicted.colors = o3d.utility.Vector3dVector(colors_pred)

        if self._has_geometry:
            self.window.remove_geometry("original")
            self.window.remove_geometry("predicted")
        self.window.add_geometry("original", pcd_original)
        self.window.add_geometry("predicted", pcd_predicted)
        self._has_geometry = True

        # On-screen text: window title (always visible in the title bar) AND
        # a floating 3D label positioned above each cloud, so the scene name
        # is readable both ways -- this is the actual point of this rewrite.
        self.window.title = f"{scene} / {frame_name}  --  {status}"
        self.window.clear_3d_labels()
        label_z = points[:, 2].max() + 1.0
        self.window.add_3d_label(points.mean(axis=0) + [0, 0, label_z - points[:, 2].max()],
                                  f"{scene}\n(original, colored by height)")
        self.window.add_3d_label((points + offset).mean(axis=0) + [0, 0, label_z - points[:, 2].max()],
                                  f"{scene}\n(predicted ground)")

        self.window.reset_camera_to_default()
        self.window.post_redraw()

        print(f"[{scene}] {frame_name} ({len(points)} pts) -- "
              f"eps={eps} min_supp={min_supp} norm_th={norm_th} -> {status}")

    def run(self):
        print("Click 'Next random frame' in the window's action panel to advance. Close the window to quit.\n")
        gui.Application.instance.add_window(self.window)
        gui.Application.instance.run()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, default="v2")
    args = parser.parse_args()

    model_path = os.path.join(os.path.dirname(DATA_ROOT), "models", f"synthetic_ppo_{args.tag}.zip")
    vecnormalize_path = model_path.replace(".zip", "_vecnormalize.pkl")
    print(f"Loading model: {model_path}")
    from stable_baselines3 import PPO
    model = PPO.load(model_path, device="cpu")
    normalize_obs = load_obs_normalizer(vecnormalize_path)

    frame_pool = discover_frame_pool()
    print(f"Discovered {len(frame_pool)} frames across "
          f"{len(set(s for s, _ in frame_pool))} scenes.")

    gui.Application.instance.initialize()
    browser = FrameBrowser(model, normalize_obs, frame_pool)
    browser.run()


if __name__ == "__main__":
    main()
