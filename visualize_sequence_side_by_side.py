import argparse
import glob
import os
import sys

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from stable_baselines3 import PPO

from features.scene_features import compute_scene_features
from ransac_env import EPS_LEVELS, MIN_SUPPORT_LEVELS, NORM_THRESH_LEVELS, find_ground_plane
from rl_evaluator import load_obs_normalizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "schnabel_cython"))
import schnabel_ransac


WORKSPACE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(WORKSPACE, "data")
MODEL_DIR = os.path.join(WORKSPACE, "models")
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, "ppo_ransac_v4_model_final.zip")
DEFAULT_VECNORMALIZE_PATH = os.path.join(MODEL_DIR, "ppo_ransac_v4_model_final_vecnormalize.pkl")


def list_frame_files(dataset):
    lidar_dir = os.path.join(DATA_ROOT, dataset, "Data_omni", "P0000", "lidar")
    files = sorted(glob.glob(os.path.join(lidar_dir, "*.ply")))
    if not files:
        raise FileNotFoundError(f"No .ply frames found under {lidar_dir}")
    return files


def load_frame_cloud(filepath, voxel_size=0.05):
    pcd = o3d.io.read_point_cloud(filepath)
    if voxel_size and voxel_size > 0.0:
        pcd = pcd.voxel_down_sample(voxel_size)

    points = np.asarray(pcd.points, dtype=np.float32)
    colors = np.asarray(pcd.colors, dtype=np.float32)
    if len(colors) != len(points):
        colors = np.zeros((0, 3), dtype=np.float32)
    return points, colors


def build_point_cloud(points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    return pcd


def make_original_colors(points, file_colors):
    if len(file_colors) == len(points) and len(file_colors) > 0:
        return file_colors
    return np.full((len(points), 3), 0.70, dtype=np.float32)


def run_model_on_points(points, model, normalize_obs, z_mode):
    features = compute_scene_features(points)
    feedback_feat = np.zeros(10, dtype=np.float32)
    obs = np.concatenate([features, feedback_feat])

    action, _ = model.predict(normalize_obs(obs), deterministic=True)
    eps = EPS_LEVELS[int(action[0])]
    min_supp = MIN_SUPPORT_LEVELS[int(action[1])]
    norm_th = NORM_THRESH_LEVELS[int(action[2])]

    shapes, _ = schnabel_ransac.detect(
        points,
        shapes=["plane"],
        relative_epsilon=False,
        epsilon=eps,
        normal_thresh=norm_th,
        min_support=min_supp,
        probability=0.001,
        normal_knn=20,
        max_shapes=20,
    )
    ground_shape, avg_z, z_align, residual = find_ground_plane(shapes, points, z_mode=z_mode)

    pred_mask = np.zeros(len(points), dtype=bool)
    if ground_shape is not None:
        pred_mask = ground_shape["inlier_mask"].copy()

    return {
        "epsilon": eps,
        "min_support": min_supp,
        "normal_thresh": norm_th,
        "ground_shape": ground_shape,
        "pred_mask": pred_mask,
        "avg_z": avg_z,
        "z_align": z_align,
        "residual": residual,
    }


def make_segmented_colors(points, pred_mask):
    colors = np.full((len(points), 3), [0.70, 0.70, 0.70], dtype=np.float32)
    colors[pred_mask] = [0.10, 0.80, 0.15]
    return colors


def compute_metrics(pred_mask, gt_mask):
    pred = np.asarray(pred_mask, dtype=bool)
    gt = np.asarray(gt_mask, dtype=bool)
    tp = int(np.sum(pred & gt))
    fp = int(np.sum(pred & ~gt))
    fn = int(np.sum(~pred & gt))
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return iou, precision, recall


class SequenceViewerApp:
    def __init__(self, dataset, model_path, vecnormalize_path, start_index=0,
                 voxel_size=0.05, point_size=2.5, z_mode="z_down"):
        self.dataset = dataset
        self.files = list_frame_files(dataset)
        self.index = max(0, min(start_index, len(self.files) - 1))
        self.voxel_size = voxel_size
        self.point_size = point_size
        self.z_mode = z_mode
        self.cache = {}

        self.model = PPO.load(model_path)
        self.normalize_obs = load_obs_normalizer(vecnormalize_path)

        self.app = gui.Application.instance
        self.app.initialize()
        self.window = self.app.create_window(
            f"{dataset} - Original vs RL Ground Segmentation", 1800, 980
        )
        self.window.set_on_layout(self._on_layout)
        self.window.set_on_key(self._on_key)

        em = self.window.theme.font_size
        margins = gui.Margins(0.5 * em, 0.5 * em, 0.5 * em, 0.5 * em)

        self.root = gui.Vert(0.35 * em, margins)
        self.window.add_child(self.root)

        title = gui.Label("3D Frame Viewer")
        title.text_color = gui.Color(0.10, 0.10, 0.10)
        self.root.add_child(title)

        self.status_label = gui.Label("")
        self.root.add_child(self.status_label)

        self.metrics_label = gui.Label("")
        self.root.add_child(self.metrics_label)

        button_row = gui.Horiz(0.25 * em)
        self.prev_button = gui.Button("Previous Frame")
        self.prev_button.set_on_clicked(self.prev_frame)
        self.next_button = gui.Button("Next Frame")
        self.next_button.set_on_clicked(self.next_frame)
        button_row.add_child(self.prev_button)
        button_row.add_child(self.next_button)
        self.root.add_child(button_row)

        headers = gui.Horiz(0.5 * em)
        self.left_header = gui.Label("Original Frame")
        self.right_header = gui.Label("Detected Ground Plane")
        headers.add_child(self.left_header)
        headers.add_stretch()
        headers.add_child(self.right_header)
        self.root.add_child(headers)

        self.scenes_row = gui.Horiz(0.5 * em)
        self.left_scene = gui.SceneWidget()
        self.left_scene.scene = rendering.Open3DScene(self.window.renderer)
        self.left_scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)
        self.left_scene.scene.set_background([1.0, 1.0, 1.0, 1.0])
        self.left_scene.scene.show_axes(True)

        self.right_scene = gui.SceneWidget()
        self.right_scene.scene = rendering.Open3DScene(self.window.renderer)
        self.right_scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)
        self.right_scene.scene.set_background([1.0, 1.0, 1.0, 1.0])
        self.right_scene.scene.show_axes(True)

        self.scenes_row.add_child(self.left_scene)
        self.scenes_row.add_child(self.right_scene)
        self.root.add_child(self.scenes_row)

        self.original_material = rendering.MaterialRecord()
        self.original_material.shader = "defaultUnlit"
        self.original_material.point_size = point_size

        self.segmented_material = rendering.MaterialRecord()
        self.segmented_material.shader = "defaultUnlit"
        self.segmented_material.point_size = point_size

        self._show_frame(self.index)

    def _on_layout(self, layout_context):
        r = self.window.content_rect
        self.root.frame = r
        
        # gui.Vert relies on calc_preferred_size, which is 0 for SceneWidget.
        # We manually expand the scenes_row to take up the rest of the window.
        y_start = self.scenes_row.frame.y
        h = max(1, r.height - (y_start - r.y))
        w = max(1, r.width)
        
        self.scenes_row.frame = gui.Rect(r.x, y_start, w, h)
        
        spacing = int(round(self.window.theme.font_size * 0.5))
        scene_w = (w - spacing) // 2
        
        self.left_scene.frame = gui.Rect(r.x, y_start, scene_w, h)
        self.right_scene.frame = gui.Rect(r.x + scene_w + spacing, y_start, w - scene_w - spacing, h)

    def _on_key(self, event):
        if event.type == gui.KeyEvent.Type.DOWN:
            if event.key == gui.KeyName.RIGHT:
                self.next_frame()
                return True
            if event.key == gui.KeyName.LEFT:
                self.prev_frame()
                return True
        return False

    def prev_frame(self):
        if self.index > 0:
            self._show_frame(self.index - 1)

    def next_frame(self):
        if self.index < len(self.files) - 1:
            self._show_frame(self.index + 1)

    def _load_frame_data(self, frame_idx):
        if frame_idx in self.cache:
            return self.cache[frame_idx]

        filepath = self.files[frame_idx]
        filename = os.path.basename(filepath)
        points, file_colors = load_frame_cloud(filepath, voxel_size=self.voxel_size)
        original_colors = make_original_colors(points, file_colors)

        result = run_model_on_points(points, self.model, self.normalize_obs, self.z_mode)
        segmented_colors = make_segmented_colors(points, result["pred_mask"])

        gt_mask = None
        gt_fraction = None
        iou = precision = recall = None
        gt_path = filepath.replace(".ply", "_gt_mask.npy")
        if os.path.exists(gt_path):
            temp_mask = np.load(gt_path)
            if len(temp_mask) == len(points):
                gt_mask = temp_mask
                gt_fraction = float(np.mean(gt_mask)) if len(gt_mask) > 0 else 0.0
                iou, precision, recall = compute_metrics(result["pred_mask"], gt_mask)
            else:
                print(f"Warning: GT mask size ({len(temp_mask)}) != points size ({len(points)}). Skipping metrics.")

        pred_fraction = float(np.mean(result["pred_mask"])) if len(result["pred_mask"]) > 0 else 0.0
        data = {
            "filepath": filepath,
            "filename": filename,
            "points": points,
            "original_pcd": build_point_cloud(points, original_colors),
            "segmented_pcd": build_point_cloud(points, segmented_colors),
            "pred_fraction": pred_fraction,
            "gt_fraction": gt_fraction,
            "iou": iou,
            "precision": precision,
            "recall": recall,
            "epsilon": result["epsilon"],
            "min_support": result["min_support"],
            "normal_thresh": result["normal_thresh"],
            "residual": result["residual"],
        }
        self.cache[frame_idx] = data
        return data

    def _show_frame(self, frame_idx):
        self.index = frame_idx
        data = self._load_frame_data(frame_idx)

        self.left_scene.scene.clear_geometry()
        self.right_scene.scene.clear_geometry()
        self.left_scene.scene.add_geometry("original", data["original_pcd"], self.original_material)
        self.right_scene.scene.add_geometry("segmented", data["segmented_pcd"], self.segmented_material)

        bbox = data["original_pcd"].get_axis_aligned_bounding_box()
        center = bbox.get_center().astype(np.float32)
        self.left_scene.setup_camera(60.0, bbox, center)
        self.right_scene.setup_camera(60.0, bbox, center)

        self.status_label.text = (
            f"Dataset: {self.dataset} | Frame {self.index + 1}/{len(self.files)} | "
            f"{data['filename']} | Left/Right arrow = previous/next frame | Mouse = orbit/pan/zoom"
        )

        metrics = (
            f"Model params: eps={data['epsilon']:.2f}, min_support={data['min_support']}, "
            f"normal_thresh={data['normal_thresh']:.2f} | "
            f"Predicted ground={data['pred_fraction'] * 100:.2f}%"
        )
        if data["gt_fraction"] is not None:
            metrics += (
                f" | GT ground={data['gt_fraction'] * 100:.2f}% | "
                f"IoU={data['iou']:.4f} | Precision={data['precision']:.4f} | Recall={data['recall']:.4f}"
            )
        if data["residual"] is not None:
            metrics += f" | Residual={data['residual']:.4f}"
        self.metrics_label.text = metrics

        self.left_header.text = "Original Frame"
        self.right_header.text = "RL Segmented Ground Plane"
        self.window.title = (
            f"{self.dataset} | {data['filename']} | "
            f"Original (left) vs RL segmented ground plane (right)"
        )

    def run(self):
        self.app.run()


def main():
    parser = argparse.ArgumentParser(
        description="Interactive side-by-side 3D frame viewer for original vs RL ground segmentation"
    )
    parser.add_argument("--dataset", type=str, default="Gascola",
                        help="Dataset folder under data/, e.g. Gascola or NordicHarbor")
    parser.add_argument("--frame_index", type=int, default=0,
                        help="0-based starting frame index")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH,
                        help="Path to the trained PPO model")
    parser.add_argument("--vecnormalize", type=str, default=DEFAULT_VECNORMALIZE_PATH,
                        help="Path to the matching VecNormalize stats")
    parser.add_argument("--voxel_size", type=float, default=0.05,
                        help="Voxel downsampling size used for display and inference")
    parser.add_argument("--point_size", type=float, default=2.5,
                        help="Rendered point size")
    parser.add_argument("--z_mode", type=str, default="z_down", choices=["z_down", "z_up"],
                        help="Ground selection convention to use for the segmentation")
    args = parser.parse_args()

    viewer = SequenceViewerApp(
        dataset=args.dataset,
        model_path=args.model,
        vecnormalize_path=args.vecnormalize,
        start_index=args.frame_index,
        voxel_size=args.voxel_size,
        point_size=args.point_size,
        z_mode=args.z_mode,
    )
    viewer.run()


if __name__ == "__main__":
    main()
