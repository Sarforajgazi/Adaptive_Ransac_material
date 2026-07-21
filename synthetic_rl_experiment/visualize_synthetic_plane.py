import sys
import os
import argparse
import pickle
import numpy as np
import open3d as o3d

# Add parent dir to path so we can import schnabel
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "schnabel_cython"))

import schnabel_ransac
from data_generator import (
    SyntheticPlaneGenerator, save_scene,
    SOURCE_FLAT_INLIER, SOURCE_BUMP, SOURCE_WAVE, SOURCE_CYLINDER, SOURCE_BOX, SOURCE_WALL,
)
from synthetic_env import SyntheticRansacEnv

SAVED_SCENES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_scenes")

# Obstacle/terrain-type colors, keyed by data_generator.py's SOURCE_* codes.
# Deliberately distinct from yellow/green (TP/FN) and red (false positive)
# and from the blue/purple plane wireframe colors, so nothing gets confused
# for something else at a glance.
SOURCE_COLOR = {
    SOURCE_BUMP: [1.0, 0.55, 0.0],      # orange
    SOURCE_WAVE: [0.0, 0.6, 0.5],       # teal
    SOURCE_CYLINDER: [0.55, 0.27, 0.07],  # brown
    SOURCE_BOX: [0.9, 0.4, 0.7],        # pink
    SOURCE_WALL: [0.0, 0.85, 0.95],     # cyan
}

def create_plane_mesh(normal, d, color, size=10.0):
    """
    Creates an Open3D triangle mesh of a plane for visualization.
    Plane equation: normal . p = d
    """
    # Create a simple grid in the XY plane
    mesh = o3d.geometry.TriangleMesh.create_box(width=size, height=size, depth=0.01)
    mesh.translate([-size/2, -size/2, 0])
    
    # The normal of the default box top face is [0, 0, 1].
    # We need to rotate [0,0,1] to match `normal`.
    z_axis = np.array([0, 0, 1])
    normal = normal / np.linalg.norm(normal)
    
    # Calculate rotation matrix from z_axis to normal
    v = np.cross(z_axis, normal)
    c = np.dot(z_axis, normal)
    if c < -0.9999: # 180 degree rotation
        R = np.array([[-1,0,0],[0,1,0],[0,0,-1]])
    elif c > 0.9999: # 0 degree rotation
        R = np.eye(3)
    else:
        s = np.linalg.norm(v)
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + np.dot(vx, vx) * ((1 - c) / (s ** 2))
        
    mesh.rotate(R, center=(0,0,0))
    
    # Translate along the normal by distance d
    mesh.translate(normal * d)
    
    # Set color and transparency
    mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()
    return mesh

def create_plane_wireframe(normal, d, color, size=8.0):
    """
    Outlines a plane as a simple 4-edge square LineSet instead of a filled
    mesh -- marks the plane's location/orientation as a border only, so it
    doesn't occlude the real (possibly correctly-classified) points that
    sit on the plane's surface underneath it. A solid mesh there was hiding
    genuine yellow (true-positive) points behind an opaque marker, making
    a well-fit scene look like it had a "hole" in the middle when it didn't.
    Plane equation: normal . p = d.
    """
    normal = normal / np.linalg.norm(normal)
    helper = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, helper)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)

    center = normal * d
    half = size / 2
    corners = np.array([
        center + half * u + half * v,
        center + half * u - half * v,
        center - half * u - half * v,
        center - half * u + half * v,
    ])
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(corners),
        lines=o3d.utility.Vector2iVector([[0, 1], [1, 2], [2, 3], [3, 0]]),
    )
    line_set.colors = o3d.utility.Vector3dVector([color, color, color, color])
    return line_set


def load_obs_normalizer(vecnormalize_path):
    """Replicates SB3's VecNormalize._normalize_obs formula from saved stats."""
    if vecnormalize_path is None or not os.path.exists(vecnormalize_path):
        return lambda obs: obs
    with open(vecnormalize_path, "rb") as f:
        vec_normalize = pickle.load(f)
    obs_rms = vec_normalize.obs_rms
    clip_obs = vec_normalize.clip_obs
    epsilon = vec_normalize.epsilon
    return lambda obs: np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + epsilon),
                                -clip_obs, clip_obs).astype(np.float32)


def choose_params_rl(model_path, vecnormalize_path, scene_options):
    """
    Runs the trained RL policy through a full episode on a pinned scene and
    returns (pts, gt_mask, n_true, d_true, eps, min_supp, norm_th, n_steps,
    scene_composition, source_labels) -- the same scene the agent actually
    saw, its final chosen params, what reset() actually generated (including
    any scene-composition options left unset to randomize -- see
    synthetic_env.py's true_* readback attributes), and a per-point label of
    what each point actually is (bump/cylinder/box/wall/etc, see
    data_generator.py's SOURCE_* constants) for obstacle-type-colored
    visualization. track_source_labels=True on the env is what makes this
    the SAME scene the agent saw rather than a second, differently-random one.
    """
    from stable_baselines3 import PPO
    model = PPO.load(model_path, device="cpu")
    normalize_obs = load_obs_normalizer(vecnormalize_path)

    env = SyntheticRansacEnv(max_steps=5, track_source_labels=True)
    obs, _ = env.reset(options=scene_options)
    scene_composition = {
        "orientation": env.true_orientation,
        "noise_type": env.true_noise_type,
        "num_bumps": env.true_num_bumps,
        "num_cylinders": env.true_num_cylinders,
        "num_boxes": env.true_num_boxes,
        "add_intersecting_plane": env.true_add_intersecting_plane,
    }
    done = False
    info = None
    n_steps = 0
    while not done:
        action, _ = model.predict(normalize_obs(obs), deterministic=True)
        obs, reward, done, _, info = env.step(action)
        n_steps += 1

    return (env.current_points, env.current_gt_mask, env.n_true, env.d_true,
            info["epsilon"], info["min_support"], info["normal_thresh"], n_steps,
            scene_composition, env.current_source_labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None,
                         help="Path to a trained PPO model .zip. Overrides --tag if given.")
    parser.add_argument("--tag", type=str, default=None,
                         help="Visualize the model trained with this tag (e.g. 'v2'). Omit to "
                              "auto-pick the most recently trained synthetic_ppo*.zip.")
    parser.add_argument("--no_rl", action="store_true", help="Force the old hardcoded-parameter baseline.")
    parser.add_argument("--noise_sigma", type=float, default=0.1)
    parser.add_argument("--inlier_ratio", type=float, default=0.3)
    parser.add_argument("--slope_angle_deg", type=float, default=15.0)
    parser.add_argument("--orientation", type=str, default="ground", choices=["ground", "vertical", "random"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_scene", type=str, default=None,
                         help="Save the generated point cloud + ground truth to "
                              "saved_scenes/<name>.ply and <name>_meta.json for later reuse.")
    parser.add_argument("--save_screenshot", type=str, default=None,
                         help="Render off-screen and save a PNG to this path instead of opening "
                              "an interactive window -- for batch-generating report figures.")
    parser.add_argument("--transparent", action="store_true",
                         help="Render the true/fitted planes as real semi-transparent filled "
                              "surfaces (Open3D's material-based renderer) instead of wireframe "
                              "outlines -- see both the plane surface and the points through it.")
    # Phase 2-5 scene-composition controls. Each defaults to None/unset, which
    # means "let it randomize the same way training does" for the RL path
    # (goes through env.reset(), which fills in unset options itself) --
    # explicit values force that feature on so you can deliberately inspect
    # bumps/clutter/an intersecting plane rather than waiting for a random
    # scene to happen to include one.
    parser.add_argument("--noise_type", type=str, default=None,
                         choices=["gaussian", "laplacian", "uniform", "spatially_varying", "mixed"],
                         help="Force a specific noise model (see data_generator.py's _apply_noise). Omit to randomize.")
    parser.add_argument("--num_bumps", type=int, default=None, help="Force a specific number of terrain bumps/craters (0-3). Omit to randomize.")
    parser.add_argument("--num_cylinders", type=int, default=None, help="Force a specific number of cylinder clutter objects. Omit to randomize.")
    parser.add_argument("--num_boxes", type=int, default=None, help="Force a specific number of box clutter objects. Omit to randomize.")
    parser.add_argument("--add_intersecting_plane", action="store_true", help="Force an intersecting wall/ramp plane to be present.")
    args = parser.parse_args()

    model_path = args.model
    if model_path is None and not args.no_rl:
        from evaluate_synthetic import resolve_model_path
        model_path = resolve_model_path(args.tag)

    scene_options = {
        "noise_sigma": args.noise_sigma,
        "inlier_ratio": args.inlier_ratio,
        "slope_angle_deg": args.slope_angle_deg,
        "orientation": args.orientation,
        "generator_seed": args.seed,
    }
    if args.noise_type is not None:
        scene_options["noise_type"] = args.noise_type
    if args.num_bumps is not None:
        scene_options["num_bumps"] = args.num_bumps
    if args.num_cylinders is not None:
        scene_options["num_cylinders"] = args.num_cylinders
    if args.num_boxes is not None:
        scene_options["num_boxes"] = args.num_boxes
    if args.add_intersecting_plane:
        scene_options["add_intersecting_plane"] = True

    if model_path is not None:
        vecnormalize_path = model_path.replace(".zip", "_vecnormalize.pkl")
        print(f"Using RL model: {model_path}")
        print(f"Using VecNormalize stats: {vecnormalize_path if os.path.exists(vecnormalize_path) else '(none found -- raw obs)'}")
        pts, gt_mask, n_true, d_true, eps, min_supp, norm_th, n_steps, scene_composition, source_labels = choose_params_rl(
            model_path, vecnormalize_path, scene_options)
        print(f"RL agent chose after {n_steps} step(s): eps={eps}, min_support={min_supp}, normal_thresh={norm_th}")
        print(f"Scene actually generated: {scene_composition}")
    else:
        print("No RL model found -- using the fixed baseline (eps=0.15, min_support=100, normal_thresh=0.85).")
        gen = SyntheticPlaneGenerator(args.seed)
        pts, gt_mask, n_true, d_true, source_labels = gen.generate_scene(
            num_points=10000,
            inlier_ratio=args.inlier_ratio,
            noise_sigma=args.noise_sigma,
            slope_angle_deg=args.slope_angle_deg,
            orientation=args.orientation,
            noise_type=args.noise_type or "gaussian",
            num_bumps=args.num_bumps or 0,
            num_cylinders=args.num_cylinders or 0,
            num_boxes=args.num_boxes or 0,
            add_intersecting_plane=args.add_intersecting_plane,
            return_labels=True,
        )
        eps, min_supp, norm_th = 0.15, 100, 0.85
        scene_composition = {
            "orientation": args.orientation,
            "noise_type": args.noise_type or "gaussian",
            "num_bumps": args.num_bumps or 0,
            "num_cylinders": args.num_cylinders or 0,
            "num_boxes": args.num_boxes or 0,
            "add_intersecting_plane": args.add_intersecting_plane,
        }

    if args.save_scene:
        prefix = os.path.join(SAVED_SCENES_DIR, args.save_scene)
        ply_path, meta_path = save_scene(prefix, pts, gt_mask, n_true, d_true, meta={
            "noise_sigma": args.noise_sigma,
            "inlier_ratio": args.inlier_ratio,
            "slope_angle_deg": args.slope_angle_deg,
            "generator_seed": args.seed,
            **scene_composition,
        })
        print(f"Scene saved to {ply_path} and {meta_path}")

    print("Running RANSAC with the chosen parameters...")
    shapes, _ = schnabel_ransac.detect(
        pts,
        shapes=["plane"],
        relative_epsilon=False,
        epsilon=eps,
        normal_thresh=norm_th,
        min_support=min_supp,
        probability=0.001,
        normal_knn=20,
        max_shapes=5,
    )

    if not shapes:
        print("RANSAC found no planes!")
        return

    best_shape = shapes[0]
    pred_mask = best_shape["inlier_mask"]

    # Calculate RANSAC plane normal
    plane_pts = pts[pred_mask]
    cov = np.cov(plane_pts.T)
    evals, evecs = np.linalg.eig(cov)
    n_ransac = evecs[:, np.argmin(evals)]

    # Ensure normal points in similar direction to n_true for visual clarity
    if np.dot(n_ransac, n_true) < 0:
        n_ransac = -n_ransac

    mean_pt = np.mean(plane_pts, axis=0)
    d_ransac = np.dot(mean_pt, n_ransac)

    angle_error = np.degrees(np.arccos(np.clip(np.dot(n_true, n_ransac), 0.0, 1.0)))
    print(f"Angle Error: {angle_error:.2f} degrees")
    
    # --- Visualization ---
    geometries = []
    
    # 1. Point Cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    
    colors = np.zeros((len(pts), 3))
    colors[:] = [0.7, 0.7, 0.7]  # default: background outlier = gray

    # Obstacle/terrain-type colors take priority over the base gray, for any
    # source_labels this scene actually has -- shows WHAT each non-flat
    # point structurally is (a bump, a cylinder, the intersecting wall...),
    # not just whether it's an inlier. Applied before the flat inlier
    # TP/FN coloring below and before the false-positive red override, so:
    # a bump point keeps its bump color regardless of hit/miss (it's real
    # ground truth either way), but a clutter point RANSAC wrongly grabs
    # still gets flagged red -- "wrongly selected" matters more visually
    # than "what type of clutter it was."
    used_source_labels = source_labels is not None
    if used_source_labels:
        for label_value, color in SOURCE_COLOR.items():
            colors[source_labels == label_value] = color
        flat_mask = gt_mask & (source_labels == SOURCE_FLAT_INLIER)
    else:
        flat_mask = gt_mask

    # Flat inlier: yellow if RANSAC found it (TP), green if it missed it (FN)
    colors[flat_mask & pred_mask] = [1.0, 1.0, 0.0]
    colors[flat_mask & ~pred_mask] = [0.0, 0.8, 0.0]

    # False positive (RANSAC grabbed a point that isn't part of the true
    # plane) always shown red, overriding any obstacle-type color -- this
    # never touches bump/wave points since those ARE gt_mask=True.
    colors[pred_mask & ~gt_mask] = [1.0, 0.0, 0.0]

    pcd.colors = o3d.utility.Vector3dVector(colors)
    geometries.append(pcd)

    fitted_label = "RL-Agent-chosen" if model_path is not None else "Fixed-baseline"
    print("\nVisualizing...")
    print(f"- Green points: Flat true-plane points RANSAC missed")
    print(f"- Yellow points: Flat true-plane points RANSAC correctly found")
    print(f"- Red points: False positives (RANSAC mistakenly grabbed a non-plane point)")
    if used_source_labels:
        print(f"- Orange points: bump/crater terrain (still true ground, just deformed)")
        print(f"- Teal points: sinusoidal/wavy terrain (still true ground, just deformed)")
        print(f"- Brown points: cylinder clutter (false-positive trap)")
        print(f"- Pink points: box clutter (false-positive trap)")
        print(f"- Cyan points: intersecting plane/wall (false-positive trap)")
    print(f"- Gray points: background noise (correctly not part of the plane)")
    print(f"- Blue plane: The actual Ground Truth Plane")
    print(f"- Purple plane: The plane RANSAC fitted using {fitted_label} parameters (eps={eps}, min_support={min_supp}, normal_thresh={norm_th})")

    plane_size = 9.0

    if args.transparent:
        # Real alpha-blended filled planes, via Open3D's material-based
        # renderer (o3d.visualization.rendering) -- the legacy Visualizer
        # used below can't alpha-blend meshes at all. Filled + transparent
        # means it shows the actual plane *surface*, not just a boundary
        # outline, while still letting the colored points show through it.
        true_mesh = create_plane_mesh(n_true, d_true, color=[0.2, 0.5, 0.8], size=plane_size)
        fitted_mesh = create_plane_mesh(n_ransac, d_ransac, color=[0.8, 0.2, 0.8], size=plane_size)

        pcd_material = o3d.visualization.rendering.MaterialRecord()
        pcd_material.shader = "defaultUnlit"
        pcd_material.point_size = 4.0

        true_material = o3d.visualization.rendering.MaterialRecord()
        true_material.shader = "defaultLitTransparency"
        true_material.base_color = [0.2, 0.5, 0.8, 0.35]

        fitted_material = o3d.visualization.rendering.MaterialRecord()
        fitted_material.shader = "defaultLitTransparency"
        fitted_material.base_color = [0.8, 0.2, 0.8, 0.35]

        named_geoms = [
            {"name": "cloud", "geometry": pcd, "material": pcd_material},
            {"name": "true_plane", "geometry": true_mesh, "material": true_material},
            {"name": "fitted_plane", "geometry": fitted_mesh, "material": fitted_material},
        ]

        if args.save_screenshot:
            os.makedirs(os.path.dirname(os.path.abspath(args.save_screenshot)) or ".", exist_ok=True)
            renderer = o3d.visualization.rendering.OffscreenRenderer(1280, 960)
            renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
            for g in named_geoms:
                renderer.scene.add_geometry(g["name"], g["geometry"], g["material"])
            bbox = pcd.get_axis_aligned_bounding_box()
            renderer.setup_camera(60.0, bbox, bbox.get_center())
            img = renderer.render_to_image()
            o3d.io.write_image(args.save_screenshot, img)
            print(f"Screenshot saved to {args.save_screenshot}")
        else:
            o3d.visualization.draw(named_geoms)
        return

    # Default: wireframe outlines, not filled meshes. A filled mesh in the
    # legacy Visualizer renders fully opaque and hides the real points
    # underneath it -- including genuine yellow (correctly recovered)
    # points, making a well-fit scene look like it has a "hole" where the
    # marker sits. A border-only outline marks the plane's location and
    # extent without covering anything, so it can span close to the full
    # cloud extent instead of being shrunk down. Nudged a few cm apart along
    # their own normals so two near-identical outlines (a good fit) don't
    # perfectly overlap into an unreadable single line.
    true_mesh = create_plane_wireframe(n_true, d_true + 0.03, color=[0.2, 0.5, 0.8], size=plane_size)
    geometries.append(true_mesh)
    fitted_mesh = create_plane_wireframe(n_ransac, d_ransac - 0.03, color=[0.8, 0.2, 0.8], size=plane_size)
    geometries.append(fitted_mesh)

    if args.save_screenshot:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_screenshot)) or ".", exist_ok=True)
        # Hidden-window rendering (not true headless -- needs a display, fine
        # on a desktop session) so this can be scripted in a loop to
        # batch-generate consistently-framed report figures instead of
        # manually screenshotting an interactive window per scene.
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=1280, height=960)
        for g in geometries:
            vis.add_geometry(g)
        render_opt = vis.get_render_option()
        render_opt.point_size = 2.0
        render_opt.line_width = 3.0
        ctr = vis.get_view_control()
        ctr.set_zoom(0.7)
        # Oblique angle instead of the default top-down-ish view -- shows the
        # point spread and the plane outlines' 3D orientation instead of
        # looking straight down the normal, which would flatten the outlines
        # into a view-on edge with no sense of depth.
        ctr.rotate(200.0, 100.0)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(args.save_screenshot, do_render=True)
        vis.destroy_window()
        print(f"Screenshot saved to {args.save_screenshot}")
    else:
        o3d.visualization.draw_geometries(geometries)

if __name__ == "__main__":
    main()
