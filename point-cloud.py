import os
import open3d as o3d
import yaml
import numpy as np

# ==============================================================================
# Configuration: File Paths
# ==============================================================================
OBJ_PATH  = "dataset/liver-registration/01/model/reconstructed_mesh_world_m.obj"
CAMK_PATH = "dataset/liver-registration/01/syn/CAM_K.yml"
CAMP_PATH = "dataset/liver-registration/01/syn/camPose.yml"
PCD_PATH   = "dataset/liver-registration/01/syn/liverPcds"

# 1. Load the 3D CT liver mesh
mesh = o3d.io.read_triangle_mesh(OBJ_PATH)
mesh.compute_vertex_normals()
mesh.paint_uniform_color([0.65, 0.28, 0.22])  # liver reddish-brown

# 2. Load camera intrinsics and poses from YAML files
with open(CAMK_PATH) as f:
    cam_k = yaml.safe_load(f)

with open(CAMP_PATH) as f:
    poses = yaml.safe_load(f)

total_frames = len(poses)
current_frame = 0


def get_camera_data(frame_idx: int):
    """
    Extracts camera parameters and builds the 3D viewing frustum:
      - Reads intrinsic matrix K and extrinsic rotation R, translation t.
      - Converts Blender camera coordinates (-Z forward) to standard OpenCV (+Z forward).
      - Returns:
          1. cam_vis : 3D wireframe pyramid of the camera.
          2. K       : 3x3 intrinsic matrix.
          3. T_w2c   : 4x4 World-to-Camera transformation matrix.
          4. t       : 3D position vector in meters.
    """
    key = f"{frame_idx:05d}"
    K = np.array(cam_k[key][0]["cam_K"]).reshape(3, 3)
    p = poses[key][0]

    # Convert Blender (-Z forward) to OpenCV (+Z forward)
    R_cv = np.array(p["cam_R_c2w"]).reshape(3, 3) @ np.diag([1, -1, -1])
    t = np.array(p["cam_t_c2w"]).reshape(3) / 1000.0  # mm to meters

    # Build 4x4 World-to-Camera matrix
    T_w2c = np.eye(4)
    T_w2c[:3, :3] = R_cv.T
    T_w2c[:3, 3] = -R_cv.T @ t

    cam_vis = o3d.geometry.LineSet.create_camera_visualization(960, 540, K, T_w2c, scale=0.15)
    return cam_vis, K, T_w2c, t


def get_point_cloud(frame_idx: int):
    """
    Loads the visible surface point cloud (ptsLiverCam) for the given frame:
      - Reads the binary PLY point cloud stored in camera coordinates (mm).
      - Scales coordinates from mm to meters (/ 1000).
      - Transforms the points from Camera Space into World Space using [R_c2w | t_c2w]
        so the points sit directly on top of the preoperative 3D liver surface!
      - Paints the points in bright glowing gold so they stand out on the liver.
    """
    ply_path = os.path.join(PCD_PATH, f"ptsLiverCam{frame_idx:05d}.ply")
    pcd = o3d.io.read_point_cloud(ply_path)
    pts = np.asarray(pcd.points)

    # Nudge points 0.8 mm toward camera lens to prevent the opaque mesh from burying them
    unit_to_cam = -pts / np.linalg.norm(pts, axis=1, keepdims=True)
    pts_nudged = pts + 0.8 * unit_to_cam

    # Scale from mm to meters
    pts_m = pts_nudged / 1000.0

    # Transform Camera Space -> World Space
    p = poses[f"{frame_idx:05d}"][0]
    R_c2w = np.array(p["cam_R_c2w"]).reshape(3, 3)
    t_c2w_m = np.array(p["cam_t_c2w"]).reshape(3) / 1000.0

    pts_world = (R_c2w @ pts_m.T).T + t_c2w_m
    pcd.points = o3d.utility.Vector3dVector(pts_world)

    # Paint with bright gold/yellow to contrast against the red-brown liver
    pcd.paint_uniform_color([1.0, 0.85, 0.10])
    return pcd


# 3. Setup Open3D Window & Initial Geometries
cam, K, T_w2c, t0 = get_camera_data(0)
pcd = get_point_cloud(0)

vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window(window_name="Liver + Visible Surface PCD (D / Right to Step)", width=960, height=540)
vis.add_geometry(mesh)
vis.add_geometry(cam)
vis.add_geometry(pcd)

# Set point rendering size for crisp visibility on top of mesh
vis.get_render_option().point_size = 3.0


def snap_view_to_camera(K_mat, T_mat):
    """
    Snaps the viewport directly through the camera lens looking at the liver.
    """
    ctr = vis.get_view_control()
    param = o3d.camera.PinholeCameraParameters()
    param.intrinsic = o3d.camera.PinholeCameraIntrinsic(960, 540, K_mat[0, 0], K_mat[1, 1], K_mat[0, 2], K_mat[1, 2])
    param.extrinsic = T_mat
    ctr.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)


def step_frame(step: int):
    """
    Steps the frame forward or backward:
      - Updates both the camera frustum and the partial surface point cloud in-place.
      - Automatically snaps the screen view through the camera lens.
      - Prints the current frame index, number of visible points, and 3D camera position.
    """
    global current_frame
    current_frame = (current_frame + step) % total_frames

    # 1. Update camera wireframe
    new_cam, K_curr, T_w2c_curr, t_curr = get_camera_data(current_frame)
    cam.points = new_cam.points
    vis.update_geometry(cam)

    # 2. Update visible surface point cloud
    new_pcd = get_point_cloud(current_frame)
    pcd.points = new_pcd.points
    pcd.colors = new_pcd.colors
    vis.update_geometry(pcd)

    # 3. Snap view into camera lens
    snap_view_to_camera(K_curr, T_w2c_curr)

    print(
        f"\r--> Frame [{current_frame:05d} / {total_frames - 1:05d}] | "
        f"Visible Points: {len(pcd.points):,} | "
        f"Cam Pos: [{t_curr[0]:+.2f}, {t_curr[1]:+.2f}, {t_curr[2]:+.2f}] m",
        end="",
        flush=True,
    )


# 4. Register Keyboard Controls
vis.register_key_callback(262, lambda v: step_frame(+1))      # Right arrow: Next frame
vis.register_key_callback(263, lambda v: step_frame(-1))      # Left arrow: Prev frame
vis.register_key_callback(ord('D'), lambda v: step_frame(+1)) # D: Next frame
vis.register_key_callback(ord('A'), lambda v: step_frame(-1)) # A: Prev frame
vis.register_key_callback(32, lambda v: step_frame(+10))      # Spacebar: +10 frames
vis.register_key_callback(ord('V'), lambda v: snap_view_to_camera(get_camera_data(current_frame)[1], get_camera_data(current_frame)[2])) # V: Re-snap

print("=" * 70)
print(" Interactive Visible Surface Point Cloud Inspector")
print("=" * 70)
print(" Controls:")
print("   * RIGHT ARROW or [D] : Next frame (+1) & snap through lens")
print("   * LEFT ARROW  or [A] : Prev frame (-1) & snap through lens")
print("   * SPACEBAR           : Jump +10 frames")
print("   * Key [V]            : Re-snap view through lens (if you moved)")
print("   * Left-Click + Drag  : Orbit around to inspect from the outside")
print("   * Scroll Wheel       : Zoom in / out")
print("=" * 70)

# Snap to camera view on launch
vis.poll_events()
vis.update_renderer()
snap_view_to_camera(K, T_w2c)

vis.run()
vis.destroy_window()
