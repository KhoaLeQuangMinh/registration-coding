import open3d as o3d
import yaml
import numpy as np

# ==============================================================================
# Configuration: File Paths
# ==============================================================================
OBJ_PATH  = "dataset/liver-registration/01/model/reconstructed_mesh_world_m.obj"
CAMK_PATH = "dataset/liver-registration/01/syn/CAM_K.yml"
CAMP_PATH = "dataset/liver-registration/01/syn/camPose.yml"

# 1. Load the 3D CT liver mesh
mesh = o3d.io.read_triangle_mesh(OBJ_PATH)
mesh.compute_vertex_normals()

# 2. Load camera intrinsics and poses from YAML files
with open(CAMK_PATH) as f:
    cam_k = yaml.safe_load(f)

with open(CAMP_PATH) as f:
    poses = yaml.safe_load(f)

total_frames = len(poses)
current_frame = 0


def get_camera_data(frame_idx: int):
    """
    Extracts camera parameters for a given frame index:
      - Reads 3x3 intrinsic matrix K (lens zoom and optical center).
      - Reads 3x3 rotation R and 3D translation t (camera position in space).
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
    t = np.array(p["cam_t_c2w"]).reshape(3) / 1000.0  # convert mm to meters to match CT mesh

    # Build 4x4 World-to-Camera matrix
    T_w2c = np.eye(4)
    T_w2c[:3, :3] = R_cv.T
    T_w2c[:3, 3] = -R_cv.T @ t

    cam_vis = o3d.geometry.LineSet.create_camera_visualization(960, 540, K, T_w2c, scale=0.15)
    return cam_vis, K, T_w2c, t


# 3. Setup Open3D Window & Initial Geometries
cam, K, T_w2c, t0 = get_camera_data(0)

vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window(window_name="Camera View (D / Right to Step & Snap)", width=960, height=540)
vis.add_geometry(mesh)
vis.add_geometry(cam)


def snap_view_to_camera(K_mat, T_mat):
    """
    Snaps the user's screen view directly through the laparoscope camera lens
    using the camera's intrinsic lens properties (K) and 6-DoF pose (T_w2c).
    """
    ctr = vis.get_view_control()
    param = o3d.camera.PinholeCameraParameters()
    param.intrinsic = o3d.camera.PinholeCameraIntrinsic(960, 540, K_mat[0, 0], K_mat[1, 1], K_mat[0, 2], K_mat[1, 2])
    param.extrinsic = T_mat
    ctr.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)


def step_frame(step: int):
    """
    Steps the frame forward or backward:
      - Updates the 3D camera pyramid geometry in-place (no GPU lag or tearing).
      - Automatically snaps the screen view to the new camera perspective.
      - Prints the current frame number and camera position in the console.
    """
    global current_frame
    current_frame = (current_frame + step) % total_frames
    new_cam, K_curr, T_w2c_curr, t_curr = get_camera_data(current_frame)
    cam.points = new_cam.points
    vis.update_geometry(cam)
    snap_view_to_camera(K_curr, T_w2c_curr)
    print(f"\r--> Snapped to Frame [{current_frame:05d} / {total_frames - 1:05d}] | Position: [{t_curr[0]:+.2f}, {t_curr[1]:+.2f}, {t_curr[2]:+.2f}] m", end="", flush=True)


# 4. Register Keyboard Controls
vis.register_key_callback(262, lambda v: step_frame(+1))      # Right arrow
vis.register_key_callback(263, lambda v: step_frame(-1))      # Left arrow
vis.register_key_callback(ord('D'), lambda v: step_frame(+1)) # D key
vis.register_key_callback(ord('A'), lambda v: step_frame(-1)) # A key
vis.register_key_callback(ord('V'), lambda v: snap_view_to_camera(get_camera_data(current_frame)[1], get_camera_data(current_frame)[2])) # V key

print("=" * 65)
print(" Interactive First-Person Camera Viewpoint Inspector")
print("=" * 65)
print(" Controls:")
print("   * RIGHT ARROW or [D] : Step to NEXT frame & SNAP view through lens")
print("   * LEFT ARROW  or [A] : Step to PREV frame & SNAP view through lens")
print("   * Key [V]            : Re-snap view through lens (if you moved)")
print("   * Left-Click + Drag  : Orbit around to look from the outside")
print("=" * 65)

# Snap to camera on initial launch
vis.poll_events()
vis.update_renderer()
snap_view_to_camera(K, T_w2c)

vis.run()
vis.destroy_window()
