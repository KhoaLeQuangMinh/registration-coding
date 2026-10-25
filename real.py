import os
import open3d as o3d
import cv2
import numpy as np

# ==============================================================================
# Configuration: File Paths
# ==============================================================================
OBJ_PATH   = "dataset/liver-registration/01/model/reconstructed_mesh_world_m.obj"
PCD_DIR    = "dataset/liver-registration/01/real/liverPcds"
LABEL_DIR  = "dataset/liver-registration/01/real/labels"

# Total 22 real clinical frames (frame_1 to frame_22)
TOTAL_FRAMES = 22
current_idx = 1  # 1-based indexing to match dataset naming

# In-memory cache for smooth frame navigation
pcd_cache = {}
img_cache = {}

# Window identifiers
IMG_WINDOW_NAME = "Real Laparoscopic Surgery View (img.png)"


def load_real_pcd(frame_num: int) -> o3d.geometry.PointCloud:
    """
    Loads the real intraoperative 3D point cloud for the given frame number:
      - Reads the 1080p raster point cloud (frame_{frame_num}_json.ply).
      - Filters out invalid/background points where Z <= 0.
      - Keeps the ~1 million valid liver surface points with their authentic
        endoscopic RGB colors.
    """
    if frame_num in pcd_cache:
        return pcd_cache[frame_num]

    ply_path = os.path.join(PCD_DIR, f"frame_{frame_num}_json.ply")
    pcd_raw = o3d.io.read_point_cloud(ply_path)

    pts = np.asarray(pcd_raw.points)
    colors = np.asarray(pcd_raw.colors)

    # Filter out invalid background points (Z <= 0)
    valid_mask = pts[:, 2] > 0
    valid_pts = pts[valid_mask]
    valid_colors = colors[valid_mask]

    pcd_clean = o3d.geometry.PointCloud()
    pcd_clean.points = o3d.utility.Vector3dVector(valid_pts)
    pcd_clean.colors = o3d.utility.Vector3dVector(valid_colors)

    pcd_cache[frame_num] = pcd_clean
    return pcd_clean


def show_real_image(frame_num: int):
    """
    Displays the real laparoscopic endoscopic image for the current frame.
    Adds a text banner showing the frame index and resolution.
    """
    if frame_num not in img_cache:
        img_path = os.path.join(LABEL_DIR, f"frame_{frame_num}_json", "img.png")
        if not os.path.exists(img_path):
            return
        img_cache[frame_num] = cv2.imread(img_path)

    img = img_cache[frame_num]
    annotated = img.copy()

    # Overlay information banner
    header_text = f"Surgery Frame: {frame_num:02d} / {TOTAL_FRAMES:02d}  (1920x1080)"
    cv2.putText(
        annotated,
        header_text,
        (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 255),
        3,
        cv2.LINE_AA,
    )
    cv2.imshow(IMG_WINDOW_NAME, annotated)
    cv2.waitKey(1)


def main():
    global current_idx

    # 1. Load Preoperative 3D CT Liver Mesh (scaled from meters to mm)
    mesh = o3d.io.read_triangle_mesh(OBJ_PATH)
    mesh.compute_vertex_normals()
    mesh.scale(1000.0, center=[0, 0, 0])
    mesh.paint_uniform_color([0.65, 0.28, 0.22])  # Anatomical reddish-brown

    # Position CT mesh to the left (X = -250 mm) for side-by-side comparison
    mesh.translate([-250.0, 0.0, 0.0])

    # 2. Setup OpenCV 2D Window
    cv2.namedWindow(IMG_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(IMG_WINDOW_NAME, 640, 360)
    cv2.moveWindow(IMG_WINDOW_NAME, 1000, 60)

    # 3. Setup Open3D 3D Window & Geometries
    initial_pcd = load_real_pcd(current_idx)
    live_pcd = o3d.geometry.PointCloud()
    live_pcd.points = initial_pcd.points
    live_pcd.colors = initial_pcd.colors

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(
        window_name="3D Real Data: [Left: CT Mesh (mm)] | [Right: Intraoperative Surface (Real RGB)]",
        width=960,
        height=540,
        left=20,
        top=60,
    )

    vis.add_geometry(mesh)
    vis.add_geometry(live_pcd)

    opt = vis.get_render_option()
    opt.point_size = 2.0
    opt.background_color = np.array([0.15, 0.15, 0.15])

    def step_frame(step: int):
        global current_idx
        current_idx = ((current_idx - 1 + step) % TOTAL_FRAMES) + 1

        new_data = load_real_pcd(current_idx)
        live_pcd.points = new_data.points
        live_pcd.colors = new_data.colors
        vis.update_geometry(live_pcd)

        show_real_image(current_idx)

        print(
            f"\r--> Frame [{current_idx:02d} / {TOTAL_FRAMES:02d}] | "
            f"Valid Points: {len(live_pcd.points):,} | "
            f"File: frame_{current_idx}_json.ply",
            end="",
            flush=True,
        )

    # 4. Register Keyboard Controls
    vis.register_key_callback(262, lambda v: step_frame(+1))      # Right arrow
    vis.register_key_callback(263, lambda v: step_frame(-1))      # Left arrow
    vis.register_key_callback(ord('D'), lambda v: step_frame(+1)) # D: Next
    vis.register_key_callback(ord('A'), lambda v: step_frame(-1)) # A: Prev

    print("=" * 80)
    print(" Interactive Real Surgical Data Inspector (Preoperative CT vs. Intraoperative)")
    print("=" * 80)
    print(" 3D Window (Left side):")
    print("   - LEFT  : Preoperative 3D CT Model (.obj, scaled to millimeters)")
    print("   - RIGHT : Reconstructed 3D Intraoperative Surface (.ply, real surgery colors)")
    print(" 2D Window (Right side):")
    print("   - Real Laparoscopic Camera View (img.png, 1080p Full-HD)")
    print("-" * 80)
    print(" Controls:")
    print("   * RIGHT ARROW or [D] : Next frame (+1) -> Updates both 3D PCD and 2D image")
    print("   * LEFT ARROW  or [A] : Prev frame (-1) -> Updates both 3D PCD and 2D image")
    print("   * Left-Click + Drag  : Free 3D orbit around both models")
    print("   * Right-Click + Drag : Pan 3D view")
    print("   * Scroll Wheel       : Zoom in/out")
    print("   * Key [Q] or [ESC]   : Close and exit")
    print("=" * 80)

    # Initial display
    show_real_image(current_idx)
    vis.poll_events()
    vis.update_renderer()

    vis.run()

    vis.destroy_window()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
