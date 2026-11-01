import open3d as o3d
import numpy as np

# ==============================================================================
# 1. Load Preoperative 3D CT Mesh (Scaled from Meters to Millimeters)
# ==============================================================================
OBJ_PATH = "dataset/liver-registration/01/model/reconstructed_mesh_world_m.obj"
mesh = o3d.io.read_triangle_mesh(OBJ_PATH)
mesh.scale(1000.0, center=[0, 0, 0])

# ==============================================================================
# 2. Dynamic Solution A + B: Calculate Triangles from Patient's Exact Surface Area
# ==============================================================================
TARGET_SPACING = 2.0  # Desired physical spacing between points in millimeters

# Measure the patient's true CT surface area dynamically (no hardcoding)
patient_area = mesh.get_surface_area()

# Calculate exact number of triangles needed to guarantee ~2.0 mm spacing:
# Target Triangles = Surface Area / (spacing)^2
target_triangles = int(patient_area / (TARGET_SPACING ** 2))

decimated_mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_triangles)

anchor_pcd = o3d.geometry.PointCloud()
anchor_pcd.points = decimated_mesh.vertices
anchor_pcd.paint_uniform_color([0.2, 0.8, 0.3])  # Emerald green

# Verification measuring tape: Measure average physical distance between neighbors
dists = anchor_pcd.compute_nearest_neighbor_distance()
mean_spacing = np.mean(dists)

print("=" * 65)
print(f"Master CT surface area:  {patient_area:,.1f} mm^2")
print(f"Dynamic target triangles: {target_triangles:,} (Calculated for {TARGET_SPACING:.1f} mm)")
print(f"Curvature anchor points: {len(anchor_pcd.points):,}")
print(f"Verified point spacing:  {mean_spacing:.2f} mm")
print("=" * 65)

# ==============================================================================
# 3. Plot Result: Green Anchor Points over the Transparent Gray Mesh Surface
# ==============================================================================
decimated_mesh.compute_vertex_normals()
decimated_mesh.paint_uniform_color([0.7, 0.7, 0.7])

o3d.visualization.draw_geometries(
    [decimated_mesh, anchor_pcd],
    window_name="Preprocessed CT: ~12k Anchor Points (Equal 2.0 mm Density)",
    width=960,
    height=540,
)
