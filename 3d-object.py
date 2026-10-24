import open3d as o3d
OBJ_PATH = "dataset/liver-registration/01/model/reconstructed_mesh_world_m.obj"
mesh = o3d.io.read_triangle_mesh(OBJ_PATH)
mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh])
