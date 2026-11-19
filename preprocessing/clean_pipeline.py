"""
Unified Preprocessing Pipeline for Liver Point Clouds and Meshes
================================================================
Implements the exact first-principles preprocessing solutions verified for P2I-LReg:

1. CT Master Mesh:
   - Scale from meters to millimeters (x1000)
   - Dynamic Quadric Decimation: Target Triangles = Surface Area / (2.0 mm)^2
   - Preserves master 69k mesh in memory while producing ~12k vertices with 1.98mm spacing.

2. Synthetic Camera PLY:
   - 2.9 mm spatial voxel downsampling
   - Produces ~4.5k clean surface points with 1.97mm spacing.

3. Real Laparoscopic PLY:
   - Reshape 1D array back to 1080x1920 camera raster grid
   - Rebuild points in the true camera frame with the calibrated principal point
     (the dataset PLYs were back-projected with the image centre (960, 540))
   - 2D striding by 4 (pts[::4, ::4]) to collapse AI upscaled duplicate blocks
   - Filter valid liver points using the 2D label mask (label.png > 0)
   - 3.0 mm spatial voxel downsampling
   - Produces ~3.0k clean points with 2.02mm spacing.
"""

import os

import cv2
import numpy as np
import open3d as o3d
import torch
import yaml


def preprocess_ct_mesh(obj_path: str, target_spacing: float = 2.0) -> np.ndarray:
    """
    Preprocesses preoperative CT mesh to standard 2.0mm physical spacing.
    Returns: (M, 3) float32 coordinates in millimeters.
    """
    mesh = o3d.io.read_triangle_mesh(obj_path)
    mesh.scale(1000.0, center=[0, 0, 0])  # Convert meters to millimeters

    # Dynamic target triangles based on patient surface area
    area = mesh.get_surface_area()
    target_triangles = int(area / (target_spacing ** 2))
    mesh_dec = mesh.simplify_quadric_decimation(target_number_of_triangles=target_triangles)

    pts = np.asarray(mesh_dec.vertices, dtype=np.float32)
    return pts


def preprocess_syn_point_cloud(ply_path: str, voxel_size: float = 2.9) -> np.ndarray:
    """
    Preprocesses synthetic camera point cloud to standard 2.0mm physical spacing.
    Returns: (N, 3) float32 coordinates in millimeters.
    """
    pcd = o3d.io.read_point_cloud(ply_path)
    down_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    pts = np.asarray(down_pcd.points, dtype=np.float32)
    return pts


def rebuild_real_camera_points(pts: np.ndarray, mask: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Re-back-projects a real PLY raster with the calibrated intrinsics K.

    The dataset PLYs were built as X = (u - 960) * d / f, Y = (v - 540) * d / f, Z = d
    (image centre instead of the calibrated cx, cy) and then shifted. Background pixels
    have d = 0, so they all land on the camera centre; subtracting that point's Z
    recovers the per-pixel depth d exactly.

    Args:
        pts:  (H, W, 3) PLY coordinates in mm
        mask: (H, W) bool, True = liver
        K:    (3, 3) calibrated intrinsics from real/CAM_K.yml
    Returns: (H, W, 3) points in the OpenCV camera frame (camera at origin, +Z forward), mm.
    """
    H, W = mask.shape
    origin = pts[~mask][0]  # every background pixel sits on this one point
    depth = pts[..., 2] - origin[2]
    v, u = np.mgrid[0:H, 0:W]
    X = (u - K[0, 2]) * depth / K[0, 0]
    Y = (v - K[1, 2]) * depth / K[1, 1]
    return np.stack([X, Y, depth], axis=-1)


def preprocess_real_point_cloud(ply_path: str, voxel_size: float = 3.0, label_path: str = None,
                                camk_path: str = None) -> np.ndarray:
    """
    Preprocesses real surgical camera point cloud to standard 2.0mm physical spacing.
    Collapses 3x4 AI depth upsampling staircase blocks via 2D grid striding.
    label_path defaults to real/labels/<frame>/label.png next to real/liverPcds/<frame>.ply,
    camk_path defaults to real/CAM_K.yml.
    Returns: (N, 3) float32 coordinates in millimeters, in the OpenCV camera frame.
    """
    frame = os.path.splitext(os.path.basename(ply_path))[0]
    real_dir = os.path.dirname(os.path.dirname(ply_path))
    if label_path is None:
        label_path = os.path.join(real_dir, "labels", frame, "label.png")
    if camk_path is None:
        camk_path = os.path.join(real_dir, "CAM_K.yml")

    pcd = o3d.io.read_point_cloud(ply_path)

    # 1. Reshape to 1080x1920 raster camera grid
    pts = np.asarray(pcd.points).reshape(1080, 1920, 3)

    # 2. Liver mask. Background pixels all sit on one point (the camera centre, Z ~ -115),
    # and liver points close to the camera can have Z < 0, so a Z > 0 test is not safe.
    mask = cv2.imread(label_path).sum(axis=-1) > 0  # (1080, 1920), True = liver

    # 3. Rebuild with the calibrated principal point
    with open(camk_path) as f:
        K = np.array(yaml.safe_load(f)["00000"][0]["cam_K"]).reshape(3, 3)
    pts = rebuild_real_camera_points(pts, mask, K)

    # 4. 2D grid stride by 4 to collapse staircase duplicate blocks
    sub_pts = pts[::4, ::4].reshape(-1, 3)
    valid = mask[::4, ::4].reshape(-1)
    clean_pcd = o3d.geometry.PointCloud()
    clean_pcd.points = o3d.utility.Vector3dVector(sub_pts[valid])

    # 5. 3.0mm voxel downsample to lock ~2.0mm physical neighbor distance
    down_pcd = clean_pcd.voxel_down_sample(voxel_size=voxel_size)
    clean_pts = np.asarray(down_pcd.points, dtype=np.float32)

    return clean_pts
