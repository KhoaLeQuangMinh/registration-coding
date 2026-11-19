"""
Show what module 3 does on one synthetic view: the camera cloud before and after alignment.

One Open3D window per starting size (close it to open the next):
  grey   CT liver surface
  red    camera cloud at the starting guess, red ball = where the guess puts the camera
  blue   camera cloud after module 3,        blue ball = where module 3 puts the camera
  green  camera cloud at the true pose,      green ball = the true camera position
Blue on top of green means module 3 found the answer. Drag to rotate, scroll to zoom.

Usage: python registration/show_module3.py [--patient 01] [--view 1124] [--sizes 0.7 1.0 1.4]
                                           [--rotation 0] [--shift 0]
"""
import argparse

import numpy as np
import open3d as o3d

from capture_range import LIVER, load_patient, load_view, rotation, sample_tumours, true_w2c, unit
from sim3_icp import register

GREY, RED, BLUE, GREEN = [0.75, 0.75, 0.75], [0.9, 0.2, 0.2], [0.2, 0.4, 0.9], [0.2, 0.75, 0.3]


def points(X: np.ndarray, color) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(X))
    pcd.paint_uniform_color(color)
    return pcd


def ball(centre: np.ndarray, color, radius: float = 5.0) -> o3d.geometry.TriangleMesh:
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.translate(centre)
    sphere.paint_uniform_color(color)
    sphere.compute_vertex_normals()
    return sphere


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", default="01")
    ap.add_argument("--view", type=int, default=1124, help="a training view (see train_syn.txt)")
    ap.add_argument("--sizes", type=float, nargs="+", default=[0.7, 1.0, 1.4], help="starting size factors")
    ap.add_argument("--rotation", type=float, default=0.0, help="starting rotation error (deg)")
    ap.add_argument("--shift", type=float, default=0.0, help="starting position error (mm)")
    args = ap.parse_args()

    P = load_patient(args.patient)
    probes, _ = sample_tumours(P, np.random.default_rng([0, int(args.patient)]), 200)
    label, cloud = load_view(P, args.view)
    T = true_w2c(P["poses"][f"{args.view:05d}"][0])
    rays = P["pix_dirs"][(label == LIVER).reshape(-1)]
    rays = rays[::max(1, len(rays) // len(cloud))]

    Q_true, d_true = T[:3, :3].T, -T[:3, :3].T @ T[:3, 3]  # camera -> CT; the camera sits at d
    m = Q_true @ cloud.mean(axis=0) + d_true
    cam_true = (probes - d_true) @ Q_true
    ct = P["target"].paint_uniform_color(GREY)

    rng = np.random.default_rng(0)
    dR, shift = rotation(unit(rng), args.rotation), args.shift * unit(rng)
    for k in args.sizes:
        start = (k, dR @ Q_true, dR @ (d_true - m) + m + shift)
        final = register(cloud, rays, P["scene"], *start)
        tre = np.linalg.norm((probes - final[2]) @ final[1] - cam_true, axis=1).mean()
        print(f"start size x{k}: final size x{final[0]:.3f}, probe tumour error {tre:.1f} mm")

        geoms = [ct]
        for (c, Q, d), color in [((1.0, Q_true, d_true), GREEN), (start, RED), (final, BLUE)]:
            geoms += [points(cloud @ (c * Q).T + d, color), ball(d, color)]
        o3d.visualization.draw_geometries(
            geoms, window_name=f"start size x{k} -> final x{final[0]:.2f}, tumour error {tre:.1f} mm")


if __name__ == "__main__":
    main()
