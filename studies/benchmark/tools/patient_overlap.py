"""
Is any LLR-LUS patient also a P2I-LReg patient? Compare the CT liver shapes.

Each pair of livers is aligned by rotation + shift (+ scale) from 24 starts (every rotation that
maps one liver's principal axes onto the other's) followed by ICP; the score is the mean symmetric
surface distance (mm) of the best start. The P2I-LReg liver is also tried mirrored, in case the two
datasets use opposite axis conventions. Two segmentations of one liver should be close and need
scale ~1 (both are in mm); different people's livers should not. Pass --reference to also score
each LUS liver against the dataset's second segmentation of the same liver
(OtherLiverSurfaceModel.obj), as a same-liver yardstick.

Usage: python benchmark/tools/patient_overlap.py [--p2ilreg-root dataset/liver-registration]
           [--reference <folder with PatientN/OtherLiverSurfaceModel.obj>]
"""
import argparse
import glob
import itertools
import os

import numpy as np
import open3d as o3d

LUS = "dataset/lus-evaluation/data/Dataset"


def cloud(path, scale=1.0, n=5000):
    mesh = o3d.io.read_triangle_mesh(path)
    mesh.scale(scale, center=[0, 0, 0])
    return np.asarray(mesh.sample_points_uniformly(n).points)


def starts(src, dst):
    """The 24 rotations that map src's principal axes onto dst's (any order, any sign)."""
    axes = lambda X: np.linalg.svd(X - X.mean(0), full_matrices=False)[2]  # noqa: E731
    As, Ad = axes(src), axes(dst)
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([1, -1], repeat=3):
            S = np.zeros((3, 3))
            S[range(3), perm] = signs
            R = Ad.T @ S @ As
            if np.linalg.det(R) > 0:
                yield R


def fit(src, dst, scaling):
    """Best mean symmetric distance (mm) and scale of src aligned onto dst by a rotation (no mirroring)."""
    a, b = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(src)), o3d.geometry.PointCloud(o3d.utility.Vector3dVector(dst))
    best = (np.inf, 1.0)
    for R in starts(src, dst):
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = dst.mean(0) - R @ src.mean(0)
        T = o3d.pipelines.registration.registration_icp(
            a, b, 30.0, T, o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=scaling),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)).transformation
        if np.linalg.det(T[:3, :3]) <= 0:
            continue
        moved = o3d.geometry.PointCloud(a).transform(T)
        d = (np.mean(moved.compute_point_cloud_distance(b)) + np.mean(b.compute_point_cloud_distance(moved))) / 2
        best = min(best, (d, np.cbrt(np.linalg.det(T[:3, :3]))))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p2ilreg-root", default="dataset/liver-registration")
    ap.add_argument("--reference", help="folder with PatientN/OtherLiverSurfaceModel.obj")
    args = ap.parse_args()
    o3d.utility.random.seed(0)

    lus = {f"LUS P{p}": cloud(f"{LUS}/Patient{p}/Preoperative Model/Liver.stl") for p in range(1, 5)}
    reg = {f"P2I-LReg {os.path.basename(os.path.dirname(os.path.dirname(f)))}": cloud(f, 1000.0)
           for f in sorted(glob.glob(f"{args.p2ilreg_root}/*/model/reconstructed_mesh_world_m.obj"))}
    if args.reference:
        for p in range(1, 5):
            d, s = fit(cloud(f"{args.reference}/Patient{p}/OtherLiverSurfaceModel.obj"), lus[f"LUS P{p}"], True)
            print(f"same liver, 2nd segmentation, LUS P{p}: {d:.1f} mm (scale {s:.3f})")
    for (ln, L), (rn, R) in itertools.product(lus.items(), reg.items()):
        for mirror in (False, True):
            X = R * [-1, 1, 1] if mirror else R
            d_rigid, _ = fit(X, L, False)
            d_sim, s = fit(X, L, True)
            print(f"{ln} vs {rn}{' (mirrored)' if mirror else ''}: rigid {d_rigid:.1f} mm, "
                  f"with scale {d_sim:.1f} mm (scale {s:.3f})")


if __name__ == "__main__":
    main()
