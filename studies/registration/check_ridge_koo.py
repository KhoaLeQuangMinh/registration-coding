"""
One-time check of the frozen ridge rule (ct_landmarks.py) against Koo et al.'s expert ridge on the
4 LLR-LUS livers. Run once; the rule is not tuned on the result.

Koo's ridge: every Ridge contour of every image (contours.xml), as vertices of Koo's own liver mesh,
pooled per patient. Koo's mesh is in another frame, so it is first aligned to the dataset liver by
rigid ICP (same liver: no scaling); the check stops if the aligned mesh is more than 2 mm (median)
from the dataset liver. Livers with a mean edge above 2 mm are split at edge midpoints until below
(shape unchanged), as the rule was frozen on ~1.4 mm meshes.

Reported per patient (decided before running):
  primary  distance from each of Koo's ridge vertices to the nearest point of our ridge:
           median, 90th percentile, % within 5 mm and 10 mm
  context  % of our ridge within 10 mm of Koo's ridge (Koo marked only what each image shows,
           so a low value is not an error)

Usage: python registration/check_ridge_koo.py [--save-dir <folder for both ridges per patient, .npz>]
"""
import argparse
import glob
import itertools
import xml.etree.ElementTree as ET

import numpy as np
import open3d as o3d

from ct_landmarks import ridge

DATASET = "dataset/lus-evaluation/data/Dataset"
KOO = "dataset/lus-evaluation/koo/contours_and_models"


def koo_ridge(p):
    """Indices of Koo's ridge vertices on Koo's mesh, all images pooled."""
    idx = set()
    for f in glob.glob(f"{KOO}/Patient{p}/Annotations/*/koo/contours.xml"):
        for c in ET.parse(f).getroot().iter("contour"):
            if c.find("contourType").text == "Ridge" and c.find("modelPoints") is not None:
                idx |= {int(v) for v in c.find("modelPoints/vertices").text.split(",")}
    return np.array(sorted(idx))


def align(src_mesh, dst_mesh):
    """Rigid transform of src onto dst (same liver): ICP from the 24 principal-axis starts, best fit kept."""
    a = src_mesh.sample_points_uniformly(20000)
    b = dst_mesh.sample_points_uniformly(20000)
    A, B = np.asarray(a.points), np.asarray(b.points)
    axes = lambda X: np.linalg.svd(X - X.mean(0), full_matrices=False)[2]  # noqa: E731
    As, Ad = axes(A), axes(B)
    best = (np.inf, None)
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([1, -1], repeat=3):
            S = np.zeros((3, 3))
            S[range(3), perm] = signs
            R = Ad.T @ S @ As
            if np.linalg.det(R) < 0:
                continue
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = R, B.mean(0) - R @ A.mean(0)
            res = o3d.pipelines.registration.registration_icp(
                a, b, 20.0, T, o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200))
            best = min(best, (res.inlier_rmse if res.fitness > 0.9 else np.inf, res.transformation), key=lambda r: r[0])
    return best[1]


def mean_edge(mesh):
    T = np.asarray(mesh.vertices)[np.asarray(mesh.triangles)]
    return np.linalg.norm(T[:, 0] - T[:, 1], axis=1).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-dir", help="not in the repo: the ridges are derived from the LLR-LUS data")
    args = ap.parse_args()
    o3d.utility.random.seed(0)
    for p in range(1, 5):
        liver = o3d.io.read_triangle_mesh(f"{DATASET}/Patient{p}/Preoperative Model/Liver.ply")
        while mean_edge(liver) > 2.0:
            liver = liver.subdivide_midpoint(1)
        liver.compute_vertex_normals()
        koo = o3d.io.read_triangle_mesh(f"{KOO}/Patient{p}/Liver.obj")
        koo.transform(align(koo, liver))

        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(liver))
        fit = np.median(scene.compute_distance(o3d.core.Tensor(np.asarray(koo.vertices, np.float32))).numpy())
        if fit > 2.0:
            raise SystemExit(f"Patient {p}: Koo's mesh does not align to the dataset liver ({fit:.1f} mm median); stopping")

        ours = np.asarray(liver.vertices)[np.concatenate(ridge(liver))]
        theirs = np.asarray(koo.vertices)[koo_ridge(p)]
        d = np.linalg.norm(theirs[:, None] - ours[None], axis=2)
        k2o, o2k = d.min(1), d.min(0)
        print(f"Patient {p}: Koo mesh fit {fit:.2f} mm; Koo ridge {len(theirs)} vertices -> ours: median {np.median(k2o):.1f} mm, "
              f"90th pct {np.percentile(k2o, 90):.1f} mm, within 5 mm {100 * np.mean(k2o <= 5):.0f} %, "
              f"within 10 mm {100 * np.mean(k2o <= 10):.0f} % | ours within 10 mm of Koo's: {100 * np.mean(o2k <= 10):.0f} %")
        if args.save_dir:
            np.savez(f"{args.save_dir}/ridge_koo_P{p}.npz", ours=ours, theirs=theirs)


if __name__ == "__main__":
    main()
