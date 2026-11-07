"""
The PCA scale recipe of Self-P2IR (Zhou et al., TMI 2025, Sec. III-C, Eq. 1-2), as written in the paper.

Monocular depth has no true size, so the camera cloud T is resized by
    sigma = size(S) / size(T)
where S is the CT model and size() projects the points onto their own principal axis (the
eigenvector of the covariance with the largest eigenvalue). The paper does not say how the
projections become one number; we use their range (max - min), fixed before any run.

Applied to a synthetic cloud (true size), the recipe returns the factor by which it would
resize that cloud: 1.0 means the recipe is exact for that view.

Validity check (python benchmark/scale_recipe.py): if the released real PLYs were made with
this recipe, size(CT) / size(PLY) would be exactly 1 on every frame.
"""
import os

import cv2
import numpy as np
import open3d as o3d

DATA_DIR = "dataset/liver-registration"


def principal_size(X: np.ndarray, how: str = "range") -> float:
    Xc = X - X.mean(axis=0)
    _, V = np.linalg.eigh(np.cov(Xc.T))
    proj = Xc @ V[:, -1]
    return float(np.ptp(proj) if how == "range" else proj.std())


def recipe_factor(ct_vertices: np.ndarray, cloud: np.ndarray, how: str = "range") -> float:
    """Factor the recipe applies to `cloud` (same units as `ct_vertices`)."""
    return principal_size(ct_vertices, how) / principal_size(cloud, how)


def main():
    for how in ("range", "std"):
        ratios = []
        for pid in sorted(os.listdir(DATA_DIR)):
            base = os.path.join(DATA_DIR, pid)
            if not os.path.isdir(os.path.join(base, "real", "liverPcds")):
                continue
            mesh = o3d.io.read_triangle_mesh(os.path.join(base, "model", "reconstructed_mesh_world_m.obj"))
            S = np.asarray(mesh.vertices) * 1000.0
            for fn in sorted(os.listdir(os.path.join(base, "real", "liverPcds"))):
                frame = fn[:-4]
                pts = np.asarray(o3d.io.read_point_cloud(os.path.join(base, "real", "liverPcds", fn)).points)
                mask = cv2.imread(os.path.join(base, "real", "labels", frame, "label.png")).sum(-1).reshape(-1) > 0
                ratios.append(recipe_factor(S, pts[mask], how))
        r = np.array(ratios)
        print(f"size(CT) / size(released PLY), {how:5s}: median {np.median(r):.3f}, "
              f"10-90 % {np.percentile(r, 10):.3f}-{np.percentile(r, 90):.3f}, "
              f"min {r.min():.3f}, max {r.max():.3f}  ({len(r)} frames; 1.000 everywhere = recipe reproduced)")


if __name__ == "__main__":
    main()
