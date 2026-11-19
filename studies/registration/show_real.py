"""
Run modules 2 + 3 on one REAL frame and show the result. There is no true pose for real frames,
so this can show an answer that is clearly wrong, but it cannot prove one is right.

  1 an Open3D window: grey CT liver, the real camera cloud placed by modules 2 + 3 in the colours
    of the photo, blue ball = where they put the camera
  2 an image, registration/out/real_<patient>_<frame>.png: the photo (left) and the photo with the
    CT liver mesh seen from the found pose drawn as a see-through white wireframe (right)

The real cloud is the released PLY rebuilt with the calibrated principal point
(preprocessing/clean_pipeline.py); its size comes from the dataset's depth scale, and module 3
re-estimates it. The real liver is deformed relative to the CT, so the fit can only be approximate.

Module 3 also pulls the CT's outline onto the liver outline in the image (sim3_icp.outline_pairs);
--no-outline turns that off, to compare.

Usage: python registration/show_real.py [--patient 01] [--frame 1] [--no-outline] [--no-window]
"""
import argparse
import os
import sys

import cv2
import numpy as np
import open3d as o3d
import yaml

from capture_range import HERE, load_patient
from multistart import register_from_scratch
from show_module3 import BLUE, GREY, ball, points

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "preprocessing"))
from clean_pipeline import preprocess_real_point_cloud  # noqa: E402


def pixel_dirs(K: np.ndarray, shape, step: int = 1) -> np.ndarray:
    """Camera-frame ray direction through the centre of every `step`-th pixel, row by row."""
    v, u = np.mgrid[0:shape[0]:step, 0:shape[1]:step] + 0.5
    return np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], axis=-1).reshape(-1, 3)


def silhouette(scene, K: np.ndarray, shape, Q: np.ndarray, d: np.ndarray, step: int = 2) -> np.ndarray:
    """Image mask of the CT seen from the camera at pose (Q, d), full size."""
    dirs = pixel_dirs(K, shape, step) @ Q.T
    rays = np.hstack([np.broadcast_to(d, dirs.shape), dirs]).astype(np.float32)
    hit = np.isfinite(scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy())
    hit = hit.reshape(-(-shape[0] // step), -(-shape[1] // step)).astype(np.uint8)
    return cv2.resize(hit, shape[::-1], interpolation=cv2.INTER_NEAREST).astype(bool)


def wireframe(image: np.ndarray, mesh, K: np.ndarray, Q: np.ndarray, d: np.ndarray,
              triangles: int = 20000, alpha: float = 0.5) -> np.ndarray:
    """The photo with every edge of the (simplified) CT mesh drawn in see-through white."""
    simple = mesh.simplify_quadric_decimation(target_number_of_triangles=triangles)
    y = (np.asarray(simple.vertices) - d) @ Q  # camera frame; the scale does not change the projection
    uv = np.stack([K[0, 0] * y[:, 0] / y[:, 2] + K[0, 2], K[1, 1] * y[:, 1] / y[:, 2] + K[1, 2]], axis=1)
    tri = np.asarray(simple.triangles)
    edges = np.unique(np.sort(np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), axis=1), axis=0)
    edges = edges[(y[edges, 2] > 0).all(axis=1)]  # both ends in front of the camera
    lines = np.zeros_like(image)
    cv2.polylines(lines, list(np.round(uv[edges]).astype(np.int32)), False, (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.addWeighted(image, 1.0, lines, alpha, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", default="01")
    ap.add_argument("--frame", type=int, default=1)
    ap.add_argument("--no-outline", action="store_true", help="do not use the liver outline in module 3")
    ap.add_argument("--no-window", action="store_true", help="only save the image")
    args = ap.parse_args()

    P = load_patient(args.patient)
    real = os.path.join(P["root"], "real")
    name = f"frame_{args.frame}_json"
    with open(os.path.join(real, "CAM_K.yml")) as f:
        K = np.array(yaml.safe_load(f)["00000"][0]["cam_K"]).reshape(3, 3)
    image = cv2.imread(os.path.join(real, "labels", name, "img.png"))
    mask = cv2.imread(os.path.join(real, "labels", name, "label.png")).sum(axis=-1) > 0

    cloud = preprocess_real_point_cloud(os.path.join(real, "liverPcds", name + ".ply")).astype(np.float64)
    rays = pixel_dirs(K, mask.shape)[mask.reshape(-1)]
    rays = rays[::max(1, len(rays) // len(cloud))]

    step = 8  # outline grid: every 8th pixel
    outline = None if args.no_outline else (
        pixel_dirs(K, mask.shape, step).reshape(-(-mask.shape[0] // step), -(-mask.shape[1] // step), 3),
        mask[::step, ::step])
    best, finals, errors = register_from_scratch(cloud, rays, P["scene"], P["centroid"], outline=outline)
    c, Q, d = best
    print("10 candidates after module 3 (fit in camera units; picked = smallest fit):")
    for p, e in sorted(zip(finals, errors), key=lambda x: x[1]):
        print(f"  size x{p[0]:.3f}  fit {e:5.2f}{'   <- picked' if p is best else ''}")

    ct_mask = silhouette(P["scene"], K, mask.shape, Q, d)
    dice = 2 * (ct_mask & mask).sum() / (ct_mask.sum() + mask.sum())
    overlay = wireframe(image, P["mesh"], K, Q, d)
    out = os.path.join(HERE, "out", f"real_{args.patient}_{args.frame}{'_no_outline' if args.no_outline else ''}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cv2.imwrite(out, cv2.resize(np.hstack([image, overlay]), (1920, 540)))
    print(f"size x{c:.3f} of the released cloud; outline overlap (Dice) {dice:.3f}; image saved to {out}")

    if args.no_window:
        return
    u = np.clip(np.round(K[0, 0] * cloud[:, 0] / cloud[:, 2] + K[0, 2]).astype(int), 0, mask.shape[1] - 1)
    v = np.clip(np.round(K[1, 1] * cloud[:, 1] / cloud[:, 2] + K[1, 2]).astype(int), 0, mask.shape[0] - 1)
    patch = points(cloud @ (c * Q).T + d, BLUE)
    patch.colors = o3d.utility.Vector3dVector(image[v, u, ::-1] / 255.0)  # each point in its pixel's colour
    o3d.visualization.draw_geometries(
        [P["target"].paint_uniform_color(GREY), patch, ball(d, BLUE)],
        window_name=f"real frame {args.frame}: size x{c:.3f}, outline Dice {dice:.3f} (no true pose)")


if __name__ == "__main__":
    main()
