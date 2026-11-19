"""
Show modules 2 + 3 working together on one synthetic view, from scratch (no knowledge of the true pose).

Three Open3D windows, one after the other (close one to open the next):
  1 starts      orange balls = cameras of module 2's starting guesses, all around the liver
  2 candidates  orange = the 10 candidates refined by module 3, blue = the one picked (best fit)
  3 answer      blue = the answer of modules 2 + 3
Grey is the CT liver; green is the true camera cloud and the true camera. Drag to rotate, scroll to zoom.
Takes about a minute before window 2 opens.

Usage: python registration/show_module2.py [--patient 01] [--view 1124]
"""
import argparse

import numpy as np
import open3d as o3d

from capture_range import LIVER, load_patient, load_view, sample_tumours, true_w2c
from multistart import N_DIRS, N_ROLLS, START_SIZE, candidates, register_from_scratch
from show_module3 import BLUE, GREEN, GREY, ball, points

ORANGE = [1.0, 0.6, 0.1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", default="01")
    ap.add_argument("--view", type=int, default=1124, help="a training view (see train_syn.txt)")
    args = ap.parse_args()

    P = load_patient(args.patient)
    probes, _ = sample_tumours(P, np.random.default_rng([0, int(args.patient)]), 200)
    label, cloud = load_view(P, args.view)
    T = true_w2c(P["poses"][f"{args.view:05d}"][0])
    rays = P["pix_dirs"][(label == LIVER).reshape(-1)]
    rays = rays[::max(1, len(rays) // len(cloud))]

    Q_true, d_true = T[:3, :3].T, -T[:3, :3].T @ T[:3, 3]
    cam_true = (probes - d_true) @ Q_true
    tre = lambda p: np.linalg.norm((probes - p[2]) @ p[1] - cam_true, axis=1).mean()  # noqa: E731
    patch = lambda p: cloud @ (p[0] * p[1]).T + p[2]  # noqa: E731
    base = [P["target"].paint_uniform_color(GREY), points(cloud @ Q_true.T + d_true, GREEN), ball(d_true, GREEN)]

    starts = candidates(cloud, P["scene"], P["centroid"], START_SIZE, N_DIRS, N_ROLLS)
    print(f"module 2: {len(starts)} starting guesses ({N_DIRS} directions x {N_ROLLS} rolls, size x{START_SIZE})")
    o3d.visualization.draw_geometries(base + [ball(d, ORANGE, 3.0) for _, _, d in starts],
                                      window_name=f"1/3 module 2: {len(starts)} starting cameras around the liver")

    best, finals, errors = register_from_scratch(cloud, rays, P["scene"], P["centroid"])
    print("10 candidates after module 3 (fit in camera units; picked = smallest fit):")
    for p, e in sorted(zip(finals, errors), key=lambda x: x[1]):
        print(f"  size x{p[0]:.3f}  fit {e:5.2f}  tumour error {tre(p):6.1f} mm{'   <- picked' if p is best else ''}")
    o3d.visualization.draw_geometries(
        base + [g for p in finals for g in (points(patch(p), BLUE if p is best else ORANGE),
                                            ball(p[2], BLUE if p is best else ORANGE, 3.0))],
        window_name="2/3 module 3 refined 10 candidates: blue = picked, orange = rejected")

    o3d.visualization.draw_geometries(base + [points(patch(best), BLUE), ball(best[2], BLUE)],
                                      window_name=f"3/3 answer: size x{best[0]:.3f}, tumour error {tre(best):.1f} mm")


if __name__ == "__main__":
    main()
