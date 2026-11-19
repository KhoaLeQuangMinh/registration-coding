"""
See one LLR-LUS image as Self-P2IR registered it, in the camera's frame (mm), with the tumour error.

Open3D window (drag to rotate, scroll to zoom; the camera sits at the small axes, looking along +z):
  coloured points   the frame's cloud (liver pixels, coloured by the photo): what Self-P2IR registered to
  grey wireframe    the CT liver placed by Self-P2IR's rigid pose
  blue points       with --bent: Self-P2IR's bent CT (its deformation step), the shape its Dice is measured on
  red mesh          the CT tumour placed by the rigid pose: Self-P2IR's tumour prediction
  green points      the TRUE tumour: its outline in the ultrasound image, placed by the tracked probe
  yellow lines      from each true tumour point to the nearest predicted tumour point (the error)
  orange mesh       with --compare koo: Koo et al.'s published tumour for the same image
  dark grey lines   the camera's view, drawn out to 50 mm
Also saves dataset/lus-evaluation/selfp2ir_views/<Patient>_<NN>.png: the photo with your liver mask (white)
and Self-P2IR's rendered liver (blue), the overlap its Dice measures.

Inputs: dataset/lus-evaluation/selfp2ir/ (lus_selfp2ir_prep.py) and the Kaggle results zip unzipped into
dataset/lus-evaluation/selfp2ir_results/ (it holds lus_results/real_frames.json and lus_results/points/).

Usage: python studies/benchmark/show_lus_selfp2ir.py --image Patient1/02 [--compare koo] [--bent] [--no-window]
"""
import argparse
import json
import os

import cv2
import numpy as np
import open3d as o3d

import lus_evaluation as ev

LUS = "dataset/lus-evaluation"
F = np.diag([1.0, -1.0, -1.0])  # Blender <-> OpenCV camera axes


def to_cam(X_their, shift):
    """Their frame (Blender axes, metres, after the cloud's shift) -> camera frame (OpenCV axes, mm)."""
    return 1000.0 * X_their @ F.T + shift


def lines(P, Q, colour):
    ls = o3d.geometry.LineSet(o3d.utility.Vector3dVector(np.vstack([P, Q])),
                              o3d.utility.Vector2iVector([[i, i + len(P)] for i in range(len(P))]))
    ls.paint_uniform_color(colour)
    return ls


def points(P, colour=None, colours=None):
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    if colours is not None:
        pc.colors = o3d.utility.Vector3dVector(colours)
    else:
        pc.paint_uniform_color(colour)
    return pc


def frustum(fx, fy, cx, cy, W, H, depth=50.0):
    c = np.array([[0, 0], [W, 0], [W, H], [0, H]], float)
    far = np.stack([(c[:, 0] - cx) / fx, (c[:, 1] - cy) / fy, np.ones(4)], 1) * depth
    P = np.vstack([[0, 0, 0], far])
    ls = o3d.geometry.LineSet(o3d.utility.Vector3dVector(P),
                              o3d.utility.Vector2iVector([[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]))
    ls.paint_uniform_color([0.25, 0.25, 0.25])
    return ls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="e.g. Patient1/02")
    ap.add_argument("--results", default=f"{LUS}/selfp2ir_results/lus_results",
                    help="folder with real_frames.json and points/ (the Kaggle zip's lus_results/)")
    ap.add_argument("--compare", choices=["koo", "adagolodjo"], default=None)
    ap.add_argument("--bent", action="store_true", help="also show Self-P2IR's bent CT (blue); it hides the red tumour")
    ap.add_argument("--no-window", action="store_true")
    args = ap.parse_args()

    local = json.load(open(f"{LUS}/selfp2ir/meta.json"))[args.image]
    run_meta = f"{args.results}/meta.json"                      # a run with its own scales (oracle)
    m = json.load(open(run_meta))[args.image] if os.path.exists(run_meta) else local
    p, name = args.image.split("/")
    shift = np.array(m["shift_mm"])
    z = np.load(f"{args.results}/points/{m['id']}_{m['frame']}.npz")
    R, t = z["R"], z["t"].reshape(3)
    rec = {(f"{r['patient']:02d}", r["frame"]): r for r in json.load(open(f"{args.results}/real_frames.json"))["frames"]}
    dice = rec[(m["id"], m["frame"])]["dice"]

    # the frame's cloud (liver pixels), coloured by the photo
    img = cv2.imread(f"{LUS}/data/Dataset/{p}/Lap Images/{name}.png")
    liver = cv2.imread(f"{LUS}/masks_manual/{p}/{name}.png", 0) > 0
    P = np.asarray(o3d.io.read_point_cloud(f"{LUS}/selfp2ir/{m['id']}/real/liverPcds/{m['frame']}.ply").points)
    P = (P + np.array(local["shift_mm"])) * (m["sigma"] / local["sigma"])   # camera frame, at this run's scale
    keep = np.flatnonzero(liver.reshape(-1))[::7]
    cloud = points(P[keep], colours=cv2.cvtColor(img, cv2.COLOR_BGR2RGB).reshape(-1, 3)[keep] / 255.0)

    # CT liver and tumour placed by the rigid pose (moved to the origin first if the run used centred CTs)
    off_path = f"{args.results}/ct_offsets.json"
    off = np.array(json.load(open(off_path))[m["id"]]) if os.path.exists(off_path) else np.zeros(3)
    place = lambda V_mm: to_cam((V_mm - off) / 1000.0 @ R.T + t, shift)  # noqa: E731
    liver_mesh = o3d.io.read_triangle_mesh(f"{LUS}/data/Dataset/{p}/Preoperative Model/Liver.ply")
    liver_mesh = liver_mesh.simplify_quadric_decimation(6000)
    liver_mesh.vertices = o3d.utility.Vector3dVector(place(np.asarray(liver_mesh.vertices)))
    wire = o3d.geometry.LineSet.create_from_triangle_mesh(liver_mesh)
    wire.paint_uniform_color([0.6, 0.6, 0.6])
    tumour = o3d.io.read_triangle_mesh(f"{LUS}/data/Dataset/{p}/Preoperative Model/Tumour.ply")
    tumour.vertices = o3d.utility.Vector3dVector(place(np.asarray(tumour.vertices)))
    tumour.compute_vertex_normals()
    tumour.paint_uniform_color([0.9, 0.15, 0.15])
    bent = points(to_cam(z["bent"], shift), [0.2, 0.45, 1.0])

    # the true tumour (ultrasound outline in the camera frame) and the error to the prediction
    prof = ev.sample_profile(ev.profile(ev.DATASET, p[len("Patient"):], name), ev.M_SAMPLES).T
    T = np.asarray(tumour.vertices)
    near = T[((prof[:, None] - T[None]) ** 2).sum(2).argmin(1)]
    diff = near - prof
    true_tumour = points(prof, [0.1, 0.75, 0.2])
    geoms = [cloud, wire, tumour, true_tumour, lines(prof, near, [1.0, 0.75, 0.0]),
             frustum(m["fx"], m["fy"], m["cx"], m["cy"], *m["size"]),
             o3d.geometry.TriangleMesh.create_coordinate_frame(size=15.0)]
    if args.bent:
        geoms.append(bent)
    if args.compare:
        V, Fc = ev.read_stl(f"{LUS}/results/Registration Methods/{args.compare.capitalize()}/{p}/{name}.stl")
        other = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V * [1, -1, -1]), o3d.utility.Vector3iVector(Fc))
        other.compute_vertex_normals()
        other.paint_uniform_color([1.0, 0.55, 0.1])
        geoms.append(other)

    print(f"{args.image} (Self-P2IR id {m['id']}, {m['frame']})")
    print(f"  Dice (their rendered liver vs your mask): {dice:.1f} %")
    print(f"  true tumour -> predicted tumour: mean {np.linalg.norm(diff, axis=1).mean():.1f} mm "
          f"(depth {diff[:, 2].mean():+.1f} mm, across {np.linalg.norm(diff[:, :2], axis=1).mean():.1f} mm)")
    print(f"  true tumour depth {prof[:, 2].mean():.0f} mm, predicted {T[:, 2].mean():.0f} mm; "
          f"cloud depth {np.percentile(P[keep, 2], [5, 50, 95]).round().astype(int).tolist()} mm")

    # 2D: your mask vs their rendered liver
    rm = np.unpackbits(z["rendered_mask"])[:np.prod(z["mask_shape"])].reshape(z["mask_shape"]).astype(bool)
    view = img.copy()
    view[rm] = (0.55 * view[rm] + 0.45 * np.array([255, 120, 0])).astype(np.uint8)
    cv2.drawContours(view, cv2.findContours(liver.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0],
                     -1, (255, 255, 255), max(2, view.shape[1] // 640))
    os.makedirs(f"{LUS}/selfp2ir_views", exist_ok=True)
    out = f"{LUS}/selfp2ir_views/{p}_{name}.png"
    cv2.imwrite(out, cv2.resize(view, (1280, round(1280 * view.shape[0] / view.shape[1]))))
    print(f"  2D view (your mask white, their rendered liver blue): {out}")

    if not args.no_window:
        o3d.visualization.draw_geometries(geoms, window_name=f"Self-P2IR on LLR-LUS {args.image}",
                                          lookat=P[keep].mean(0), front=[0, 0, -1], up=[0, -1, 0], zoom=0.7)


if __name__ == "__main__":
    main()
