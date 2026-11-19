"""
Score Self-P2IR's LLR-LUS predictions at the tumour, next to its own Dice.

selfp2ir_real.py gives, per frame, the rigid pose (R, t) that maps the CT (metres) onto the frame's cloud
as their loader sees it: Blender camera axes, metres, after the cloud was shifted (lus_selfp2ir_prep.py).
Undoing both puts the CT, and so the tumour, in the camera frame the cloud claims (OpenCV axes, mm):
    X_cam = 1000 * F (R X_ct + t) + shift,   F = diag(1, -1, -1)
Only the rigid pose places the tumour: their bending step moves sampled surface points, not the tumour.

Writes one tumour per image (OpenGL camera axes, as LLR-LUS expects) under
dataset/lus-evaluation/selfp2ir_tumours/PatientN/NN.stl, then:
  - the official score (lus_evaluation.py): TRE per patient (one ICP per patient), inclusion test;
  - per image: their Dice, and the mean distance from the ultrasound tumour profile to the predicted tumour
    (no ICP), split into depth (along the camera's viewing axis) and across it.

Usage: python studies/benchmark/lus_selfp2ir_score.py --results <selfp2ir_real.py output json>
"""
import argparse
import json
import os
import sys

import numpy as np
import open3d as o3d

import lus_evaluation as ev

LUS = "dataset/lus-evaluation"
F = np.diag([1.0, -1.0, -1.0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", default=f"{LUS}/selfp2ir_tumours")
    args = ap.parse_args()

    run_meta = os.path.join(os.path.dirname(args.results), "meta.json")     # a run with its own scales (oracle)
    meta = json.load(open(run_meta if os.path.exists(run_meta) else f"{LUS}/selfp2ir/meta.json"))
    by_frame = {(f"{r['patient']:02d}", r["frame"]): r for r in json.load(open(args.results))["frames"]}
    off_path = os.path.join(os.path.dirname(args.results), "ct_offsets.json")   # run on CTs moved to the origin
    offsets = json.load(open(off_path)) if os.path.exists(off_path) else {}
    if offsets:
        print("CT livers were moved to the origin for this run; the tumour is moved the same way")
    rows = []
    for key, m in sorted(meta.items()):
        p, name = key.split("/")
        r = by_frame.get((m["id"], m["frame"]))
        if r is None:
            print(f"{key}: no prediction")
            continue
        tumour = o3d.io.read_triangle_mesh(f"{LUS}/data/Dataset/{p}/Preoperative Model/Tumour.ply")
        V = (np.asarray(tumour.vertices) - np.array(offsets.get(m["id"], [0.0, 0.0, 0.0]))) / 1000.0
        cam = 1000.0 * (V @ np.array(r["R"]).T + np.array(r["t"])) @ F.T + np.array(m["shift_mm"])
        tumour.vertices = o3d.utility.Vector3dVector(cam @ F.T)              # OpenCV -> OpenGL camera axes
        tumour.compute_triangle_normals()
        os.makedirs(f"{args.out}/{p}", exist_ok=True)
        o3d.io.write_triangle_mesh(f"{args.out}/{p}/{name}.stl", tumour)

        prof = ev.profile(ev.DATASET, p[len("Patient"):], name)               # 3 x n, camera frame, mm
        near = cam[((prof.T[:, None] - cam[None]) ** 2).sum(2).argmin(1)]
        diff = near - prof.T
        rows.append(dict(image=key, dice=r["dice"], dist=float(np.linalg.norm(diff, axis=1).mean()),
                         depth=float(diff[:, 2].mean()), across=float(np.linalg.norm(diff[:, :2], axis=1).mean()),
                         true_depth=float(prof[2].mean()), pred_depth=float(cam[:, 2].mean())))

    print(f"{'image':13s} {'Dice %':>6s} {'profile->tumour':>15s} {'depth error':>11s} {'across':>7s} "
          f"{'true depth':>10s} {'predicted':>9s}  (mm)")
    for x in rows:
        print(f"{x['image']:13s} {x['dice']:6.1f} {x['dist']:15.1f} {x['depth']:+11.1f} {x['across']:7.1f} "
              f"{x['true_depth']:10.0f} {x['pred_depth']:9.0f}")
    d, e = np.array([x["dice"] for x in rows]), np.array([x["dist"] for x in rows])
    print(f"\nall {len(rows)} images: Dice {d.mean():.1f} +- {d.std():.1f} %, profile->tumour {np.median(e):.1f} mm (median), "
          f"Spearman(Dice, distance) {np.corrcoef(np.argsort(np.argsort(d)), np.argsort(np.argsort(e)))[0, 1]:+.2f}")
    print("\nOfficial LLR-LUS score (lus_evaluation.py):")
    sys.argv = ["lus_evaluation.py", "--method-dir", args.out]
    ev.main()


if __name__ == "__main__":
    main()
