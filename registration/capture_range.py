"""
Capture range of module 3 (sim3_icp.py) on clean synthetic training views.

Start from the true pose pushed off by a known amount, run module 3, and check whether it gets
back. Success: mean error at the patient's 200 probe tumours below 5 mm. Each error is tested
on its own (rotation, shift, size), then all three together.

The starting guess, applied to the true camera-to-CT map:
  size     the cloud is k times too big about the camera centre (as a wrong depth scale would be)
  rotation by `deg` about a random axis through the cloud centre
  shift    by `mm` in a random direction
Only training views are used (train_syn.txt); views whose cloud does not fit the CT at the true
pose (the extra object in patients 10 and 11) are skipped.

Usage: python registration/capture_range.py [--views-per-patient 20] [--tests size ...]
"""
import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "benchmark"))
from pilot_dice_vs_tumour import LIVER, load_patient, load_view, sample_tumours, surface_dists, true_w2c  # noqa: E402
from sim3_icp import register, rotation  # noqa: E402

PATIENTS = ["01", "06", "10", "11"]
N_PROBES = 200
SUCCESS_MM = 5.0
TESTS = {  # name -> list of (rotation deg, shift mm, size factor)
    "rotation": [(deg, 0, 1.0) for deg in (5, 10, 20, 30, 45, 60, 90)],
    "shift": [(0, mm, 1.0) for mm in (5, 10, 20, 30, 50)],
    "size": [(0, 0, k) for k in (0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4, 1.6, 1.8, 2.0, 2.5)],
    "together": [(5, 5, 1.05), (10, 10, 1.1), (20, 15, 1.15), (30, 25, 1.2), (45, 35, 1.3)],
}


def unit(rng) -> np.ndarray:
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)


def training_views(P, rng, n: int):
    """n random training views whose cloud fits the CT at the true pose, with rays through liver pixels.

    Every k-th liver pixel is used, so there are about as many rays as camera points and the two
    matching directions weigh equally.
    """
    with open(os.path.join(P["root"], "train_syn.txt")) as f:
        views = [int(line.strip().replace("ptsLiverCam", "")) for line in f if line.strip()]
    chosen = []
    for view in rng.permutation(views):
        label, cloud = load_view(P, view)
        T = true_w2c(P["poses"][f"{view:05d}"][0])
        if len(cloud) >= 100 and (surface_dists(T, P, cloud) > 3.0).mean() <= 0.02:
            rays = P["pix_dirs"][(label == LIVER).reshape(-1)]
            chosen.append((view, cloud, rays[::max(1, len(rays) // len(cloud))], T))
        if len(chosen) == n:
            return chosen
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--views-per-patient", type=int, default=20)
    ap.add_argument("--tests", nargs="+", choices=list(TESTS), default=list(TESTS))
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    args = ap.parse_args()
    tests = {name: TESTS[name] for name in args.tests}
    os.makedirs(args.out, exist_ok=True)

    rows = []
    for pid in PATIENTS:
        P = load_patient(pid)
        probes, _ = sample_tumours(P, np.random.default_rng([0, int(pid)]), N_PROBES)
        rng = np.random.default_rng([1, int(pid)])
        for view, cloud, rays, T in training_views(P, rng, args.views_per_patient):
            Q_true, d_true = T[:3, :3].T, -T[:3, :3].T @ T[:3, 3]  # camera -> CT
            m = Q_true @ cloud.mean(axis=0) + d_true                # cloud centre in the CT frame
            cam_true = (probes - d_true) @ Q_true                   # probes in the true camera frame
            for test, levels in tests.items():
                for deg, mm, k in levels:
                    if test == "together" and rng.random() < 0.5:
                        k = 1 / k                                   # too small as often as too big
                    dR = rotation(unit(rng), deg)
                    c, Q, d = register(cloud, rays, P["scene"], k, dR @ Q_true,
                                       dR @ (d_true - m) + m + mm * unit(rng))
                    tre = np.linalg.norm((probes - d) @ Q - cam_true, axis=1)
                    rows.append(dict(test=test, rotation_deg=deg, shift_mm=mm, size=round(k, 4),
                                     patient=pid, view=view, mean_tre_mm=tre.mean(),
                                     rotation_error_deg=np.rad2deg(np.arccos(np.clip(
                                         (np.trace(Q @ Q_true.T) - 1) / 2, -1, 1))),
                                     scale_error_pct=100 * (c - 1)))
            print(f"[{pid}] view {view:05d} done", flush=True)

    name = "capture_range" + ("" if len(tests) == len(TESTS) else "_" + "_".join(tests))
    with open(os.path.join(args.out, name + ".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    for test, levels in tests.items():
        print(f"\n{test}: start error -> success (mean probe error < {SUCCESS_MM:.0f} mm), median error")
        for deg, mm, k in levels:
            sel = [r for r in rows if r["test"] == test and r["rotation_deg"] == deg and r["shift_mm"] == mm
                   and r["size"] in ((k, round(1 / k, 4)) if test == "together" else (k,))]
            tre = np.array([r["mean_tre_mm"] for r in sel])
            print(f"  {deg:>2} deg  {mm:>2} mm  x{k:<4}  {100 * (tre < SUCCESS_MM).mean():5.1f} %   "
                  f"{np.median(tre):6.1f} mm   ({len(sel)} runs)")


if __name__ == "__main__":
    main()
